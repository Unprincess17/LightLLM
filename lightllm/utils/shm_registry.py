import ctypes
import json
import os
from typing import Any, Dict, List, Optional

import psutil
from filelock import FileLock
from multiprocessing import shared_memory

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_REGISTRY_PREFIX = "/tmp/lightllm_shm_registry_"
_IPC_RMID = 0


def _get_service_name() -> Optional[str]:
    return os.getenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")


def _get_registry_path() -> Optional[str]:
    service_name = _get_service_name()
    if service_name is None:
        return None
    return f"{_REGISTRY_PREFIX}{service_name}.json"


def _load_registry(path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return {"entries": []}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except Exception as e:
        logger.warning(f"failed to load shm registry {path}: {e}")
    return {"entries": []}


def _save_registry(path: str, data: Dict[str, List[Dict[str, Any]]]) -> None:
    entries = data.get("entries", [])
    if not entries:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2, sort_keys=True)


def _upsert_entry(match_key: str, match_value: Any, new_entry: Dict[str, Any]) -> None:
    path = _get_registry_path()
    if path is None:
        return

    lock = FileLock(f"{path}.lock")
    with lock:
        data = _load_registry(path)
        entries = [entry for entry in data["entries"] if entry.get(match_key) != match_value]
        entries.append(new_entry)
        data["entries"] = entries
        _save_registry(path, data)


def _remove_entry(predicate) -> None:
    path = _get_registry_path()
    if path is None:
        return

    lock = FileLock(f"{path}.lock")
    with lock:
        data = _load_registry(path)
        data["entries"] = [entry for entry in data["entries"] if not predicate(entry)]
        _save_registry(path, data)


def register_posix_shm(name: str, creator_pid: Optional[int] = None) -> None:
    _upsert_entry(
        match_key="registry_key",
        match_value=f"posix:{name}",
        new_entry={
            "registry_key": f"posix:{name}",
            "kind": "posix",
            "name": name,
            "creator_pid": os.getpid() if creator_pid is None else creator_pid,
        },
    )


def unregister_posix_shm(name: str) -> None:
    _remove_entry(lambda entry: entry.get("kind") == "posix" and entry.get("name") == name)


def register_sysv_shm(key: int, shmid: int, creator_pid: Optional[int] = None) -> None:
    _upsert_entry(
        match_key="registry_key",
        match_value=f"sysv:{key}",
        new_entry={
            "registry_key": f"sysv:{key}",
            "kind": "sysv",
            "key": key,
            "shmid": shmid,
            "creator_pid": os.getpid() if creator_pid is None else creator_pid,
        },
    )


def unregister_sysv_shm(key: Optional[int] = None, shmid: Optional[int] = None) -> None:
    def _predicate(entry: Dict[str, Any]) -> bool:
        if entry.get("kind") != "sysv":
            return False
        if key is not None and entry.get("key") == key:
            return True
        if shmid is not None and entry.get("shmid") == shmid:
            return True
        return False

    _remove_entry(_predicate)


def cleanup_stale_registered_shm() -> None:
    path = _get_registry_path()
    if path is None:
        return

    lock = FileLock(f"{path}.lock")
    with lock:
        data = _load_registry(path)
        if not data["entries"]:
            return

        libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")
        libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
        libc.shmget.restype = ctypes.c_int
        libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        libc.shmctl.restype = ctypes.c_int

        keep_entries = []
        for entry in data["entries"]:
            creator_pid = entry.get("creator_pid")
            if creator_pid is not None and psutil.pid_exists(creator_pid):
                keep_entries.append(entry)
                continue

            kind = entry.get("kind")
            try:
                if kind == "posix":
                    name = entry["name"]
                    try:
                        shm = shared_memory.SharedMemory(name=name, create=False)
                    except FileNotFoundError:
                        shm = None
                    if shm is not None:
                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            pass
                        finally:
                            shm.close()
                    logger.info(f"startup cleanup removed stale POSIX shm {name}")
                elif kind == "sysv":
                    shmid = int(entry.get("shmid", -1))
                    if shmid >= 0 and libc.shmctl(shmid, _IPC_RMID, None) != 0:
                        key = int(entry.get("key", -1))
                        if key >= 0:
                            shmid = libc.shmget(key, 0, 0)
                            if shmid >= 0:
                                libc.shmctl(shmid, _IPC_RMID, None)
                    logger.info(
                        f"startup cleanup removed stale System V shm key={entry.get('key')} shmid={entry.get('shmid')}"
                    )
                else:
                    keep_entries.append(entry)
            except Exception as e:
                logger.warning(f"failed to cleanup stale shm entry {entry}: {e}")

        data["entries"] = keep_entries
        _save_registry(path, data)
