import torch
import ctypes
import dataclasses
import errno
import os
import xxhash
import threading
import time
import numpy as np
import triton
from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args, enable_huge_page, get_llm_data_type
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_num_key_value_heads, get_head_dim, get_layer_num
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.common.kv_cache_mem_manager import (
    MemoryManager,
    INT8KVMemoryManager,
    CalibrationFP8KVMemoryManager,
    ExportCalibrationMemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    Deepseek2MemoryManager,
    Deepseek2FP8KVMemoryManager,
)

from typing import List, Tuple, Optional
from tqdm import tqdm
from lightllm.utils.auto_shm_cleanup import register_cleanup_callback, register_sysv_shm_for_cleanup
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.shm_registry import register_sysv_shm, unregister_sysv_shm

logger = init_logger(__name__)


def compute_token_list_hash(tokens: List[int], cpu_cache_token_page_size: int) -> List[int]:
    if len(tokens) == 0:
        return []

    chunks_hash_value = []
    hsum = xxhash.xxh3_128()

    # 计算每个分块的哈希值, 但是输入token需要少一个，因为
    # 如果计算所有的token，会导致输入input_len 命中全长的
    # cpu cache, 导致prefill 过程无法有输入来导出下一个输出。
    calcu_num = (len(tokens) - 1) // cpu_cache_token_page_size

    for i in range(calcu_num):
        start_index = i * cpu_cache_token_page_size
        end_index = (i + 1) * cpu_cache_token_page_size
        chunk = tokens[start_index:end_index]
        chunk_np = np.array(chunk, dtype=np.uint64)
        hsum.update(chunk_np.tobytes())
        hash_value = hsum.intdigest()
        chunks_hash_value.append(hash_value)

    return chunks_hash_value


@lru_cache(maxsize=None)
def calcu_cpu_cache_meta() -> "CpuKVCacheMeta":
    args = get_env_start_args()
    assert args.enable_cpu_cache

    mem_manager_class = select_mem_manager_class()
    if mem_manager_class is Deepseek2MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=512 + 64,
            data_type=get_llm_data_type(),
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is Deepseek2FP8KVMemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=512 + 64 + 2,
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=get_num_key_value_heads(args.model_dir) * 2,
            head_dim=get_head_dim(args.model_dir),
            data_type=get_llm_data_type(),
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is PPLINT8KVMemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=get_num_key_value_heads(args.model_dir) * 2,
            head_dim=get_head_dim(args.model_dir),
            data_type=torch.int8,
            scale_head_dim=get_head_dim(args.model_dir) // 8,
            scale_data_type=get_llm_data_type(),
        )
    else:
        logger.error(f"not support mem manager: {mem_manager_class} for cpu kv cache")
        raise Exception(f"not support mem manager: {mem_manager_class} for cpu kv cache")

    if args.mtp_mode is not None:
        # TODO 可能会存在不同mtp模式的精度问题
        cpu_cache_meta.layer_num += 1

    cpu_cache_page_num = int(
        (args.cpu_cache_storage_size * 1024 * 1024 * 1024) / (cpu_cache_meta.calcu_one_page_size())
    )
    cpu_cache_meta.page_num = cpu_cache_page_num

    logger.info(f"cpu kv cache page num: {cpu_cache_meta.page_num}")

    return cpu_cache_meta


@dataclasses.dataclass
class CpuKVCacheMeta:
    page_num: int
    token_page_size: int
    layer_num: int
    num_heads: int
    head_dim: int
    data_type: torch.dtype
    scale_head_dim: int
    scale_data_type: torch.dtype

    def calcu_size(self):
        return self.page_num * self.calcu_one_page_size()

    def calcu_one_page_size(self):
        return (
            self.token_page_size
            * self.layer_num
            * self.num_heads
            * (self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize)
        )

    def get_merged_head_dim(self):
        """
        返回将head_dim 和 scale_head_dim 看成融合成一个head_dim时候, head_dim的长度。
        """
        assert (
            self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize
        ) % self.data_type.itemsize == 0
        return (
            self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize
        ) // self.data_type.itemsize


def _get_sysv_libc():
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p
    libc.shmdt.argtypes = (ctypes.c_void_p,)
    libc.shmdt.restype = ctypes.c_int
    libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    libc.shmctl.restype = ctypes.c_int
    return libc


@dataclasses.dataclass
class SystemVShmHandle:
    key: int
    shmid: int
    shm_addr: int
    size: int
    is_owner: bool
    pin_handle: Optional["AsyncRegistrationHandle"] = None
    _closed: bool = False
    _destroyed: bool = False

    def bind_pin_handle(self, pin_handle: Optional["AsyncRegistrationHandle"]) -> None:
        self.pin_handle = pin_handle

    def close(self) -> None:
        if self._closed:
            return

        if self.pin_handle is not None:
            try:
                self.pin_handle.unregister()
            except Exception as e:
                logger.warning(f"failed to unregister pinned host memory for shm key={self.key}: {e}")
            self.pin_handle = None

        if self.shm_addr:
            libc = _get_sysv_libc()
            ret = libc.shmdt(ctypes.c_void_p(self.shm_addr))
            if ret != 0:
                err = ctypes.get_errno()
                logger.warning(f"shmdt failed for shm key={self.key}, shmid={self.shmid}, errno={err}")
            self.shm_addr = 0

        self._closed = True

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.close()
        if not self.is_owner:
            return

        unregister_sysv_shm(key=self.key, shmid=self.shmid)
        libc = _get_sysv_libc()
        ret = libc.shmctl(self.shmid, 0, None)
        if ret != 0:
            err = ctypes.get_errno()
            logger.warning(f"IPC_RMID failed for shm key={self.key}, shmid={self.shmid}, errno={err}")


def _create_or_attach_shm_handle(key: int, size: int, create: bool) -> SystemVShmHandle:
    libc = _get_sysv_libc()

    requested_size = size
    use_hugetlb = enable_huge_page()

    # 计算大页大小（默认从 /proc/meminfo 读取 Hugepagesize）
    def _get_default_hugepage_size() -> int:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("Hugepagesize:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return kb * 1024
        except Exception:
            pass
        return 2 * 1024 * 1024  # fallback 2MB

    IPC_CREAT = 0o1000
    IPC_EXCL = 0o2000
    shmflg = 0o666 | IPC_CREAT  # 权限和 IPC_CREAT 标志
    if use_hugetlb:
        # 向上对齐到大页大小
        huge_sz = _get_default_hugepage_size()
        size_to_alloc = triton.cdiv(requested_size, huge_sz) * huge_sz
        SHM_HUGETLB = 0o4000
        shmflg |= SHM_HUGETLB
        logger.info(
            f"Using SHM_HUGETLB, hugepage_size={huge_sz} bytes, requested={requested_size}, alloc={size_to_alloc}"
        )
    else:
        size_to_alloc = requested_size
        logger.info(f"Using regular pages, requested={requested_size}, alloc={size_to_alloc}")

    is_owner = False
    if create:
        shmid = libc.shmget(key, size_to_alloc, shmflg | IPC_EXCL)
        hugepages_num = (size_to_alloc + 1024 * 1024 * 1024 - 1) // (1024 * 1024 * 1024)
        if shmid < 0:
            err = ctypes.get_errno()
            if err == errno.EEXIST:
                create = False
                shmid = libc.shmget(key, 0, 0)
                if shmid < 0:
                    err = ctypes.get_errno()
                    raise Exception(f"Error locating existing shared memory (errno={err})")
            else:
                if use_hugetlb:
                    raise Exception(
                        f"shmget with SHM_HUGETLB failed (errno={err}). Falling back to regular pages."
                        f"You may need to configure hugepages manually, e.g.,"
                        f"sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"/& default_hugepagesz=1G \
                            hugepagesz=1G hugepages={hugepages_num}/' /etc/default/grub"
                        f"sudo update-grub"
                        f"sudo reboot"
                    )
                raise Exception(f"Error creating regular shared memory (errno={err})")
        else:
            is_owner = True
            register_sysv_shm_for_cleanup(key, shmid)
            register_sysv_shm(key, shmid)
    else:
        shmid = libc.shmget(key, 0, 0)
        if shmid < 0:
            shmid = libc.shmget(key, size, 0)
        if shmid < 0:
            err = ctypes.get_errno()
            raise Exception(f"Error locating existing shared memory (errno={err})")

    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)
    if shm_addr == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        raise Exception(f"Error attaching shared memory (errno={err})")

    handle = SystemVShmHandle(key=key, shmid=shmid, shm_addr=shm_addr, size=size, is_owner=is_owner)
    register_cleanup_callback(handle.destroy if is_owner else handle.close)
    logger.info(f"Attached to SHM key={key}, shmid={shmid}, addr={shm_addr}, owner={is_owner}")

    if is_owner:
        def _pre_warm_memory():
            page_size = _get_default_hugepage_size() if use_hugetlb else 4096
            arr = np.ctypeslib.as_array(ctypes.cast(shm_addr, ctypes.POINTER(ctypes.c_uint8)), shape=(size_to_alloc,))
            volatile_sum = int(arr[::page_size].sum())
            logger.info(f"pre warmed shared memory pages successfully, checksum={volatile_sum})")

        th = threading.Thread(target=_pre_warm_memory, name=f"cpu_cache_pre_warm_{key}", daemon=True)
        th.start()

    return handle


def create_shm_kv_cache_handle(key: int, size: int) -> SystemVShmHandle:
    return _create_or_attach_shm_handle(key=key, size=size, create=True)


def attach_shm_kv_cache_handle(key: int, size: int) -> SystemVShmHandle:
    return _create_or_attach_shm_handle(key=key, size=size, create=False)


def create_shm_kv_cache_ptr(key: int, size: int) -> int:
    return create_shm_kv_cache_handle(key=key, size=size).shm_addr


def register_shm_ptr_to_pin(shm_ptr: int, size: int) -> "AsyncRegistrationHandle":
    """Start async cudaHostRegister on the given [shm_ptr, shm_ptr+size) and return a handle."""
    chunk_bytes = 128 * 1024 * 1024  # 128M性能最好
    tasks: list[tuple[int, int]] = []
    offset = 0
    while offset < size:
        seg_len = min(chunk_bytes, size - offset)
        tasks.append((offset, seg_len))
        offset += seg_len

    handle = AsyncRegistrationHandle(total_tasks=len(tasks), shm_ptr=shm_ptr, size=size, tasks=tasks)

    def _worker():
        try:
            cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
            cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            cuda.cudaHostRegister.restype = ctypes.c_int
            cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
            cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

            cudaHostRegisterFlag = 3

            torch.cuda.set_device(get_current_device_id())
            # TODO 这个地方的分块注册是否具备合法性和合理性。
            for offset, seg_len in tasks:
                ptr = ctypes.c_void_p(shm_ptr + offset)
                r = cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg_len), cudaHostRegisterFlag)
                if r != 0:
                    raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")
                handle.task_count += 1

            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"cudaHostGetDevicePointer failed with error code {res}")
            assert host_ptr.value == device_ptr.value
            handle.registered = True
        except Exception as e:
            handle.error = e
        finally:
            handle.tasks_finished.set()

    th = threading.Thread(target=_worker, name=f"cpu_cache_register_{shm_ptr}", daemon=True)
    handle.thread = th
    th.start()
    return handle


class AsyncRegistrationHandle:
    """A handle for async host memory registration.

    - wait(): blocks until registration finishes, prints tqdm progress, and returns device pointer (int).
    """

    def __init__(self, total_tasks: int, shm_ptr: int, size: int, tasks: list[tuple[int, int]]):
        self.total_tasks = total_tasks
        self.task_count = 0
        self.thread: Optional[threading.Thread] = None
        self.tasks_finished = threading.Event()
        self.shm_ptr = shm_ptr
        self.size = size
        self.tasks = tasks
        self.error: Optional[Exception] = None
        self.registered = False
        self.unregistered = False

    def wait(self):
        """Block until the async registration completes. Only here we print tqdm progress."""
        last_count = 0
        desc = f"pid {os.getpid()} Registering pinned host memory (async)"
        with tqdm(total=self.total_tasks, desc=desc) as pbar:
            while not self.tasks_finished.is_set():
                cur = self.task_count
                if cur > last_count:
                    pbar.update(cur - last_count)
                    last_count = cur
                time.sleep(0.01)
            # final update
            cur = self.task_count
            if cur > last_count:
                pbar.update(cur - last_count)
                last_count = cur

        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

        if self.error is not None:
            raise self.error

        return

    def unregister(self):
        if self.unregistered:
            return

        try:
            self.wait()
        except Exception as e:
            logger.warning(f"skip cudaHostUnregister because registration failed for shm_ptr={self.shm_ptr}: {e}")
            self.unregistered = True
            return

        if not self.registered:
            self.unregistered = True
            return

        cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
        cuda.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        cuda.cudaHostUnregister.restype = ctypes.c_int

        torch.cuda.set_device(get_current_device_id())
        failed_offsets = []
        for offset, _ in self.tasks:
            ptr = ctypes.c_void_p(self.shm_ptr + offset)
            res = cuda.cudaHostUnregister(ptr)
            if res != 0:
                failed_offsets.append((offset, res))

        if failed_offsets:
            failed_offsets_str = ", ".join(f"offset={offset} err={res}" for offset, res in failed_offsets)
            raise Exception(f"cudaHostUnregister failed for shm_ptr={self.shm_ptr}: {failed_offsets_str}")

        self.unregistered = True


def attach_shm_kv_cache_ptr(key: int, size: int) -> int:
    return attach_shm_kv_cache_handle(key=key, size=size).shm_addr
