from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from filelock import FileLock
from lightllm.utils.log_utils import init_logger
from lightllm.utils.auto_shm_cleanup import register_posix_shm_for_cleanup
from lightllm.utils.shm_registry import register_posix_shm, unregister_posix_shm

logger = init_logger(__name__)


def create_or_link_shm(name, expected_size, force_mode=None, auto_cleanup=False):
    """
    Args:
        name: name of the shared memory
        expected_size: expected size of the shared memory, if expected_size == -1, no check for size linked.
        force_mode: force mode
            - 'create': force create new shared memory, if exists, delete and create
            - 'link': force link to existing shared memory, if not exists, raise exception
            - None (default): smart mode, link to existing, if not exists, create

    Returns:
        shared_memory.SharedMemory: shared memory object

    Raises:
        FileNotFoundError: when force_mode='link' but shared memory not exists
        ValueError: when force_mode='link' but size mismatch
    """
    lock_name = f"/tmp/{name}.lock"

    if force_mode == "create":
        with FileLock(lock_name):
            return _force_create_shm(name, expected_size, auto_cleanup)
    elif force_mode == "link":
        return _force_link_shm(name, expected_size)
    else:
        with FileLock(lock_name):
            return _smart_create_or_link_shm(name, expected_size, auto_cleanup)


def _force_create_shm(name, expected_size, auto_cleanup):
    """强制创建新的共享内存"""
    try:
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
    except:
        pass

    # 创建新的共享内存
    shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
    _mark_shared_memory_owner(shm, is_owner=True)
    register_posix_shm_for_cleanup(name)
    register_posix_shm(name)
    return shm


def _force_link_shm(name, expected_size):
    """强制连接到已存在的共享内存,
    如果 expected_size 为 -1, 则不进行link的size校验比对"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        _mark_shared_memory_owner(shm, is_owner=False)
        _untrack_attached_shared_memory(shm)
        # 验证大小
        if expected_size != -1 and shm.size != expected_size:
            shm.close()
            raise ValueError(f"Shared memory {name} size mismatch: expected {expected_size}, got {shm.size}")
        # logger.info(f"Force linked to existing shared memory: {name} (size={expected_size})")
        return shm
    except Exception as e:
        raise e


def _smart_create_or_link_shm(name, expected_size, auto_cleanup):
    """优先连接，不存在则创建"""
    try:
        shm = _force_link_shm(name=name, expected_size=expected_size)
        return shm
    except:
        pass

    return _force_create_shm(name=name, expected_size=expected_size, auto_cleanup=auto_cleanup)


def _mark_shared_memory_owner(shm: shared_memory.SharedMemory, is_owner: bool) -> shared_memory.SharedMemory:
    shm._lightllm_owner = is_owner
    shm._lightllm_untracked = False
    return shm


def _untrack_attached_shared_memory(shm: shared_memory.SharedMemory) -> None:
    if getattr(shm, "_lightllm_owner", False):
        return
    if getattr(shm, "_lightllm_untracked", False):
        return
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
        shm._lightllm_untracked = True
    except Exception as e:
        logger.warning(f"failed to unregister shared_memory {shm.name} from resource_tracker: {e}")


def is_shm_owner(shm: shared_memory.SharedMemory) -> bool:
    return bool(getattr(shm, "_lightllm_owner", False))


def destroy_shared_memory(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    try:
        if is_shm_owner(shm):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            unregister_posix_shm(shm.name)
    finally:
        shm.close()
