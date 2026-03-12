import multiprocessing.shared_memory as shm
from lightllm.utils.shm_utils import create_or_link_shm
from lightllm.utils.shm_registry import unregister_posix_shm


def create_shm(name, data):
    try:
        data_size = len(data)
        shared_memory = create_or_link_shm(name=name, expected_size=data_size, force_mode="create")
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
        shared_memory.close()
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def read_shm(name):
    shared_memory = create_or_link_shm(name=name, expected_size=-1, force_mode="link")
    try:
        return shared_memory.buf.tobytes()
    finally:
        shared_memory.close()


def free_shm(name):
    try:
        shared_memory = shm.SharedMemory(name=name)
    except FileNotFoundError:
        return

    try:
        shared_memory.unlink()
        unregister_posix_shm(name)
    finally:
        shared_memory.close()


def get_shm_name_data(uid):
    return str(uid) + "-data"
