import errno

from lightllm.utils.kv_cache_utils import SystemVShmHandle
from lightllm.utils.kv_cache_utils import create_shm_kv_cache_handle
from lightllm.utils.shm_utils import destroy_shared_memory


class DummyShm:
    def __init__(self, name: str, owner: bool):
        self.name = name
        self._lightllm_owner = owner
        self.closed = 0
        self.unlinked = 0

    def close(self):
        self.closed += 1

    def unlink(self):
        self.unlinked += 1


def test_destroy_shared_memory_owner_unlinks(monkeypatch):
    calls = []
    monkeypatch.setattr("lightllm.utils.shm_utils.unregister_posix_shm", lambda name: calls.append(name))

    shm = DummyShm("owner_shm", owner=True)
    destroy_shared_memory(shm)

    assert shm.unlinked == 1
    assert shm.closed == 1
    assert calls == ["owner_shm"]


def test_destroy_shared_memory_attacher_only_closes(monkeypatch):
    calls = []
    monkeypatch.setattr("lightllm.utils.shm_utils.unregister_posix_shm", lambda name: calls.append(name))

    shm = DummyShm("linked_shm", owner=False)
    destroy_shared_memory(shm)

    assert shm.unlinked == 0
    assert shm.closed == 1
    assert calls == []


def test_systemv_handle_destroy_is_idempotent(monkeypatch):
    class DummyLibC:
        def __init__(self):
            self.shmdt_calls = 0
            self.shmctl_calls = 0

        def shmdt(self, addr):
            self.shmdt_calls += 1
            return 0

        def shmctl(self, shmid, cmd, buf):
            self.shmctl_calls += 1
            return 0

    dummy_libc = DummyLibC()
    unregister_calls = []
    monkeypatch.setattr("lightllm.utils.kv_cache_utils._get_sysv_libc", lambda: dummy_libc)
    monkeypatch.setattr(
        "lightllm.utils.kv_cache_utils.unregister_sysv_shm",
        lambda key=None, shmid=None: unregister_calls.append((key, shmid)),
    )

    handle = SystemVShmHandle(key=7, shmid=11, shm_addr=13, size=17, is_owner=True)
    handle.destroy()
    handle.destroy()

    assert dummy_libc.shmdt_calls == 1
    assert dummy_libc.shmctl_calls == 1
    assert unregister_calls == [(7, 11)]


def test_create_shm_kv_cache_handle_falls_back_to_attacher_when_segment_exists(monkeypatch):
    class DummyLibC:
        def __init__(self):
            self.shmget_calls = []

        def shmget(self, key, size, flags):
            self.shmget_calls.append((key, size, flags))
            if len(self.shmget_calls) == 1:
                return -1
            return 123

        def shmat(self, shmid, addr, flags):
            return 456

        def shmdt(self, addr):
            return 0

        def shmctl(self, shmid, cmd, buf):
            return 0

    dummy_libc = DummyLibC()
    cleanup_callbacks = []
    cleanup_registrations = []
    registry_registrations = []

    monkeypatch.setattr("lightllm.utils.kv_cache_utils._get_sysv_libc", lambda: dummy_libc)
    monkeypatch.setattr("lightllm.utils.kv_cache_utils.enable_huge_page", lambda: False)
    monkeypatch.setattr("lightllm.utils.kv_cache_utils.ctypes.get_errno", lambda: errno.EEXIST)
    monkeypatch.setattr(
        "lightllm.utils.kv_cache_utils.register_cleanup_callback", lambda callback: cleanup_callbacks.append(callback)
    )
    monkeypatch.setattr(
        "lightllm.utils.kv_cache_utils.register_sysv_shm_for_cleanup",
        lambda key, shmid=None: cleanup_registrations.append((key, shmid)),
    )
    monkeypatch.setattr(
        "lightllm.utils.kv_cache_utils.register_sysv_shm",
        lambda key, shmid, creator_pid=None: registry_registrations.append((key, shmid, creator_pid)),
    )

    handle = create_shm_kv_cache_handle(key=77, size=4096)

    assert handle.shmid == 123
    assert handle.shm_addr == 456
    assert handle.is_owner is False
    assert cleanup_registrations == []
    assert registry_registrations == []
    assert len(cleanup_callbacks) == 1
