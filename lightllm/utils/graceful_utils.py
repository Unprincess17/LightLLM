import sys
import signal
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_shutdown_requested = False


def graceful_registry(sub_module_name):
    def graceful_shutdown(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info(f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown...")
        if signum == signal.SIGTERM:
            logger.info(f"{sub_module_name} recive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)
    return


def is_shutdown_requested():
    return _shutdown_requested
