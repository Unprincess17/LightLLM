import os
from typing import Any, Optional


def build_lightweight_health_status(args: Optional[Any], httpserver_manager: Optional[Any]) -> tuple[int, dict]:
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return 503, {"message": "Error", "check": "lightweight", "reason": "debug_forced_failure"}

    if args is None:
        return 503, {"message": "Error", "check": "lightweight", "reason": "args_not_initialized"}

    run_mode = getattr(args, "run_mode", "unknown")
    if run_mode == "pd_master":
        return 200, {"message": "Ok", "check": "lightweight", "run_mode": run_mode}

    if httpserver_manager is None:
        return 503, {"message": "Error", "check": "lightweight", "reason": "httpserver_manager_not_initialized"}

    tokenizer_ready = getattr(httpserver_manager, "tokenizer", None) is not None
    router_socket_ready = getattr(httpserver_manager, "send_to_router", None) is not None
    latest_success_mark_obj = getattr(httpserver_manager, "latest_success_infer_time_mark", None)
    latest_success_mark = int(latest_success_mark_obj.get_value()) if latest_success_mark_obj is not None else 0
    if not tokenizer_ready or not router_socket_ready:
        return (
            503,
            {
                "message": "Error",
                "check": "lightweight",
                "reason": "manager_dependencies_not_ready",
                "tokenizer_ready": tokenizer_ready,
                "router_socket_ready": router_socket_ready,
            },
        )

    return (
        200,
        {
            "message": "Ok",
            "check": "lightweight",
            "run_mode": run_mode,
            "model_name": getattr(args, "model_name", None),
            "tokenizer_ready": tokenizer_ready,
            "router_socket_ready": router_socket_ready,
            "latest_success_infer_time_mark": latest_success_mark,
        },
    )
