"""
ComfyUI / ComfyDeploy API compatibility.

ComfyUI >= 0.3.48 expects:
    await execution.validate_prompt(prompt_id, prompt, partial_execution_list)

Older comfyui-deploy (and some custom nodes) still call:
    execution.validate_prompt(prompt)            # 1 arg
    await execution.validate_prompt(prompt_id, prompt)  # 2 args

That raises:
    TypeError: validate_prompt() missing 2 required positional arguments

We wrap execution.validate_prompt at MSN package load time so legacy callers keep working.
"""

from __future__ import annotations

import inspect
import uuid


def install_validate_prompt_compat() -> None:
    try:
        import execution
    except ImportError:
        return

    original = execution.validate_prompt
    if getattr(original, "_msn_validate_prompt_compat", False):
        return

    try:
        param_count = len(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return

    if param_count < 3:
        return

    if inspect.iscoroutinefunction(original):

        async def validate_prompt_compat(*args, **kwargs):
            if len(args) == 1 and not kwargs:
                return await original(str(uuid.uuid4()), args[0], None)
            if len(args) == 2 and not kwargs:
                return await original(args[0], args[1], None)
            return await original(*args, **kwargs)

    else:

        def validate_prompt_compat(*args, **kwargs):
            if len(args) == 1 and not kwargs:
                return original(str(uuid.uuid4()), args[0], None)
            if len(args) == 2 and not kwargs:
                return original(args[0], args[1], None)
            return original(*args, **kwargs)

    validate_prompt_compat._msn_validate_prompt_compat = True
    execution.validate_prompt = validate_prompt_compat
