"""
ComfyUI / ComfyDeploy API compatibility.

ComfyUI >= 0.3.48:
    async def validate_prompt(prompt_id, prompt, partial_execution_list)

Pinned comfyui-deploy (e.g. 7b734c4) still does:
    valid = execution.validate_prompt(prompt)   # sync, 1 arg, no await
    if valid[0]: ...

That raises either:
    TypeError: validate_prompt() missing 2 required positional arguments
    TypeError: 'coroutine' object is not subscriptable

This module wraps execution.validate_prompt so legacy sync subscripting and
modern await both work.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import uuid
from collections.abc import Coroutine
from typing import Any


def _run_coro_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


class _ValidatePromptResult:
  """Awaitable validate_prompt result that also supports legacy sync valid[0]."""

  __slots__ = ("_coro", "_result")

  def __init__(self, coro: Coroutine[Any, Any, Any]):
    self._coro = coro
    self._result: Any = None

  def _resolve(self) -> Any:
    if self._result is None:
      self._result = _run_coro_sync(self._coro)
    return self._result

  def __await__(self):
    return self._coro.__await__()

  def __getitem__(self, key: Any) -> Any:
    return self._resolve()[key]

  def __len__(self) -> int:
    return len(self._resolve())

  def __iter__(self):
    return iter(self._resolve())

  def __repr__(self) -> str:
    if self._result is not None:
      return repr(self._result)
    return "<ValidatePromptResult pending>"


def _normalize_validate_prompt_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
  if kwargs:
    return args, kwargs
  if len(args) == 1:
    return (str(uuid.uuid4()), args[0], None), {}
  if len(args) == 2:
    return (args[0], args[1], None), {}
  return args, kwargs


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

  async def _call_original(*args: Any, **kwargs: Any) -> Any:
    return await original(*args, **kwargs)

  def validate_prompt_compat(*args: Any, **kwargs: Any) -> Any:
    norm_args, norm_kwargs = _normalize_validate_prompt_args(args, kwargs)
    coro = _call_original(*norm_args, **norm_kwargs)
    return _ValidatePromptResult(coro)

  validate_prompt_compat._msn_validate_prompt_compat = True
  execution.validate_prompt = validate_prompt_compat
