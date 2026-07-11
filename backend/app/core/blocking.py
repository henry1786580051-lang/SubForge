import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_blocking(
    function: Callable[..., T],
    *args: Any,
    on_cancel: Callable[[], Any] | None = None,
) -> T:
    """Run blocking work without deleting its resources during cancellation.

    ``run_in_executor`` cannot stop a native Python/ML worker when the awaiting
    coroutine is cancelled. Shield the worker and wait for it to release input
    files before allowing the caller's ``finally`` block to clean them up.
    """
    loop = asyncio.get_running_loop()
    worker = loop.run_in_executor(None, function, *args)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if on_cancel is not None:
            try:
                on_cancel()
            except Exception:
                logger.exception("Blocking task cancellation hook failed")
        try:
            await asyncio.shield(worker)
        except Exception:
            logger.debug("Cancelled blocking worker exited with an error", exc_info=True)
        raise
