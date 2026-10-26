"""Keep the embedded HTTP and WebSocket server local to its frontend."""

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEV_ORIGINS = frozenset({("http", "localhost", 3000), ("http", "127.0.0.1", 3000)})


def _local_origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in _LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port
        if port == 0:
            return None
        return parsed.scheme, parsed.hostname, port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


class LocalRequestGuard:
    """Reject rebinding hosts and foreign browser requests before any route runs.

    Same-origin pages work on dynamically allocated desktop ports. Source-mode
    development additionally permits the frontend on port 3000. Origin-less
    native/CLI clients remain supported; cross-site browser requests do not.
    """

    def __init__(self, app: ASGIApp, *, allow_dev_frontend: bool = False):
        self.app = app
        self.allow_dev_frontend = allow_dev_frontend

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        scheme = {"ws": "http", "wss": "https"}.get(scope["scheme"], scope["scheme"])
        hosts = headers.getlist("host")
        origin = _local_origin(f"{scheme}://{hosts[0]}") if len(hosts) == 1 else None
        status, detail = 400, "Invalid local Host"
        if origin is not None:
            origins = headers.getlist("origin")
            supplied = _local_origin(origins[0]) if len(origins) == 1 else None
            dev = self.allow_dev_frontend and supplied in _DEV_ORIGINS
            valid = not origins or (len(origins) == 1 and (supplied == origin or dev))
            if headers.get("sec-fetch-site") == "cross-site" and not dev:
                valid = False
            if valid:
                await self.app(scope, receive, send)
                return
            status, detail = 403, "Untrusted request origin"
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await PlainTextResponse(detail, status_code=status)(scope, receive, send)
