import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from shared.logging.logger_config import get_logger

logger = get_logger("API_GATEWAY")

class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        logger.info(f"[{request.method}] {request.url.path} -> Status: {response.status_code} ({duration:.2f}ms)")
        return response