import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < self.window_seconds]
        
        if len(self.clients[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429, 
                content={"detail": "Limite de requêtes dépassée (Rate limit)"}
            )
            
        self.clients[client_ip].append(now)
        return await call_next(request)