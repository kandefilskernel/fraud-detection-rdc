import uvicorn
from fastapi import FastAPI
from gateway.middleware.request_logger import RequestLoggerMiddleware
from gateway.middleware.rate_limiter import RateLimiterMiddleware
from gateway.routes.proxy_routes import router as proxy_router
from gateway.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0"
)

app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(RateLimiterMiddleware, max_requests=settings.RATE_LIMIT_REQUESTS)
app.include_router(proxy_router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "API Gateway"}

if __name__ == "__main__":
    uvicorn.run("gateway.main:app", host="0.0.0.0", port=8000, reload=True)