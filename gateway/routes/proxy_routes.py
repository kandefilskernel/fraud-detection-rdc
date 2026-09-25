from fastapi import APIRouter

router = APIRouter()

@router.get("/api/v1/health")
async def gateway_health():
    return {"gateway": "UP", "status": 200}