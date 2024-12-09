from fastapi import APIRouter
from api.endpoints import root, analyze

router = APIRouter()

router.include_router(root.router)
router.include_router(analyze.router, prefix="/analyze")
