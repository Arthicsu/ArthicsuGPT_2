from fastapi import APIRouter
from .routers import ssd_router

router = APIRouter()

router.include_router(ssd_router.router)