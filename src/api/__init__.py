from fastapi import APIRouter
from .routers import satellite_router

router = APIRouter()

router.include_router(satellite_router.router)