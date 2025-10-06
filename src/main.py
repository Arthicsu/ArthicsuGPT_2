from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.api.routers import satellite_router


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(satellite_router.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1")