from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.quotes import router as quotes_router
from app.routers.pools import router as pools_router
from app.routers.backtests import router as backtests_router
from app.routers.strategies import router as strategies_router
from app.routers.factory import router as factory_router


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Quote API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(quotes_router)
    app.include_router(pools_router)
    app.include_router(backtests_router)
    app.include_router(strategies_router)
    app.include_router(factory_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
