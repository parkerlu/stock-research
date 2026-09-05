import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.quotes import router as quotes_router
from app.routers.pools import router as pools_router
from app.routers.backtests import router as backtests_router
from app.routers.strategies import router as strategies_router
from app.routers.factory import router as factory_router
from app.routers.indicators import router as indicators_router
from app.routers.screening import router as screening_router
from app.routers.strategy_pool import router as strategy_pool_router
from app.routers.system import router as system_router
from app.routers.forecast import router as forecast_router
from app.routers.paper import router as paper_router
from app.routers.sectors import router as sectors_router
from app.services.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Quote API", version="0.1.0", lifespan=lifespan)
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
    app.include_router(indicators_router)
    app.include_router(screening_router)
    app.include_router(strategy_pool_router)
    app.include_router(system_router)
    app.include_router(forecast_router)
    app.include_router(paper_router)
    app.include_router(sectors_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
