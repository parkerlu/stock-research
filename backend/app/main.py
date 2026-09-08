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
from app.routers.toplist import router as toplist_router
from app.services.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)


def _selfcheck_commands() -> None:
    """启动自检: 定时任务用到的命令模块必须都能导入。

    ⚠️ 2026-09-08 一天之内栽了两次同样的坑: 代码写好、逻辑串好、git 也提交了,
       但【容器镜像里没有那个文件】, 于是那一步静默跳过 ——
       fill_daily 没拷进来导致日线卡在 2301 只;
       build_panels 缺失导致三个指标用了四天前的缓存面板。
       两次都不报错, 都是靠人发现。这里让它启动就喊。
    """
    import importlib
    import logging as _lg

    _log = _lg.getLogger("selfcheck")
    need = ["ensure_daily", "fill_daily", "build_panels", "update_indicators",
            "build_shape_scores", "sync_shape_signals", "sync_mmweek_signals",
            "sync_sar_signals", "sync_breakout_signals"]
    missing = []
    for m in need:
        try:
            importlib.import_module(f"app.commands.{m}")
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{m}({type(exc).__name__})")
    if missing:
        _log.error("⚠️ 定时任务依赖的命令缺失或导入失败: %s —— "
                   "镜像可能没重建, 相关步骤会静默跳过", ", ".join(missing))
    else:
        _log.info("启动自检: %d 个命令模块全部可导入", len(need))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _selfcheck_commands()
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
    app.include_router(toplist_router)
    app.include_router(forecast_router)
    app.include_router(paper_router)
    app.include_router(sectors_router)
    from app.routers.kronos_pred import router as kronos_router
    app.include_router(kronos_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
