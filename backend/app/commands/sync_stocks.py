"""One-off script to sync stock_basic from data source into PostgreSQL."""
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.datasources.akshare_provider import AKShareProvider
from app.datasources.manager import DataSourceManager
from app.datasources.tushare_provider import TuShareProvider
from app.models.schema import StockBasic


async def sync():
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    primary = TuShareProvider(token=settings.tushare_token)
    fallback = AKShareProvider()
    manager = DataSourceManager(primary=primary, fallback=fallback)

    print("Fetching stock_basic from data source...")
    df = await manager.fetch_stock_basic()
    print(f"Got {len(df)} stocks. Upserting into database...")

    async with session_factory() as session:
        for _, row in df.iterrows():
            stock = StockBasic(
                ts_code=row["ts_code"],
                symbol=row["symbol"],
                name=row["name"],
                area=row.get("area"),
                industry=row.get("industry"),
                market=row.get("market"),
                list_date=row.get("list_date"),
                is_active=True,
            )
            await session.merge(stock)
        await session.commit()

    print("Done.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(sync())
