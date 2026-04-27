"""One-off script to sync stock_basic from TuShare into PostgreSQL."""
import asyncio
import math

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.models.schema import StockBasic


async def sync():
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    provider = TuShareProvider(token=settings.tushare_token)

    print("Fetching stock_basic from TuShare...")
    df = await provider.fetch_stock_basic()
    print(f"Got {len(df)} stocks. Upserting into database...")

    def clean(val):
        """Convert NaN/None to None for nullable string columns."""
        if val is None:
            return None
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    async with session_factory() as session:
        for _, row in df.iterrows():
            stock = StockBasic(
                ts_code=row["ts_code"],
                symbol=row["symbol"],
                name=row["name"],
                area=clean(row.get("area")),
                industry=clean(row.get("industry")),
                market=clean(row.get("market")),
                list_date=row.get("list_date"),
                is_active=True,
            )
            await session.merge(stock)
        await session.commit()

    print("Done.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(sync())
