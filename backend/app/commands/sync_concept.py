"""同步同花顺概念板块与成分股。

用法: python -m app.commands.sync_concept

⚠️ 上游只提供当前成分快照, 没有历史进出记录 —— 这张表能回答"现在谁属于哪个板块",
但绝不能用来做历史回测(拿今天的成分去测过去 = 前视偏差)。
板块的 list_date 则是真实时间戳, 可以安全使用。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import ConceptMember, ConceptSector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sync_concept")

# 非主题板块 —— 宽基指数样本股 / 交易属性, 不是"热点", 单独标记以便前端过滤
NON_THEME = ("成份股", "样本股", "指数", "融资融券", "沪股通", "深股通",
             "标准券", "MSCI", "富时", "转债标的", "股权转让", "举牌")


def _call(pro, name, **kw):
    for _ in range(4):
        try:
            return getattr(pro, name)(**kw)
        except Exception as e:  # noqa: BLE001
            if "超限" in str(e) or "频率" in str(e):
                time.sleep(15)
                continue
            raise
    raise RuntimeError(f"{name} 重试用尽")


async def main() -> None:
    import tushare as ts

    pro = ts.pro_api(settings.tushare_token)
    idx = _call(pro, "ths_index", exchange="A", type="N")
    log.info("板块 %d 个", len(idx))

    eng = create_async_engine(settings.database_url)
    Session = async_sessionmaker(eng, expire_on_commit=False)

    rows = []
    for _, r in idx.iterrows():
        ld = str(r.get("list_date") or "")
        rows.append({
            "ts_code": r["ts_code"],
            "name": str(r["name"])[:64],
            "count": int(r["count"]) if pd.notna(r.get("count")) else None,
            "exchange": r.get("exchange"),
            "list_date": datetime.strptime(ld, "%Y%m%d").date() if len(ld) == 8 else None,
            "type": r.get("type"),
            "updated_at": datetime.now(),
        })
    async with Session() as db:
        stmt = insert(ConceptSector).values(rows)
        await db.execute(stmt.on_conflict_do_update(
            index_elements=["ts_code"],
            set_={c: stmt.excluded[c] for c in
                  ["name", "count", "exchange", "list_date", "type", "updated_at"]},
        ))
        await db.commit()
    log.info("板块写入完成")

    total = 0
    for i, (_, r) in enumerate(idx.iterrows(), 1):
        try:
            m = _call(pro, "ths_member", ts_code=r["ts_code"])
        except Exception as e:  # noqa: BLE001
            log.warning("成分失败 %s: %s", r["ts_code"], str(e)[:60])
            continue
        if m is None or len(m) == 0:
            continue
        vals = [{"sector_code": r["ts_code"], "ts_code": x["con_code"],
                 "name": str(x.get("con_name") or "")[:64]}
                for _, x in m.iterrows() if x.get("con_code")]
        if not vals:
            continue
        async with Session() as db:
            await db.execute(delete(ConceptMember).where(
                ConceptMember.sector_code == r["ts_code"]))
            stmt = insert(ConceptMember).values(vals)
            await db.execute(stmt.on_conflict_do_nothing(constraint="uq_concept_member"))
            await db.commit()
        total += len(vals)
        if i % 50 == 0:
            log.info("成分 %d/%d, 累计 %d 条", i, len(idx), total)

    async with Session() as db:
        n1 = (await db.execute(text("select count(*) from concept_sector"))).scalar()
        n2 = (await db.execute(text("select count(*) from concept_member"))).scalar()
        n3 = (await db.execute(text("select count(distinct ts_code) from concept_member"))).scalar()
    log.info("完成: 板块 %s, 成分关系 %s, 覆盖个股 %s", n1, n2, n3)
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
