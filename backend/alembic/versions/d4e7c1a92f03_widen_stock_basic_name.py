"""widen stock_basic.name for ETF/LOF names

VARCHAR(20) 装不下 ETF 简称。tushare fund_basic 实测 2879 个 ETF 里有 113 个
超过 20 字符，最长 32（如 "广发道琼斯美国石油开发与生产指数(QDII-LOF)-A-CNY"）。

因为 ETF 元数据是一次 51 行的批量 upsert，只要有一行超长，整批事务回滚 ——
结果是 ETF 元数据一条都写不进去，随后同步 K 线时又去查这些根本不存在的记录。
日志中可见 (2026-07-29 / 07-30):
    ERROR: value too long for type character varying(20)

Revision ID: d4e7c1a92f03
Revises: c3f8a2d91e4b
Create Date: 2026-08-01
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e7c1a92f03"
down_revision = "c3f8a2d91e4b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "stock_basic",
        "name",
        existing_type=sa.String(20),
        type_=sa.String(64),
        existing_nullable=False,
    )


def downgrade() -> None:
    # 回退会截断超过 20 字符的名字，否则 Postgres 直接报错。
    op.execute("UPDATE stock_basic SET name = left(name, 20) WHERE length(name) > 20")
    op.alter_column(
        "stock_basic",
        "name",
        existing_type=sa.String(64),
        type_=sa.String(20),
        existing_nullable=False,
    )
