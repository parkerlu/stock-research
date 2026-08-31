"""缠论买点预计算缓存

虚拟盘回放要"点一下走一天", 每天现算 4400 只票的缠论要 75 秒, 无法交互。
整段历史一次算完只需几十秒, 落表后回放变成索引查询。

trade_date 是可操作日(分型 + CONFIRM_LAG=2), 消费方不用再处理 lag。

Revision ID: f7c2d3e8b914
Revises: e5b1c8f2a731
Create Date: 2026-08-31
"""
from alembic import op
import sqlalchemy as sa

revision = "f7c2d3e8b914"
down_revision = "e5b1c8f2a731"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chan_signal",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts_code", sa.String(12), nullable=False, index=True),
        sa.Column("trade_date", sa.Date, nullable=False, index=True),
        sa.Column("kind", sa.String(4), nullable=False),
        sa.Column("fractal_date", sa.Date, nullable=False),
    )
    op.create_index("ix_chan_signal_date", "chan_signal", ["trade_date", "kind"])
    op.create_unique_constraint("uq_chan_signal", "chan_signal",
                                ["ts_code", "trade_date", "kind"])


def downgrade() -> None:
    op.drop_table("chan_signal")
