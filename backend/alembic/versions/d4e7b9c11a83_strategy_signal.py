"""通用策略信号表, 取代 chan_signal

Revision ID: d4e7b9c11a83
Revises: c8f1a2b34d56
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e7b9c11a83"
down_revision = "c8f1a2b34d56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_signal",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("ts_code", sa.String(12), nullable=False),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.UniqueConstraint("strategy", "ts_code", "trade_date",
                            name="uq_strategy_signal"),
    )
    op.create_index("ix_strategy_signal_lookup", "strategy_signal",
                    ["strategy", "trade_date"])
    # chan_signal 留着当审计证据, 但代码已不再读写它
    op.execute("COMMENT ON TABLE chan_signal IS "
               "'缠论信号(已下架, 仅作 2026-09 未来函数审计的证据保留)'")


def downgrade() -> None:
    op.drop_index("ix_strategy_signal_lookup", table_name="strategy_signal")
    op.drop_table("strategy_signal")
