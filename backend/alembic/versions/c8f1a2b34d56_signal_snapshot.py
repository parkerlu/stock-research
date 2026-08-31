"""每日信号快照表 —— 用真实前进数据量 ZigZag 重绘漂移

Revision ID: c8f1a2b34d56
Revises: a3d9e1f75c02
"""
from alembic import op
import sqlalchemy as sa

revision = "c8f1a2b34d56"
down_revision = "a3d9e1f75c02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signal_snapshot",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("ts_code", sa.String(12), nullable=False),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column("fractal_date", sa.Date, nullable=True),
        sa.Column("kind", sa.String(2), nullable=False),
        sa.Column("close_at_signal", sa.Numeric(12, 4), nullable=True),
        sa.Column("amount_20d_k", sa.Numeric(16, 2), nullable=True),
        sa.Column("still_valid", sa.Boolean, nullable=True),
        sa.Column("checked_on", sa.Date, nullable=True),
        sa.UniqueConstraint("snapshot_date", "ts_code", "trade_date", "kind",
                            name="uq_signal_snapshot"),
    )
    op.create_index("ix_signal_snapshot_date", "signal_snapshot", ["snapshot_date"])


def downgrade() -> None:
    op.drop_index("ix_signal_snapshot_date", table_name="signal_snapshot")
    op.drop_table("signal_snapshot")
