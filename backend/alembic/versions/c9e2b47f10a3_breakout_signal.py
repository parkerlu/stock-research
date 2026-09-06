"""breakout_signal —— 突破60日高点 × 吸筹强

Revision ID: c9e2b47f10a3
Revises: b6f1a83d05c7
"""
from alembic import op
import sqlalchemy as sa

revision = "c9e2b47f10a3"
down_revision = "b6f1a83d05c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "breakout_signal",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("prob", sa.Numeric(8, 5), nullable=False),
        sa.Column("hh60", sa.Numeric(12, 4), nullable=False),
    )
    op.create_index("ix_breakout_date", "breakout_signal", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_breakout_date", table_name="breakout_signal")
    op.drop_table("breakout_signal")
