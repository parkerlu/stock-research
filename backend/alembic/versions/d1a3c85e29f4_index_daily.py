"""index_daily —— 指数日线(中证1000等)

Revision ID: d1a3c85e29f4
Revises: c9e2b47f10a3
"""
from alembic import op
import sqlalchemy as sa

revision = "d1a3c85e29f4"
down_revision = "c9e2b47f10a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_daily",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("vol", sa.Numeric(20, 2), nullable=False),
        sa.Column("amount", sa.Numeric(20, 2), nullable=False),
    )
    op.create_index("ix_index_lookup", "index_daily", ["ts_code", "trade_date"])


def downgrade() -> None:
    op.drop_index("ix_index_lookup", table_name="index_daily")
    op.drop_table("index_daily")
