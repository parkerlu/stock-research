"""index_bar —— 指数多周期K线

Revision ID: e4b7d2915c60
Revises: d1a3c85e29f4
"""
from alembic import op
import sqlalchemy as sa

revision = "e4b7d2915c60"
down_revision = "d1a3c85e29f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_bar",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("freq", sa.String(8), primary_key=True),
        sa.Column("bar_time", sa.DateTime, primary_key=True),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("vol", sa.Numeric(20, 2), nullable=False),
    )
    op.create_index("ix_ibar_lookup", "index_bar", ["ts_code", "freq", "bar_time"])


def downgrade() -> None:
    op.drop_index("ix_ibar_lookup", table_name="index_bar")
    op.drop_table("index_bar")
