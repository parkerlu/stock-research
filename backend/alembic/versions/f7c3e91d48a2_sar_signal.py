"""sar_signal —— SAR翻多 × 吸筹强

Revision ID: f7c3e91d48a2
Revises: e4b7d2915c60
"""
from alembic import op
import sqlalchemy as sa

revision = "f7c3e91d48a2"
down_revision = "e4b7d2915c60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sar_signal",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("prob", sa.Numeric(8, 5), nullable=False),
    )
    op.create_index("ix_sar_date", "sar_signal", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_sar_date", table_name="sar_signal")
    op.drop_table("sar_signal")
