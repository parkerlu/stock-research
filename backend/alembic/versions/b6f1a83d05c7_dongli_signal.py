"""dongli_signal —— 动力线上穿0.2 的日子

Revision ID: b6f1a83d05c7
Revises: a4d7e91c26b8
"""
from alembic import op
import sqlalchemy as sa

revision = "b6f1a83d05c7"
down_revision = "a4d7e91c26b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dongli_signal",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("dl_value", sa.Numeric(8, 4), nullable=False),
    )
    op.create_index("ix_dongli_date", "dongli_signal", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_dongli_date", table_name="dongli_signal")
    op.drop_table("dongli_signal")
