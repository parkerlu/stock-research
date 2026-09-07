"""maimai_weekly —— 买卖很准周线版

Revision ID: a2f9c47e83b1
Revises: f7c3e91d48a2
"""
from alembic import op
import sqlalchemy as sa

revision = "a2f9c47e83b1"
down_revision = "f7c3e91d48a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "maimai_weekly",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("week_end", sa.Date, primary_key=True),
        sa.Column("buy_line", sa.Numeric(10, 4), nullable=False),
    )
    op.create_index("ix_mmwk_date", "maimai_weekly", ["week_end"])


def downgrade() -> None:
    op.drop_index("ix_mmwk_date", table_name="maimai_weekly")
    op.drop_table("maimai_weekly")
