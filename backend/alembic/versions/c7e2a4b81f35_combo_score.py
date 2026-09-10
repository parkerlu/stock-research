"""combo_score —— 筹码 × 裸K 共振打分

Revision ID: c7e2a4b81f35
Revises: b3d5f1a92c74
"""
from alembic import op
import sqlalchemy as sa

revision = "c7e2a4b81f35"
down_revision = "b3d5f1a92c74"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "combo_score",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("ev_rawk", sa.Numeric(10, 6), nullable=False),
        sa.Column("ev_chips", sa.Numeric(10, 6), nullable=False),
        sa.Column("ev", sa.Numeric(10, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(8, 5), nullable=False),
    )
    op.create_index("ix_combo_date", "combo_score", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_combo_date", table_name="combo_score")
    op.drop_table("combo_score")
