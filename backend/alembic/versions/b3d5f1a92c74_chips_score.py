"""chips_score —— 筹码模型每日全市场打分

Revision ID: b3d5f1a92c74
Revises: a2f9c47e83b1
"""
from alembic import op
import sqlalchemy as sa

revision = "b3d5f1a92c74"
down_revision = "a2f9c47e83b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chips_score",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("p_up", sa.Numeric(8, 5), nullable=False),
        sa.Column("p_dn", sa.Numeric(8, 5), nullable=False),
        sa.Column("ev", sa.Numeric(10, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(8, 5), nullable=False),
    )
    op.create_index("ix_chips_date", "chips_score", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_chips_date", table_name="chips_score")
    op.drop_table("chips_score")
