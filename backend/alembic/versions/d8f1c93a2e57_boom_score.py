"""boom_score —— 起爆模型每日打分

Revision ID: d8f1c93a2e57
Revises: c7e2a4b81f35
"""
from alembic import op
import sqlalchemy as sa

revision = "d8f1c93a2e57"
down_revision = "c7e2a4b81f35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "boom_score",
        sa.Column("ts_code", sa.String(12), primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("p_up", sa.Numeric(8, 5), nullable=False),
        sa.Column("p_dn", sa.Numeric(8, 5), nullable=False),
        sa.Column("ev", sa.Numeric(10, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(8, 5), nullable=False),
    )
    op.create_index("ix_boom_date", "boom_score", ["trade_date"])


def downgrade() -> None:
    op.drop_index("ix_boom_date", table_name="boom_score")
    op.drop_table("boom_score")
