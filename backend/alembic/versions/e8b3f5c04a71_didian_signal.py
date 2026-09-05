"""didian signal

Revision ID: e8b3f5c04a71
Revises: d7a4c2e91b53
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e8b3f5c04a71"
down_revision: Union[str, None] = "d7a4c2e91b53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "didian_signal",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("score", sa.Numeric(12, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("grade", sa.String(length=4), nullable=False),
    )
    op.create_index("ix_didian_lookup", "didian_signal", ["ts_code", "trade_date"])
    op.create_index("ix_didian_date_rank", "didian_signal", ["trade_date", "rank_pct"])


def downgrade() -> None:
    op.drop_table("didian_signal")
