"""maimai v3.5 signal

Revision ID: f2c8d604a915
Revises: e8b3f5c04a71
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f2c8d604a915"
down_revision: Union[str, None] = "e8b3f5c04a71"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "maimai35_signal",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("score", sa.Numeric(12, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("grade", sa.String(length=4), nullable=False),
    )
    op.create_index("ix_mm35_lookup", "maimai35_signal", ["ts_code", "trade_date"])
    op.create_index("ix_mm35_date_rank", "maimai35_signal", ["trade_date", "rank_pct"])


def downgrade() -> None:
    op.drop_table("maimai35_signal")
