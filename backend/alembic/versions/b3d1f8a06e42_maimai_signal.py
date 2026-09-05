"""maimai signal

Revision ID: b3d1f8a06e42
Revises: a1c9e4d72b60
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3d1f8a06e42"
down_revision: Union[str, None] = "a1c9e4d72b60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "maimai_signal",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("score", sa.Numeric(12, 6), nullable=False),
        sa.Column("rank_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("grade", sa.String(length=4), nullable=False),
    )
    op.create_index("ix_maimai_lookup", "maimai_signal", ["ts_code", "trade_date"])
    op.create_index("ix_maimai_date", "maimai_signal", ["trade_date"])


def downgrade() -> None:
    op.drop_table("maimai_signal")
