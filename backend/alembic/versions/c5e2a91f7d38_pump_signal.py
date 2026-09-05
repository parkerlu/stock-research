"""pump signal

Revision ID: c5e2a91f7d38
Revises: b3d1f8a06e42
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c5e2a91f7d38"
down_revision: Union[str, None] = "b3d1f8a06e42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pump_signal",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("prob", sa.Numeric(8, 5), nullable=False),
        sa.Column("rank_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("grade", sa.String(length=4), nullable=False),
    )
    op.create_index("ix_pump_lookup", "pump_signal", ["ts_code", "trade_date"])
    op.create_index("ix_pump_date_rank", "pump_signal", ["trade_date", "rank_pct"])


def downgrade() -> None:
    op.drop_table("pump_signal")
