"""top list

Revision ID: a4d7e91c26b8
Revises: f2c8d604a915
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a4d7e91c26b8"
down_revision: Union[str, None] = "f2c8d604a915"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "top_list",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("close", sa.Numeric(12, 4), nullable=True),
        sa.Column("pct_change", sa.Numeric(10, 4), nullable=True),
        sa.Column("turnover_rate", sa.Numeric(10, 4), nullable=True),
        sa.Column("l_buy", sa.Numeric(18, 2), nullable=True),
        sa.Column("l_sell", sa.Numeric(18, 2), nullable=True),
        sa.Column("net_amount", sa.Numeric(18, 2), nullable=True),
        sa.Column("reason", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_toplist_lookup", "top_list", ["ts_code", "trade_date"])
    op.create_index("ix_toplist_date", "top_list", ["trade_date"])


def downgrade() -> None:
    op.drop_table("top_list")
