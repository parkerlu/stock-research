"""maimai signal side

Revision ID: d7a4c2e91b53
Revises: c5e2a91f7d38
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d7a4c2e91b53"
down_revision: Union[str, None] = "c5e2a91f7d38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 买卖很准原本只入库了买点; 补上卖出信号(强势段被打破 → 止盈离场)
    op.add_column("maimai_signal",
                  sa.Column("side", sa.String(length=4), nullable=False,
                            server_default="buy"))
    op.drop_constraint("maimai_signal_pkey", "maimai_signal", type_="primary")
    op.create_primary_key("maimai_signal_pkey", "maimai_signal",
                          ["ts_code", "trade_date", "side"])


def downgrade() -> None:
    op.drop_constraint("maimai_signal_pkey", "maimai_signal", type_="primary")
    op.create_primary_key("maimai_signal_pkey", "maimai_signal",
                          ["ts_code", "trade_date"])
    op.drop_column("maimai_signal", "side")
