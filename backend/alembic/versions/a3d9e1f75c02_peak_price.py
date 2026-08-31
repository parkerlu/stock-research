"""paper_position 增加 peak_price (移动止盈用)

Revision ID: a3d9e1f75c02
Revises: f7c2d3e8b914
"""
from alembic import op
import sqlalchemy as sa

revision = "a3d9e1f75c02"
down_revision = "f7c2d3e8b914"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("paper_position",
                  sa.Column("peak_price", sa.Numeric(12, 4), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_position", "peak_price")
