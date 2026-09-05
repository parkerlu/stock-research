"""concept sector and member

Revision ID: a1c9e4d72b60
Revises: e1a5c73f2b40
Create Date: 2026-09-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1c9e4d72b60"
down_revision: Union[str, None] = "e1a5c73f2b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "concept_sector",
        sa.Column("ts_code", sa.String(length=12), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("count", sa.Integer(), nullable=True),
        sa.Column("exchange", sa.String(length=10), nullable=True),
        sa.Column("list_date", sa.Date(), nullable=True),
        sa.Column("type", sa.String(length=4), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_concept_sector_name", "concept_sector", ["name"])
    op.create_index("ix_concept_sector_list_date", "concept_sector", ["list_date"])

    op.create_table(
        "concept_member",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sector_code", sa.String(length=12), nullable=False),
        sa.Column("ts_code", sa.String(length=12), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("sector_code", "ts_code", name="uq_concept_member"),
    )
    op.create_index("ix_concept_member_stock", "concept_member", ["ts_code"])
    op.create_index("ix_concept_member_sector", "concept_member", ["sector_code"])


def downgrade() -> None:
    op.drop_table("concept_member")
    op.drop_table("concept_sector")
