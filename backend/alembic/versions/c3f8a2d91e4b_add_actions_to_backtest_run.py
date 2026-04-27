"""add actions column to backtest_run

Revision ID: c3f8a2d91e4b
Revises: ad610a51675b
Create Date: 2026-04-17
"""
from alembic import op
import sqlalchemy as sa

revision = "c3f8a2d91e4b"
down_revision = "ad610a51675b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("backtest_run", sa.Column("actions", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("backtest_run", "actions")
