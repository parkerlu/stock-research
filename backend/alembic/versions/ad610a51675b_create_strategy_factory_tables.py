"""create strategy factory tables

Revision ID: ad610a51675b
Revises: 7bf562a33435
Create Date: 2026-04-17 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad610a51675b'
down_revision: Union[str, None] = '7bf562a33435'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'factory_job',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('ts_code', sa.String(20), nullable=False, index=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('total_candidates', sa.Integer(), server_default='0'),
        sa.Column('evaluated', sa.Integer(), server_default='0'),
        sa.Column('passed', sa.Integer(), server_default='0'),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'strategy',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('ts_code', sa.String(20), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('template', sa.String(50), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('is_pinned', sa.Boolean(), server_default='false'),
        sa.Column('annualized_return', sa.Numeric(12, 6), nullable=True),
        sa.Column('net_profit', sa.Numeric(14, 4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(8, 6), nullable=True),
        sa.Column('win_rate', sa.Numeric(6, 4), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('profit_factor', sa.Numeric(10, 4), nullable=True),
        sa.Column('final_capital', sa.Numeric(14, 4), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('job_id', sa.String(36), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_strategy_ts_code', 'strategy', ['ts_code'])

    op.create_table(
        'backtest_run',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('ts_code', sa.String(20), nullable=False, index=True),
        sa.Column('strategy_id', sa.Integer(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('initial_capital', sa.Numeric(14, 4), server_default='10000'),
        sa.Column('position_ratios', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('trades', sa.JSON(), nullable=True),
        sa.Column('equity_curve', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('backtest_run')
    op.drop_index('ix_strategy_ts_code', table_name='strategy')
    op.drop_table('strategy')
    op.drop_table('factory_job')
