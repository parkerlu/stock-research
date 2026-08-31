"""虚拟盘 (paper trading) 表

按验证过的 chan-2buy 最优配置做实盘跟踪:
  10 万本金 / 10 仓位等权 / 止损6% / +4%出半(保本上移) / +8%清仓

四张表:
  paper_account   账户与策略参数快照 (改参数应新建账户)
  paper_position  持仓, 分批止盈后 shares 递减
  paper_trade     全部动作流水
  paper_equity    每日净值快照

Revision ID: e5b1c8f2a731
Revises: d4e7c1a92f03
Create Date: 2026-08-31
"""
from alembic import op
import sqlalchemy as sa

revision = "e5b1c8f2a731"
down_revision = "d4e7c1a92f03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_account",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(60), nullable=False, unique=True),
        sa.Column("initial_capital", sa.Numeric(16, 2), nullable=False),
        sa.Column("cash", sa.Numeric(16, 2), nullable=False),
        sa.Column("slots", sa.Integer, nullable=False, server_default="10"),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("started_on", sa.Date, nullable=False),
        sa.Column("last_run_date", sa.Date),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "paper_position",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.Integer, nullable=False, index=True),
        sa.Column("ts_code", sa.String(12), nullable=False, index=True),
        sa.Column("name", sa.String(64)),
        sa.Column("open_date", sa.Date, nullable=False),
        sa.Column("open_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("init_shares", sa.Integer, nullable=False),
        sa.Column("shares", sa.Integer, nullable=False),
        sa.Column("stop_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("tier1_done", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("tier2_done", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("realized_pnl", sa.Numeric(16, 2), nullable=False, server_default="0"),
        sa.Column("status", sa.String(10), nullable=False, server_default="open"),
        sa.Column("close_date", sa.Date),
        sa.Column("close_reason", sa.String(20)),
    )
    op.create_index("ix_paper_pos_acct", "paper_position", ["account_id", "status"])
    op.create_table(
        "paper_trade",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.Integer, nullable=False, index=True),
        sa.Column("position_id", sa.Integer, index=True),
        sa.Column("ts_code", sa.String(12), nullable=False, index=True),
        sa.Column("name", sa.String(64)),
        sa.Column("trade_date", sa.Date, nullable=False, index=True),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("shares", sa.Integer, nullable=False),
        sa.Column("amount", sa.Numeric(16, 2), nullable=False),
        sa.Column("fee", sa.Numeric(12, 2), nullable=False, server_default="0"),
        sa.Column("pnl", sa.Numeric(16, 2)),
        sa.Column("pnl_pct", sa.Numeric(10, 4)),
        sa.Column("note", sa.Text),
    )
    op.create_index("ix_paper_trade_acct", "paper_trade", ["account_id", "trade_date"])
    op.create_table(
        "paper_equity",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.Integer, nullable=False, index=True),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column("cash", sa.Numeric(16, 2), nullable=False),
        sa.Column("market_value", sa.Numeric(16, 2), nullable=False),
        sa.Column("equity", sa.Numeric(16, 2), nullable=False),
        sa.Column("n_positions", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_paper_eq", "paper_equity", ["account_id", "trade_date"], unique=True)


def downgrade() -> None:
    op.drop_table("paper_equity")
    op.drop_table("paper_trade")
    op.drop_table("paper_position")
    op.drop_table("paper_account")
