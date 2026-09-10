"""strategy_pool —— 策略的唯一定义源; 同时清掉旧的 strategy 工厂数据

Revision ID: e5a2b71c4d09
Revises: d8f1c93a2e57
"""
from alembic import op
import sqlalchemy as sa

revision = "e5a2b71c4d09"
down_revision = "d8f1c93a2e57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_pool",
        sa.Column("key", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(60), nullable=False),
        sa.Column("summary", sa.String(200), nullable=False),
        sa.Column("detail", sa.Text, nullable=False),
        sa.Column("since", sa.String(7), nullable=False),
        sa.Column("ratio", sa.Numeric(6, 2)),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("slots", sa.Integer, nullable=False, server_default="20"),
        sa.Column("is_live", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="100"),
    )
    op.create_index("ix_pool_live", "strategy_pool", ["is_live"])
    # ⚠️ 用户 2026-09-10 决定: 策略工厂那 101 条参数组合连数据一起清掉。
    #    那是"同一个模板在单只票上调参"的产物, 与现在这套"全市场模型选股"
    #    不是一回事, 留着只会让人以为策略池里有一百多个策略。
    op.execute("delete from strategy")


def downgrade() -> None:
    op.drop_index("ix_pool_live", table_name="strategy_pool")
    op.drop_table("strategy_pool")
