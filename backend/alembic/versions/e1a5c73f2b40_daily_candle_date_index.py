"""daily_candle 单列 trade_date 索引 —— 虚拟盘每次点击都卡在这

Revision ID: e1a5c73f2b40
Revises: d4e7b9c11a83
"""
from alembic import op

revision = "e1a5c73f2b40"
down_revision = "d4e7b9c11a83"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # /api/paper/signals 每次都要 "select trade_date order by trade_date desc limit 1"
    # 取库里最新交易日。原有索引是 (ts_code, trade_date) 复合的, 光按 trade_date
    # 排序用不上, 于是全表排序 1145 万行 —— 实测扫 94 万个数据块、耗时 2.4 秒,
    # 占了"点一次下一日"总耗时的 70%。
    # 加单列索引后 2417ms -> 0.19ms, 端到端 2.25s -> 0.15s。
    # 幂等: 生产上曾用 CREATE INDEX CONCURRENTLY 手工建过, 直接 create_index
    # 会撞 DuplicateTable。
    op.execute("CREATE INDEX IF NOT EXISTS ix_daily_candle_trade_date "
               "ON daily_candle (trade_date)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_daily_candle_trade_date")
