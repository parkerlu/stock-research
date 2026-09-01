# 策略筛选台架（四道闸）

为避免重蹈 chan-2buy 的覆辙而建。那次 10.7 年 229.7 倍全部来自三条未来函数，
修完只剩 0.72 倍。教训是：**任何一道闸缺席，都能独立造出一个假的好结果。**

## 用法

脚本要在 backend 容器的 `/app` 下跑（`sys.path` 的关系）：

```bash
docker compose cp backend/scripts/screening/screen.py backend:/app/
docker compose exec -T -w /app backend python -u screen.py          # 快批 22 个模板
docker compose exec -T -w /app backend python -u screen.py --slow   # 慢批 20 个
docker compose exec -T -w /app backend python -u causal_check.py    # S1
docker compose exec -T -w /app backend python -u oos.py '["模板名"]' # S3
docker compose exec -T -w /app backend python -u s4.py 模板名 [mid]  # S4
```

## 四道闸

| 闸 | 脚本 | 淘汰线 | 为什么必要 |
|---|---|---|---|
| ① 因果性 | `causal_check.py` | 撤销率 > 2% | 缠论 ZigZag 重绘让约一半信号事后消失，且消失的正是输家 |
| ② 随机基准 | `screen.py` | 超额 < 随机种子极差 | 阶梯出场规则本身就能凭空刷出结构性超额 |
| ③ 多重检验 | `screen.py` | \|t\| < 3.2 | 测 42 个挑最好的，5% 显著性下平均有 2 个纯属运气 |
| ④ 组合层面 | `s4.py` | z < 2.5 或非 12/12 全胜 | 单笔超额换年化的乘法假设仓位永不空闲；10 仓位下未必兑现 |

## 三个必须手动确认的坑

**0. 多周期穿越（最隐蔽）。** 用到周线/月线的策略，必须确认"当周未完成时不
引用本周数据"。`tdx_classics` 原来的写法把整周聚合值广播回该周每一天，于是
周一就用到了周五收盘 —— 偷看 4 天。跑 `week_leak.py`：它在每周第 1~4 天切，
若第 1~4 天差异显著而第 5 天为 0，即为穿越。修复前 4.16/3.49/2.45/2.06/0.00，
修复后全 0。

⚠️ 通用的 `causal_check.py` **测不出这个**：随机切点下信号落进残缺周的概率
太低，4129 个信号只报 0.05% 不一致，低检验力会被误读成干净。切点已从 6 加到
24，但多周期策略仍须单独跑 `week_leak.py`。


**1. 模型训练期。** 加载训练好的模型的模板（`mm-*` 读 maimai_filter_multi.json、
`rev-*` 读 reversal_filter_multi.json，截止 2024-09-01；`ml_direct_*` 用
GroupKFold 按**股票**分组、模型 2026-04 生成 = 见过全部历史），它们的"样本外"
必须落在训练截止之后，否则是模型在自己的训练集上考试。切过去后这批全部崩塌。

**2. 随机对照的种子数。** 10 仓位 + 2 年窗口下，3 个种子的极差就有 37.7pp——
比策略间的差异还大。至少 12 个种子才能把噪声底量准。

## 当前结论

唯一通过全部四道闸的是 `tdx-dual-kdj`（日周 KDJ 共振，纯公式无训练）。
两个互不重叠窗口：熊市 +41.13%（随机 −8.91%，z=7.62）、反弹 +59.34%
（随机 +17.81%，z=3.58），均 12/12 全胜。
