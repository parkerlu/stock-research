# TDX 指标接入 设计文档

**日期**: 2026-04-23
**目标**: 在 K 线图上接入通达信（TDX）风格的技术指标，第一个目标指标是"多周期 KDJ 共振"。
**非目标**: 通用 TDX 公式解析器；用户自由粘贴公式；ML 模型训练。

---

## 1. 背景与动机

PRD 第 16 节规划了"通达信复合指标接入"。当前系统只有 8 个内置规则策略（MA/RSI/MACD/Bollinger/KDJ 及组合），对用户实际交易风格支持不足。用户提供了一个多周期 KDJ 共振指标的 TDX 公式，期望在 K 线图上可视化该指标。

调研结论：GitHub 上有成熟的 TDX 函数库（MyTT 2.7k 星、funcat 1k 星），但：
- 它们只是函数库（不是公式解析器）
- 都不处理绘图函数（STICKLINE/DRAWICON）
- 唯一的通用解析器 `tdx_formula` 已 9 年未更新

因此本次不走"通用解析器"路线，而采用"手工移植 + 自建基础设施"路线，交付最小可用版本并为后续指标奠定可扩展的基础。

## 2. 范围

### 本期交付（In Scope）

1. **TDX 基础函数库**（~15 个函数，自研，避免 GPL 污染）
   - `EMA, SMA, MA, HHV, LLV, CROSS, REF, IF, ABS, MAX, MIN, SUM, COUNT, EVERY, EXIST`
2. **指标运行框架**
   - 统一的 `IndicatorResult` 数据结构
   - 指标注册表（registry）
   - REST API：`GET /api/indicators` 和 `GET /api/indicators/{name}`
3. **第一个指标：多周期 KDJ 共振**
   - 计算 4 个周期（13/21/34/55）的 K/D 值
   - 仅渲染 K13/D14/K55/D55 四条可见线（遵循原 TDX 公式约定）
   - K21/D21/K34/D34 为内部计算值，用于共振检测
   - 横向参考线：0、20、50、80、100
   - 顶部共振（TOBU）底部共振（DIBU）高亮柱
4. **前端渲染层**
   - 基于 klinecharts `registerIndicator` API 的动态指标渲染
   - Lines / HLines / Bands / Markers 四种视觉元素支持
   - 工具栏新增 "TDX 指标 ▾" 下拉菜单

### 非本期（Out of Scope）

- 动力线（阶段指标）— 后续独立 spec
- 信号灯指标 — 后续独立 spec
- 用户自由粘贴 TDX 公式的 UI 输入框
- TDX 指标作为回测策略参与策略工厂
- 多周期 KDJ 用于生成交易信号（仅作可视化）

## 3. 系统架构

### 3.1 代码组织

```
backend/
├── app/
│   ├── services/tdx/
│   │   ├── __init__.py
│   │   ├── functions.py                   # TDX 基础函数库
│   │   ├── indicators/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                    # IndicatorResult + 数据类
│   │   │   └── multi_kdj.py               # 多周期 KDJ 实现
│   │   └── registry.py                    # 指标注册表
│   └── routers/
│       └── indicators.py                  # REST API

backend/tests/
└── services/tdx/
    ├── test_functions.py                  # 基础函数单测
    └── test_multi_kdj.py                  # 多周期 KDJ 单测

frontend/
├── src/
│   ├── api/indicators.ts                  # API client
│   ├── types/indicator.ts                 # TypeScript 类型
│   └── components/ChartArea/
│       ├── TdxIndicatorManager.ts         # 动态注册/移除指标
│       ├── tdxIndicatorRenderer.ts        # Canvas 绘制 bands/markers/hlines
│       └── Toolbar.tsx                    # 增加 "TDX 指标 ▾" 菜单
```

### 3.2 数据流

```
用户点击 "TDX 指标 > 多周期 KDJ"
  ↓
Frontend: GET /api/indicators/multi_kdj?ts_code=600519.SH&tf=1d&from=...&to=...
  ↓
Backend Router: indicators.py
  ↓
registry.get("multi_kdj") → multi_kdj.compute(ohlcv_df)
  ↓
multi_kdj.py 使用 functions.py 的 SMA/HHV/LLV/CROSS 计算各周期 K/D
  ↓
返回 IndicatorResult JSON
  ↓
Frontend: TdxIndicatorManager 调用 klinecharts.registerIndicator 注册动态指标
  ↓
chart.createIndicator("TDX_multi_kdj", true) 在新面板渲染
```

### 3.3 核心数据结构

```python
# backend/app/services/tdx/indicators/base.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class IndicatorLine:
    name: str                    # 如 "K13"
    values: list[float | None]   # 与 K 线数组等长，前置无效值用 None
    color: str                   # hex 色号 "#4caf50"
    thickness: int = 1

@dataclass
class IndicatorHLine:
    name: str                    # 如 "顶部"
    value: float                 # 如 80
    color: str
    dashed: bool = False         # 对应 TDX 的 POINTDOT

@dataclass
class IndicatorBand:
    # 每个元素是一个 bar 位置的矩形
    timestamps: list[int]        # UTC 毫秒时间戳
    y1: float                    # 矩形 Y 坐标下边界
    y2: float                    # 矩形 Y 坐标上边界
    color: str
    opacity: float = 0.3

@dataclass
class IndicatorMarker:
    timestamp: int               # UTC 毫秒
    value: float                 # Y 坐标
    color: str
    icon: Literal["dot", "triangle_up", "triangle_down"] = "dot"

@dataclass
class IndicatorResult:
    name: str                    # "multi_kdj"
    label: str                   # "多周期 KDJ 共振"
    pane: Literal["main", "sub"] = "sub"
    y_axis_range: tuple[float, float] | None = None  # 如 (0, 100)
    lines: list[IndicatorLine] = field(default_factory=list)
    hlines: list[IndicatorHLine] = field(default_factory=list)
    bands: list[IndicatorBand] = field(default_factory=list)
    markers: list[IndicatorMarker] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

## 4. TDX 基础函数库

`backend/app/services/tdx/functions.py` — 自研，参考 MyTT 实现方式，基于 pandas/numpy。

### 4.1 需实现的函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `MA` | `MA(series, n)` | 简单移动平均 |
| `EMA` | `EMA(series, n)` | 指数移动平均 α=2/(n+1) |
| `SMA` | `SMA(series, n, m=1)` | **中国版** SMA，递推公式 `Y = (m*X + (n-m)*prev_Y) / n` |
| `HHV` | `HHV(series, n)` | 滚动最大值 |
| `LLV` | `LLV(series, n)` | 滚动最小值 |
| `REF` | `REF(series, n)` | 向前取值 n 期 |
| `CROSS` | `CROSS(a, b)` | `a` 上穿 `b` 返回 1，其他为 0 |
| `IF` | `IF(cond, yes, no)` | 三元表达式的向量化版本 |
| `ABS` | `ABS(series)` | 绝对值 |
| `MAX` | `MAX(a, b)` | 按位最大值 |
| `MIN` | `MIN(a, b)` | 按位最小值 |
| `SUM` | `SUM(series, n)` | 滚动 n 期求和 |
| `COUNT` | `COUNT(cond, n)` | 条件在 n 期内为真的次数 |
| `EVERY` | `EVERY(cond, n)` | 条件是否在 n 期内全部为真 |
| `EXIST` | `EXIST(cond, n)` | 条件是否在 n 期内至少一次为真 |

### 4.2 关键实现要点

1. **SMA 中国版**：
   ```python
   def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
       # Y_today = (m * X_today + (n - m) * Y_yesterday) / n
       alpha = m / n
       return series.ewm(alpha=alpha, adjust=False).mean()
   ```
   这是 TDX 最容易被误实现的函数 — 等价于 EMA with `α = M/N`，不是标准 SMA。

2. **CROSS**：
   ```python
   def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
       return ((a > b) & (a.shift(1) <= b.shift(1))).astype(int)
   ```

3. **前置 NaN**：所有滚动函数（MA/HHV/LLV/REF）的前 N-1 个值为 NaN，API 层在序列化时转换为 JSON `null`。

## 5. 第一个指标：多周期 KDJ 共振

### 5.1 算法规约

```python
# backend/app/services/tdx/indicators/multi_kdj.py
def compute(df: pd.DataFrame) -> IndicatorResult:
    """
    输入：df 包含 columns [timestamp, open, high, low, close]
    最小数据量：55 个 bar（否则 55 周期 KDJ 无法计算）
    """
    close, high, low = df["close"], df["high"], df["low"]

    # 四个周期的 KDJ
    kdj_params = [(13, "K13", "D14"), (21, "K21", "D21"), (34, "K34", "D34"), (55, "K55", "D55")]
    kdj_data = {}
    for period, k_name, d_name in kdj_params:
        var = (close - LLV(low, period)) / (HHV(high, period) - LLV(low, period)) * 100
        sma_window = 5 if period == 55 else 3  # 55 周期用 5 日 SMA
        k = SMA(var, sma_window, 1)
        d = SMA(k, sma_window, 1)
        kdj_data[k_name] = k
        kdj_data[d_name] = d

    # 共振检测
    dibu = (kdj_data["K13"] < 20) & (kdj_data["D14"] < 20) & \
           (kdj_data["K21"] < 20) & (kdj_data["D21"] < 20) & \
           (kdj_data["K34"] < 20) & (kdj_data["D34"] < 20) & \
           (kdj_data["K55"] < 20) & (kdj_data["D55"] < 20)
    tobu = (kdj_data["K13"] > 80) & (kdj_data["D14"] > 80) & \
           (kdj_data["K21"] > 80) & (kdj_data["D21"] > 80) & \
           (kdj_data["K34"] > 80) & (kdj_data["D34"] > 80) & \
           (kdj_data["K55"] > 80) & (kdj_data["D55"] > 80)

    # 组装 IndicatorResult
    return IndicatorResult(
        name="multi_kdj",
        label="多周期 KDJ 共振",
        pane="sub",
        y_axis_range=(0, 100),
        # 遵循原 TDX 公式约定：仅 K13/D14/K55/D55 为可见输出，
        # K21/D21/K34/D34 仅用于 DIBU/TOBU 共振检测，不渲染
        lines=[
            IndicatorLine("K13", kdj_data["K13"].tolist(), "#FF6B6B"),
            IndicatorLine("D14", kdj_data["D14"].tolist(), "#FFE66D"),
            IndicatorLine("K55", kdj_data["K55"].tolist(), "#00FF00", thickness=1),
            IndicatorLine("D55", kdj_data["D55"].tolist(), "#FF9933", thickness=1),
        ],
        hlines=[
            IndicatorHLine("天线", 100, "#FFFFFF", dashed=True),
            IndicatorHLine("顶部", 80, "#55AA77"),
            IndicatorHLine("中轴线", 50, "#FFFFFF", dashed=True),
            IndicatorHLine("底部", 20, "#55AA77"),
            IndicatorHLine("底线", 0, "#FFFFFF", dashed=True),
        ],
        bands=[
            # 底部共振：红色柱
            IndicatorBand(
                timestamps=df.loc[dibu, "timestamp"].tolist(),
                y1=60, y2=80, color="#FF0000", opacity=0.6,
            ),
            # 顶部共振：绿色柱
            IndicatorBand(
                timestamps=df.loc[tobu, "timestamp"].tolist(),
                y1=20, y2=40, color="#00FF00", opacity=0.6,
            ),
        ],
    )
```

### 5.2 数值一致性验证

关键测试：用一段真实历史数据（如 600519.SH 2024 年全年）手工在通达信软件算出 K55/D55 的部分值，对比 Python 实现。允许浮点误差 < 0.01。

## 6. REST API

### 6.1 路由

```python
# backend/app/routers/indicators.py

@router.get("/api/indicators", response_model=list[IndicatorMeta])
def list_indicators() -> list[IndicatorMeta]:
    """返回所有已注册指标的元信息"""
    return [IndicatorMeta(name=m.name, label=m.label, pane=m.pane) for m in registry.all()]

@router.get("/api/indicators/{name}")
def get_indicator(name: str, ts_code: str, tf: str = "1d",
                  from_: date = Query(alias="from"), to: date = Query(...)) -> IndicatorResult:
    """计算并返回指定指标的数据"""
    impl = registry.get(name)  # 404 if not found
    df = fetch_ohlcv(ts_code, tf, from_, to)  # 复用 quote_service
    if len(df) < impl.min_bars:
        return IndicatorResult(
            name=name, label=impl.label,
            warnings=[f"数据不足，需要至少 {impl.min_bars} 个 bar，当前只有 {len(df)} 个"]
        )
    return impl.compute(df)
```

### 6.2 响应示例

```json
{
  "name": "multi_kdj",
  "label": "多周期 KDJ 共振",
  "pane": "sub",
  "y_axis_range": [0, 100],
  "lines": [
    {"name": "K13", "values": [null, null, ..., 45.2, 48.1, 52.3], "color": "#FF6B6B", "thickness": 1}
  ],
  "hlines": [
    {"name": "顶部", "value": 80, "color": "#55AA77", "dashed": false}
  ],
  "bands": [
    {"timestamps": [1704067200000, 1704153600000], "y1": 60, "y2": 80, "color": "#FF0000", "opacity": 0.6}
  ],
  "markers": [],
  "warnings": []
}
```

## 7. 前端实现

### 7.1 动态指标注册

```typescript
// frontend/src/components/ChartArea/TdxIndicatorManager.ts
import { registerIndicator } from "klinecharts";

const registered = new Set<string>();

export function ensureRegistered(result: IndicatorResult): string {
  const key = `TDX_${result.name}`;
  if (registered.has(key)) return key;

  registerIndicator({
    name: key,
    shortName: result.label,
    calcParams: [],
    figures: result.lines.map((l) => ({
      key: l.name, title: `${l.name}: `, type: "line",
    })),
    calc: (dataList) => {
      // 将 IndicatorResult 的 lines 按时间戳对齐到 dataList
      return alignLinesToKline(result.lines, dataList);
    },
    draw: ({ ctx, xAxis, yAxis, barSpace, visibleRange, chartStore }) => {
      drawHLines(ctx, result.hlines, xAxis, yAxis);
      drawBands(ctx, result.bands, xAxis, yAxis, barSpace);
      drawMarkers(ctx, result.markers, xAxis, yAxis);
      return false; // 继续让默认 figures 绘制线条
    },
  });

  registered.add(key);
  return key;
}
```

### 7.2 工具栏入口

在 `Toolbar.tsx` 增加新下拉菜单："TDX 指标 ▾"，选项由 `GET /api/indicators` 动态获取。选择后：
1. 调用 `GET /api/indicators/{name}?...`
2. `ensureRegistered(result)` 注册
3. `chart.createIndicator(key, true)` 创建子面板

再次点击同一项时移除（切换行为）。

### 7.3 Canvas 绘制细节

- **HLines**：横跨整个面板宽度的水平线，`ctx.strokeStyle = color`，`dashed` 时用 `ctx.setLineDash([4, 4])`
- **Bands**：对每个 timestamp 查找 `xAxis` 坐标，用 `ctx.fillRect(x - barSpace/2, y2, barSpace, y1 - y2)` 绘制
- **Markers**：根据 icon 类型绘制圆点/三角形

## 8. 测试策略

### 8.1 单元测试

`tests/services/tdx/test_functions.py`：
- `MA / EMA`：与 pandas rolling mean / ewm 结果对比
- `SMA`：用已知的 `SMA([1,2,3,4,5], 3, 1)` 递推公式手算结果对比
- `HHV / LLV`：滚动窗口边界条件（窗口大于序列长度）
- `CROSS`：构造金叉/死叉案例，验证返回的 0/1 序列
- `REF / IF / ABS / MAX / MIN / SUM`：基础案例

`tests/services/tdx/test_multi_kdj.py`：
- 固定随机种子生成 200 个 bar 的 OHLCV
- 验证 `IndicatorResult` 结构完整性（4 条 lines、5 条 hlines、2 个 bands）
- 验证 `K55/D55` 在末尾几个值与手工计算一致
- 数据不足（<55 个 bar）时返回 warning

### 8.2 集成测试

- `GET /api/indicators` 返回包含 `multi_kdj` 的列表
- `GET /api/indicators/multi_kdj?ts_code=600519.SH&tf=1d&from=2024-01-01&to=2024-12-31` 返回合法 JSON

### 8.3 手工验收

1. 打开贵州茅台 → 工具栏 "TDX 指标 ▾" → 选"多周期 KDJ 共振"
2. K 线图下方出现新面板，4 条可见 K/D 线（K13 粉红、D14 黄、K55 绿粗线、D55 橙）颜色正确
3. 80 和 20 处有绿色水平线，0/50/100 有白色虚线
4. 回溯到 2024 年初低位（股价跌到 1322 附近），指标面板底部出现红色柱
5. 再次点击菜单 → 指标移除

## 9. 实施顺序（供 writing-plans 参考）

1. Backend: `tdx/functions.py` + 单测
2. Backend: `tdx/indicators/base.py` 数据结构
3. Backend: `tdx/indicators/multi_kdj.py` + 单测
4. Backend: `tdx/registry.py` + `routers/indicators.py` + API 集成测试
5. Frontend: `types/indicator.ts` + `api/indicators.ts`
6. Frontend: `TdxIndicatorManager.ts` + `tdxIndicatorRenderer.ts`
7. Frontend: `Toolbar.tsx` 菜单集成
8. 手工验收 + 集成测试

## 10. 风险与开放问题

| 风险 | 缓解方式 |
|------|----------|
| SMA 中国版误实现导致与通达信数值不一致 | 用 MyTT 测试用例对比验证 |
| klinecharts `registerIndicator` + 自定义 `draw` 的 API 不稳定 | 预留 Canvas 原生绘制回退方案 |
| 大数据量（>1000 bar）渲染卡顿 | 初期数据量默认不超过一年（~250 bar），后续按需优化 |
| 前端缓存问题：切换股票后指标数据未刷新 | Manager 在 ts_code 变化时调用 `chart.removeIndicator` 再重新获取 |

## 11. 未来扩展（不属于本期）

- **更多指标**：动力线、信号灯、MACD 金叉密度、量价异常探测
- **选股器**：基于 TDX 公式的多股扫描（TDX 公式标注满足条件的股票）
- **指标用于回测**：把 TDX 指标的 BUY/SELL 信号接入策略工厂作为候选策略
