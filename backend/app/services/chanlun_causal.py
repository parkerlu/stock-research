# [审计 2026-09] 因果版缠论 2 类买点 —— "全历史算一遍" == "逐日实时算"。
#
# 背景: chanlun.py 的批量管线存在 ZigZag 重绘: find_strokes 的"连续同类分型
# 保留更极端者"会用后来的更低底**回溯替换**已有底分型, 于是一个当时成立、
# 实时会下单的 2 类买点, 在全历史重算里消失。chan_signal 表由全历史批量构建,
# 等于只保留"事后仍成立"的信号 —— 被推翻的恰好是输家 (实测 100 只票:
# 631/1436 = 44% 的实时信号被抹掉, 其单笔期望 -1.86%, 中位数 -6% = 直接止损)。
#
# 本模块不是"冻结端点"的另一套缠论, 而是**对批量算法的因果化求值**:
# 对每个 k, 输出与 find_class2_buys(data[0:k+1]) 在右端 (bar_idx + LAG == k)
# 完全相同的结果。依据是批量管线的一个结构性质 —— 每一层列表都是
# "前缀稳定, 只有尾部可变":
#   · merge_inclusion 只原地修改 bars[-1] (或追加);
#   · 分型中心 i 依赖 bars[i-1..i+1], 故只有中心 = len(bars)-2 的分型是
#     暂定的 (右邻 bars[-1] 还会被后续合并改动), 中心 <= len(bars)-3 已定;
#   · cleaned 只有 cleaned[-1] 会被"同类更极端"替换;
#   · strokes 只有最后一笔会变; MACD hist 逐根因果 (ewm 递推)。
# 所以: 把已稳定的部分增量提交, 每根 K 只重算一小段尾部, O(n) 就能得到
# 与逐日截断重算 O(n^2) 完全一致的实时信号集。等价性已对 100 只样本票
# (2015-12 起全部可操作日) 与逐日截断复算逐一比对验证。
from __future__ import annotations

from app.services.chanlun import macd_arrays

LAG = 2          # 与 paper_trading.CONFIRM_LAG 一致
MIN_PREFIX = 130  # 与 chan_signal_build.MIN_BARS 一致: 前缀不足 130 根不出信号


def causal_class2_days(close, high, low) -> list[int]:
    """返回所有"实时口径可操作日" k 的列表:
    k 使得 find_class2_buys(data[0:k+1]) 含 bar_idx == k-LAG 的 2 买。"""
    n = len(close)
    if n == 0:
        return []
    # MACD hist 因果 (ewm adjust=False 递推只依赖过去) —— 全序列一次算好,
    # 与任意前缀上的批量计算逐位相同; 负面积用前缀和 O(1) 查询。
    _, _, hist = macd_arrays(close)
    neg_cs = [0.0] * (n + 1)
    for i in range(n):
        v = hist[i]
        neg_cs[i + 1] = neg_cs[i] + (-v if v < 0 else 0.0)

    def neg_area(a: int, b: int) -> float:
        return neg_cs[b + 1] - neg_cs[a] if b >= a else 0.0

    # ---- 合并K状态 (逐根照抄 merge_inclusion 的循环体) ----
    M_idx: list[int] = [0]
    M_hi: list[float] = [float(high[0])]
    M_lo: list[float] = [float(low[0])]
    direction = 0

    # ---- 已提交的分型/清洗序列 ----
    # cleaned 元素: (type, bar_idx, price)   type: +1 顶 / -1 底
    cleaned: list[tuple[int, int, float]] = []
    committed_center = 0     # 已提交的最大分型中心 (中心 <= len(M)-3 均已提交)

    def fractal_at(c: int):
        """判 M 中心 c 是否分型 (需 c-1, c, c+1 都存在)。"""
        if c < 1 or c + 1 >= len(M_idx):
            return None
        ah, al = M_hi[c - 1], M_lo[c - 1]
        bh, bl = M_hi[c], M_lo[c]
        ch, cl = M_hi[c + 1], M_lo[c + 1]
        if bh > ah and bh > ch and bl > al and bl > cl:
            return (+1, M_idx[c], bh)
        if bl < al and bl < cl and bh < ah and bh < ch:
            return (-1, M_idx[c], bl)
        return None

    def fold(seq: list, f: tuple) -> None:
        """find_strokes 的 cleaned 处理规则 (min_bars=5), 原地作用于 seq."""
        if not seq:
            seq.append(f); return
        pt, pi, pp = seq[-1]
        ft, fi, fp = f
        if pt == ft:
            if ft == +1 and fp > pp:
                seq[-1] = f
            elif ft == -1 and fp < pp:
                seq[-1] = f
        else:
            if fi - pi >= 5:
                seq.append(f)

    def c1_pair(pv, cur) -> bool:
        """批量 find_class1_buys 中相邻两段下笔 (pv, cur) 是否构成 1 买。
        笔用 (start_idx, start_price, end_idx, end_price) 表示。"""
        ps, psp, pe, pep = pv
        cs, csp, ce, cep = cur
        if cep >= pep * 1.001:
            return False
        pa = neg_area(ps, pe)
        ca = neg_area(cs, ce)
        if pa <= 0 or ca >= pa * 0.95:
            return False
        if cs - pe > 90:
            return False
        return True

    out: list[int] = []

    for k in range(1, n):
        h, lo = float(high[k]), float(low[k])
        lh, ll = M_hi[-1], M_lo[-1]
        appended = False
        if (lh >= h and ll <= lo) or (h >= lh and lo <= ll):
            # 含包合并 —— 只改最后一根
            if direction >= 0:
                M_hi[-1] = max(lh, h); M_lo[-1] = max(ll, lo)
            else:
                M_hi[-1] = min(lh, h); M_lo[-1] = min(ll, lo)
            M_idx[-1] = k
        else:
            if h > lh:
                direction = 1
            elif h < lh:
                direction = -1
            M_idx.append(k); M_hi.append(h); M_lo.append(lo)
            appended = True

        L = len(M_idx)
        if appended and L >= 4:
            # 中心 L-3 的右邻 (L-2) 已定型 —— 提交它的最终判定
            c = L - 3
            if c > committed_center:
                f = fractal_at(c)
                if f:
                    fold(cleaned, f)
                committed_center = c

        if k < MIN_PREFIX - 1:
            continue

        # ---- 本 k 的有效尾部: 已提交 cleaned + 暂定分型 (中心 L-2) ----
        eff_tail = cleaned[-12:]
        base = len(cleaned) - len(eff_tail)     # eff_tail[0] 的绝对位置
        prov = fractal_at(L - 2)
        if prov is not None:
            eff_tail = list(eff_tail)
            fold(eff_tail, prov)
        m = len(eff_tail)
        total = base + m                        # 有效 cleaned 总长
        # 笔 j (绝对下标) 连接 cleaned[j] -> cleaned[j+1], 共 total-1 笔。
        # 右端 2 买要检查 s_now 为最后几笔的情形 (暂定分型可能让新笔出现,
        # 使 bar_idx == k-2 的笔不再是最后一笔)。
        for jj in range(max(0, m - 4), m - 1):
            # s_now: eff_tail[jj] -> eff_tail[jj+1]
            aT, aI, aP = eff_tail[jj]
            bT, bI, bP = eff_tail[jj + 1]
            if not (aT == +1 and bT == -1):      # down 笔: 顶 -> 底
                continue
            if bI != k - LAG:
                continue
            j_abs = base + jj                    # s_now 的绝对笔号 = base+jj
            # 需要 s_prev = 笔 j_abs-2 (down), s_mid = 笔 j_abs-1 (up),
            # 且批量 c2 循环要求 j_abs >= 2
            if j_abs < 2 or jj < 2:
                continue                         # 笔数不足 (12 根尾窗下只在极早期出现)
            pT, pI, pP = eff_tail[jj - 2]
            qT, qI, qP = eff_tail[jj - 1]
            if not (pT == +1 and qT == -1):      # s_prev 必须 down (顶->底->顶->底)
                continue
            # 新低必须高于 1 类买点低点
            if bP <= qP:
                continue
            # s_prev 的 c1 判定: 它的前一段下笔是 j_abs-4 -> 端点 eff_tail[jj-4..jj-3]
            if j_abs - 4 < 0 or jj - 4 < 0:
                continue
            rT, rI, rP = eff_tail[jj - 4]
            sT, sI, sP = eff_tail[jj - 3]
            if not (rT == +1 and sT == -1):
                continue
            # 批量 find_class1_buys 还有 len(strokes) < 3 直接返回空 ——
            # 这里 j_abs-4 >= 0 已保证至少 5 笔, 条件更严, 不冲突。
            if not c1_pair((rI, rP, sI, sP), (pI, pP, qI, qP)):
                continue
            out.append(k)
            break
    return out
