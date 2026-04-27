"""Strategy pool — JSON-backed metadata + visibility/order layer over TEMPLATE_REGISTRY.

A pool entry references a template_id from TEMPLATE_REGISTRY and adds:
  display_name, concept, family, description, is_active, sort_order, metrics.

On first read the pool auto-seeds from TEMPLATE_REGISTRY + the v2 mining REPORT.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.services.strategy_templates import TEMPLATE_REGISTRY

POOL_PATH = Path("/app/data/strategy_pool.json")
POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()


# ----- Seed metadata for known templates (matches v2 REPORT) ---------
_SEED_META: dict[str, dict] = {
    "426-1": {"concept": "动力线", "family": "D02_dl_cross_05", "description": "动力线 上穿 0.5",
              "metrics": {"is_win": 84.6, "is_avg": 6.7, "oos_win": 87.8, "oos_avg": 7.3, "trades": 3804}},
    "426-2": {"concept": "KDJ", "family": "K03_kd_cross", "description": "KDJ K/D 金叉",
              "metrics": {"is_win": 76.9, "is_avg": 7.6, "oos_win": 80.7, "oos_avg": 7.3, "trades": 4021}},
    "426-3": {"concept": "RSI", "family": "R01_rsi_30", "description": "RSI 上穿 30 (超卖反弹)",
              "metrics": {"is_win": 75.1, "is_avg": 7.3, "oos_win": 79.3, "oos_avg": 8.1, "trades": 3448}},
    "426-4": {"concept": "KDJ", "family": "K02_j_cross_20", "description": "KDJ J 上穿 20",
              "metrics": {"is_win": 80.8, "is_avg": 9.6, "oos_win": 85.7, "oos_avg": 10.7, "trades": 2792}},
    "426-5": {"concept": "动力线", "family": "D03_dl_cross_10", "description": "动力线 上穿 1.0",
              "metrics": {"is_win": 78.6, "is_avg": 9.0, "oos_win": 83.2, "oos_avg": 10.0, "trades": 1176}},
    "426-6": {"concept": "SMA", "family": "S01_sma_5_20", "description": "SMA 5/20 金叉",
              "metrics": {"is_win": 94.9, "is_avg": 7.3, "oos_win": 91.7, "oos_avg": 16.5, "trades": 158}},
    "426-7": {"concept": "Breakout", "family": "BR01_high20", "description": "20日高点突破",
              "metrics": {"is_win": 77.1, "is_avg": 7.2, "oos_win": 77.3, "oos_avg": 11.7, "trades": 153}},
    "426-8": {"concept": "SMA", "family": "S01_sma_5_20", "description": "SMA 5/20 慢版",
              "metrics": {"is_win": 90.6, "is_avg": 8.7, "oos_win": 93.8, "oos_avg": 23.6, "trades": 106}},
    "426-9": {"concept": "Breakout", "family": "BR02_high50", "description": "50日高点突破",
              "metrics": {"is_win": 80.4, "is_avg": 7.9, "oos_win": 77.8, "oos_avg": 11.6, "trades": 97}},
    "426-10": {"concept": "BB", "family": "B03_bb_upper", "description": "Bollinger 上轨突破",
              "metrics": {"is_win": 77.6, "is_avg": 6.8, "oos_win": 82.1, "oos_avg": 13.3, "trades": 98}},
    "mm-30": {"concept": "买卖很准", "family": "MaimaiFilter", "description": "高频 thr=0.30 + 贪婪 ATR",
              "metrics": {}},
    "mm-40": {"concept": "买卖很准", "family": "MaimaiFilter", "description": "中频 thr=0.40 + 贪婪 ATR",
              "metrics": {}},
    "mm-50": {"concept": "买卖很准", "family": "MaimaiFilter", "description": "Sweet thr=0.50 + 贪婪 ATR",
              "metrics": {}},
    "mm-55": {"concept": "买卖很准", "family": "MaimaiFilter", "description": "严格 thr=0.55 + 贪婪 ATR (吃最大波段)",
              "metrics": {}},
    "mm-broad-40": {"concept": "买卖很准", "family": "MaimaiBroad", "description": "5 信号 OR 触发 + thr=0.40",
              "metrics": {}},
    "mm-broad-50": {"concept": "买卖很准", "family": "MaimaiBroad", "description": "5 信号 OR 触发 + thr=0.50",
              "metrics": {}},
    "mm-pure-40": {"concept": "买卖很准", "family": "MaimaiPure", "description": "纯 mm 双向: buy 信号关闭买入, sell 信号触发卖出",
              "metrics": {}},
    "mm-pure-50": {"concept": "买卖很准", "family": "MaimaiPure", "description": "纯 mm 双向严格版 thr=0.50",
              "metrics": {}},
    "rev-30": {"concept": "反转融合", "family": "Reversal", "description": "高频 thr=0.30 (~10/yr 候选)",
              "metrics": {}},
    "rev-40": {"concept": "反转融合", "family": "Reversal", "description": "买卖很准+动力线+KDJ+RSI thr=0.40",
              "metrics": {}},
    "rev-50": {"concept": "反转融合", "family": "Reversal", "description": "Sweet thr=0.50 + 贪婪 ATR (10/年)",
              "metrics": {}},
    "rev-55": {"concept": "反转融合", "family": "Reversal", "description": "严格 thr=0.55 + ATR(2.5)",
              "metrics": {}},
    "rev-60": {"concept": "反转融合", "family": "Reversal", "description": "极严 thr=0.60",
              "metrics": {}},
    "ml_direct_atr_swing": {"concept": "ML", "family": "ATR Swing", "description": "ATR 自适应吃波段",
              "metrics": {}},
    "ml_direct_classic_greedy": {"concept": "ML", "family": "Classic Greedy",
              "description": "经典 trail，78.6% 胜率", "metrics": {}},
    "ml_direct_top1_tight_lock": {"concept": "ML", "family": "Top1 Tight Lock", "description": "", "metrics": {}},
    "ml_direct_top2_mid_greedy": {"concept": "ML", "family": "Top2 Mid Greedy", "description": "", "metrics": {}},
    "ml_direct_top3_high_greedy": {"concept": "ML", "family": "Top3 High Greedy", "description": "", "metrics": {}},
    "ml_direct_fast_turnover": {"concept": "ML", "family": "Fast Turnover", "description": "", "metrics": {}},
    "ml_direct_high_freq_v2": {"concept": "ML", "family": "High Freq V2", "description": "", "metrics": {}},
    "ml_direct_max_freq": {"concept": "ML", "family": "Max Freq", "description": "高频", "metrics": {}},
    "ml_direct_high_freq": {"concept": "ML", "family": "High Freq Legacy", "description": "", "metrics": {}},
    "ml_direct": {"concept": "ML", "family": "Decision (Legacy)", "description": "", "metrics": {}},
}


def _seed() -> list[dict]:
    """Build initial pool from TEMPLATE_REGISTRY + seed metadata."""
    out = []
    # Order: 426-1..5 first (seed metadata order), then ml_direct_*
    order_keys = [k for k in _SEED_META.keys() if k in TEMPLATE_REGISTRY]
    for k in TEMPLATE_REGISTRY:
        if k not in order_keys:
            order_keys.append(k)
    for i, tid in enumerate(order_keys):
        cls = TEMPLATE_REGISTRY[tid]
        meta = _SEED_META.get(tid, {})
        try:
            display_name = cls().name
        except Exception:
            display_name = cls.__name__
        out.append({
            "template_id": tid,
            "display_name": display_name,
            "concept": meta.get("concept", "Other"),
            "family": meta.get("family", ""),
            "description": meta.get("description", ""),
            "metrics": meta.get("metrics", {}),
            "is_active": True,
            "sort_order": i,
            "source": "preset",
        })
    return out


def _load() -> list[dict]:
    if not POOL_PATH.exists():
        entries = _seed()
        _save(entries)
        return entries
    try:
        with POOL_PATH.open() as f:
            data = json.load(f)
        entries = data.get("entries", [])
        # Reconcile with TEMPLATE_REGISTRY: drop entries whose class no longer exists,
        # add any new templates not yet in pool (appended at end, active by default).
        known = {e["template_id"] for e in entries}
        for tid in TEMPLATE_REGISTRY:
            if tid not in known:
                cls = TEMPLATE_REGISTRY[tid]
                meta = _SEED_META.get(tid, {})
                try:
                    display_name = cls().name
                except Exception:
                    display_name = cls.__name__
                entries.append({
                    "template_id": tid,
                    "display_name": display_name,
                    "concept": meta.get("concept", "Other"),
                    "family": meta.get("family", ""),
                    "description": meta.get("description", ""),
                    "metrics": meta.get("metrics", {}),
                    "is_active": True,
                    "sort_order": max((e.get("sort_order", 0) for e in entries), default=-1) + 1,
                    "source": "preset",
                })
        # Drop ones whose template_id is no longer in registry
        entries = [e for e in entries if e["template_id"] in TEMPLATE_REGISTRY]
        return entries
    except Exception:
        entries = _seed()
        _save(entries)
        return entries


def _save(entries: list[dict]) -> None:
    POOL_PATH.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2))


def list_pool(active_only: bool = False, concept: str | None = None) -> list[dict]:
    with _LOCK:
        entries = _load()
    if active_only:
        entries = [e for e in entries if e.get("is_active", True)]
    if concept:
        entries = [e for e in entries if e.get("concept") == concept]
    entries.sort(key=lambda e: e.get("sort_order", 0))
    return entries


def list_concepts() -> list[str]:
    with _LOCK:
        entries = _load()
    return sorted({e.get("concept", "Other") for e in entries})


def update_entry(template_id: str, patch: dict[str, Any]) -> dict | None:
    with _LOCK:
        entries = _load()
        for e in entries:
            if e["template_id"] == template_id:
                for k in ("display_name", "concept", "description", "is_active"):
                    if k in patch:
                        e[k] = patch[k]
                _save(entries)
                return e
    return None


def delete_entry(template_id: str) -> bool:
    with _LOCK:
        entries = _load()
        new_entries = [e for e in entries if e["template_id"] != template_id]
        if len(new_entries) == len(entries):
            return False
        _save(new_entries)
        return True


def reorder(order: list[str]) -> list[dict]:
    """Apply a new ordering by template_id list. Unknown ids are appended."""
    with _LOCK:
        entries = _load()
        idx_map = {tid: i for i, tid in enumerate(order)}
        for e in entries:
            tid = e["template_id"]
            e["sort_order"] = idx_map.get(tid, len(order) + 1)
        entries.sort(key=lambda e: e.get("sort_order", 0))
        # Renumber to be sequential
        for i, e in enumerate(entries):
            e["sort_order"] = i
        _save(entries)
        return entries
