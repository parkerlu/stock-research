// 批量实时报价 hook — 一次请求拿一组股票, 定时刷新。
// 用后端的 /api/quotes/snapshots (腾讯批量接口), N 只股票只发 1 个请求。
import { useEffect, useRef, useState } from "react";
import { getSnapshots } from "../api/quotes";
import type { Snapshot } from "../types/quote";

/**
 * @param codes    ts_code 列表; 为空时不发请求
 * @param intervalMs 轮询间隔, 默认 15s (列表页不需要秒级)
 */
export function useLiveQuotes(
  codes: string[],
  intervalMs = 15000
): Record<string, Snapshot> {
  const [quotes, setQuotes] = useState<Record<string, Snapshot>>({});
  // 用字符串 key 做依赖, 避免每次渲染新数组引用导致的重复拉取
  const key = codes.join(",");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!key) {
      setQuotes({});
      return;
    }
    let alive = true;
    const list = key.split(",");

    const run = async () => {
      try {
        const data = await getSnapshots(list);
        if (alive) setQuotes(data);
      } catch {
        /* 静默失败: 列表仍显示名称, 只是没有报价 */
      }
      if (alive) timer.current = setTimeout(run, intervalMs);
    };
    run();

    return () => {
      alive = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [key, intervalMs]);

  return quotes;
}
