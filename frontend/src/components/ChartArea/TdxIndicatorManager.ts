import type { KLineData } from "klinecharts";
import { registerIndicator } from "klinecharts";
import type { IndicatorResult } from "../../types/indicator";

// Key under which we register klinecharts indicators. Includes result version
// so re-creation with new data forces a re-registration.
const TDX_NAME_PREFIX = "TDX_";

interface TdxBarData {
  [key: string]: number;
}

const resultCache = new Map<string, IndicatorResult>();
const registeredNames = new Set<string>();

function klineIndicatorName(tdxName: string): string {
  return `${TDX_NAME_PREFIX}${tdxName}`;
}

/**
 * Register a klinecharts indicator for the given TDX IndicatorResult.
 * Safe to call multiple times with different data for the same name — the
 * internal lookup table is updated and the chart re-draws on next tick.
 */
function registerOnce(result: IndicatorResult): string {
  const kcName = klineIndicatorName(result.name);
  resultCache.set(result.name, result);

  if (registeredNames.has(kcName)) {
    return kcName;
  }

  registerIndicator<TdxBarData>({
    name: kcName,
    shortName: result.label,
    calcParams: [],
    figures: result.lines.map((l) => ({
      key: l.name,
      title: `${l.name}: `,
      type: "line",
      styles: () => ({ color: l.color, size: l.thickness }),
    })),
    calc: (dataList: KLineData[]) => {
      const current = resultCache.get(result.name);
      if (!current) return dataList.map(() => ({}));

      // Build timestamp -> index lookup from the cached result
      const tsIndex = new Map<number, number>();
      current.timestamps.forEach((ts, i) => tsIndex.set(ts, i));

      return dataList.map((bar) => {
        const idx = tsIndex.get(bar.timestamp);
        if (idx === undefined) return {};
        const out: TdxBarData = {};
        for (const line of current.lines) {
          const v = line.values[idx];
          if (v !== null && v !== undefined) out[line.name] = v;
        }
        return out;
      });
    },
    draw: ({ ctx, chart, bounding, yAxis, indicator }) => {
      const current = resultCache.get(result.name);
      if (!current) return false;

      // Horizontal reference lines
      for (const h of current.hlines) {
        const y = yAxis.convertToPixel(h.value);
        ctx.save();
        ctx.strokeStyle = h.color;
        ctx.lineWidth = 1;
        if (h.dashed) ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(bounding.left, y);
        ctx.lineTo(bounding.left + bounding.width, y);
        ctx.stroke();
        ctx.restore();
      }

      // Bands (STICKLINE): per-bar rectangles
      for (const band of current.bands) {
        if (band.timestamps.length === 0) continue;
        const y1 = yAxis.convertToPixel(band.y1);
        const y2 = yAxis.convertToPixel(band.y2);
        const top = Math.min(y1, y2);
        const height = Math.abs(y2 - y1);
        const barWidth = 4;

        ctx.save();
        ctx.fillStyle = band.color;
        ctx.globalAlpha = band.opacity;
        for (const ts of band.timestamps) {
          const point = chart.convertToPixel(
            { timestamp: ts },
            { paneId: indicator.paneId, absolute: false }
          );
          const coord = Array.isArray(point) ? point[0] : point;
          if (coord?.x === undefined) continue;
          ctx.fillRect(coord.x - barWidth / 2, top, barWidth, height);
        }
        ctx.restore();
      }

      return false;
    },
  });

  registeredNames.add(kcName);
  return kcName;
}

/**
 * Update cached data for an already-registered indicator.
 * The chart will pick up the new data on next redraw when we call
 * `chart.overrideIndicator` or re-create the indicator.
 */
export function setIndicatorData(result: IndicatorResult): string {
  return registerOnce(result);
}

export function getKlineIndicatorName(tdxName: string): string {
  return klineIndicatorName(tdxName);
}
