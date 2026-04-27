import { registerOverlay } from "klinecharts";

/**
 * Build the text label for a trade marker.
 *   Buy:  "买 ¥123.45"
 *   Sell: "卖 ¥125.67 +5.20%"  (or "-2.30%")
 */
export function buildTradeLabel(
  type: "buy" | "sell",
  price: number,
  pnlPct?: number
): string {
  if (type === "buy") {
    return `买 ¥${price.toFixed(2)}`;
  }
  const pctStr =
    pnlPct !== undefined
      ? ` ${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(2)}%`
      : "";
  return `卖 ¥${price.toFixed(2)}${pctStr}`;
}

let registered = false;

/**
 * Register a custom "tradeMarker" overlay once globally. Renders a colored
 * rectangle (red for buy, green for sell) with white text, plus a triangle
 * pointer to the K-line.
 *
 * Use via:
 *   chart.createOverlay({
 *     name: "tradeMarker",
 *     points: [{ timestamp, value: price }],
 *     extendData: { type: "buy" | "sell", text: "..." },
 *   });
 */
export function registerTradeMarker(): void {
  if (registered) return;
  registered = true;

  registerOverlay({
    name: "tradeMarker",
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ overlay, coordinates, bounding }) => {
      const data = overlay.extendData as
        | { type: "buy" | "sell"; text: string }
        | undefined;
      if (!data) return [];
      const isBuy = data.type === "buy";
      // Buy:  ▲ + price at BOTTOM edge, dashed vertical line up to bar low
      // Sell: ▼ + price at TOP    edge, dashed vertical line down to bar high
      const color = isBuy ? "#e94560" : "#4caf50";
      const fontSize = 10;
      const tri = 5;
      const cx = coordinates[0].x;
      const barY = coordinates[0].y;     // bar low (buy) or bar high (sell) in px
      const paneH = bounding?.height ?? 400;
      const edgeMargin = 12;

      let triTipY: number, triBaseY: number, textY: number;
      let lineStartY: number, lineEndY: number;
      if (isBuy) {
        triBaseY = paneH - edgeMargin;
        triTipY = triBaseY - tri * 1.6;
        textY = triTipY - 3;
        lineStartY = barY + 3;            // just below bar low
        lineEndY = triTipY - 1;           // just above triangle tip
      } else {
        triBaseY = edgeMargin;
        triTipY = triBaseY + tri * 1.6;
        textY = triTipY + 3;
        lineStartY = triTipY + 1;
        lineEndY = barY - 3;              // just above bar high
      }

      return [
        // Dashed vertical line connecting marker to actual bar
        {
          type: "line",
          attrs: {
            coordinates: [
              { x: cx, y: lineStartY },
              { x: cx, y: lineEndY },
            ],
          },
          styles: {
            style: "dashed",
            color,
            size: 1,
            dashedValue: [3, 3],
          },
          ignoreEvent: true,
        },
        {
          type: "polygon",
          attrs: {
            coordinates: [
              { x: cx, y: triTipY },
              { x: cx - tri, y: triBaseY },
              { x: cx + tri, y: triBaseY },
            ],
          },
          styles: {
            style: "fill",
            color,
            borderSize: 0,
            borderColor: "transparent",
          },
          ignoreEvent: true,
        },
        {
          type: "text",
          attrs: {
            x: cx,
            y: textY,
            text: data.text,
            align: "center",
            baseline: isBuy ? "bottom" : "top",
          },
          styles: {
            // Override all the klinecharts defaults that produce blue bg + white text
            style: "fill",
            color,
            size: fontSize,
            family: "Helvetica Neue",
            weight: "600",
            backgroundColor: "transparent",
            borderColor: "transparent",
            borderSize: 0,
            borderRadius: 0,
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 0,
            paddingBottom: 0,
          },
          ignoreEvent: true,
        },
      ];
    },
    styles: {
      // Hide the default control-point indicator dots that klinecharts adds
      point: {
        color: "transparent",
        borderColor: "transparent",
        borderSize: 0,
        radius: 0,
        activeColor: "transparent",
        activeBorderColor: "transparent",
        activeBorderSize: 0,
        activeRadius: 0,
      },
    },
  });
}
