export interface StrategyItem {
  id: number;
  name: string;
  template: string;
  parameters: Record<string, number>;
  is_pinned: boolean;
  annualized_return: number | null;
  net_profit: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  final_capital: number | null;
  job_id: string | null;
  created_at: string | null;
}

export interface FactoryJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  total_candidates: number;
  evaluated: number;
  passed: number;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface BacktestReport {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  ts_code: string;
  strategy_id: number | null;
  metrics: BacktestMetrics | null;
  trades: TradeDetail[] | null;
  equity_curve: EquityPoint[] | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface BacktestMetrics {
  net_profit: number;
  net_profit_pct: number;
  annualized_return: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  final_capital: number;
}

export interface TradeDetail {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}
