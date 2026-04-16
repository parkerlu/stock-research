import { PoolList } from "./PoolList";
import { PoolDetail } from "./PoolDetail";

export function PoolPanel() {
  return (
    <div className="pool-panel">
      <PoolList />
      <div className="panel-divider" />
      <PoolDetail />
    </div>
  );
}
