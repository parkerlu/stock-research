import { useState } from "react";
import { SearchBar } from "./SearchBar";
import { RecentList } from "./RecentList";
import { FavoriteList } from "./FavoriteList";
import { SnapshotCard } from "./SnapshotCard";

type Tab = "recent" | "favorites";

export function SearchPanel() {
  // 默认收藏 —— 自选是最常用的入口, 且带实时报价
  const [tab, setTab] = useState<Tab>("favorites");

  return (
    <div className="search-panel">
      <SearchBar />
      <div className="panel-tabs">
        <button
          className={tab === "recent" ? "active" : ""}
          onClick={() => setTab("recent")}
        >
          最近查看
        </button>
        <button
          className={tab === "favorites" ? "active" : ""}
          onClick={() => setTab("favorites")}
        >
          我的收藏 ⭐
        </button>
      </div>
      <div className="panel-list">
        {tab === "recent" ? <RecentList /> : <FavoriteList />}
      </div>
      <div className="panel-divider" />
      <SnapshotCard />
    </div>
  );
}
