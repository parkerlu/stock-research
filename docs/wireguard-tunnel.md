# 阿里云 ⇄ 家中 Mac mini 的 WireGuard 隧道

## 为什么

数据只留一份（用户选的方案 A）。生产站点在阿里云（2 核 / 8G、出网约 3.4 Mbps），
数据库在家里的 Mac mini（10 核 / 24G、上行约 50 Mbps）。隧道打通后阿里云直连
家里的 Postgres，不再维护第二份数据。

**代价说明白：家里断电 / 断网 / 换宽带 ⇒ 网站 502。** 这是方案 A 的固有取舍，
用户知情并接受。要消除这个单点，见文末"若要改成方案 C"。

## 拓扑

    用户 → Caddy(阿里云:443) → frontend(:8080) → backend(:8000)
                                                      ↓ WireGuard 10.8.0.1 → 10.8.0.2
                                              Mac mini 的 Postgres(:5432)

阿里云是服务端（有公网 IP），Mac mini 是客户端（在 NAT 后，主动拨出）。

## 配置位置

| 端 | 文件 | 地址 |
|---|---|---|
| 阿里云 | `/etc/wireguard/wg0.conf` | 10.8.0.1 |
| Mac mini | `/etc/wireguard/wg0.conf`（源在 `~/.wg/wg0.conf`）| 10.8.0.2 |

数据库密码在 Mac mini 的 `~/.wg/pgpass`（600 权限）。

## 阿里云安全组

必须放行 **UDP 51820** 入方向。服务器自身 ufw 是关的，拦截只可能在安全组。
排查口诀：SSH(22/tcp) 通但 `tcpdump -ni any udp port 51820` 抓不到包 = 安全组没开。

## 运维

```bash
# 看握手（latest handshake 在 2 分钟内即正常）
ssh root@101.201.39.240 'wg show'

# 家里重连
sudo wg-quick down wg0 && sudo wg-quick up wg0

# 阿里云端
systemctl status wg-quick@wg0
```

服务端已 `systemctl enable`，开机自启。客户端配了 `PersistentKeepalive = 25`
应对家用路由器回收 NAT 映射。**Mac mini 重启后需手动 `wg-quick up wg0`**
（macOS 上没配开机自启，要配可用 launchd）。

## 踩过的坑

**1. N+1 查询被隧道放大。** `/api/paper/status` 原来逐个持仓查当日价，局域网内
无感；搬到隧道后每次往返 30ms，几百个历史持仓就是十几秒。改成 `DISTINCT ON`
一次批量取，1.85s → 0.24s。**跨网络的数据库，任何 N+1 都会被放大 30 倍。**

**2. 弱密码。** 配置时发现家里 Postgres 绑在 `0.0.0.0:5432`、密码 `__REDACTED_DB_PASSWORD__`
——家里任何设备连上 WiFi 都能直连。已换 28 位随机密码，绑定收窄成
`127.0.0.1` + `10.8.0.2`。

## 回滚

生产的 `docker-compose.prod.yml.bak-YYYYMMDD` 是切换前的版本，本地 db 容器
和数据都还在（`/opt/stock/data/postgres`），另有 `/root/stock_prod_*.dump`。
把 `DATABASE_URL` 改回 `@db:5432` 重建 backend 即可回到独立数据库。

## 若要改成方案 C（消除单点）

家里做主库、阿里云做流式复制备库：家里断线时备库自动降级为只读，站点仍可
访问；恢复后自动追上。库只有 2.4G，初始同步走隧道几分钟。
