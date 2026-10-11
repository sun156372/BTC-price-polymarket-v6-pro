# 云服务器更新指南

针对 2026-10-11 模拟盘截图中的三个问题。**按顺序执行，不要跳步。**

---

## 一、截图里的三个问题

### 问题 1：进程崩溃
```
[2026-10-11T05:34:22Z] btc-updown-5m-1791696600: KeyError: 'btc-updown-5m-1791696600'
```

**原因**：`recorder.mark_decision()` 用 `self.windows[slug]` 直接取字典，但那个窗口
**从未被 observe 过**。三种常见触发情况：

* 进程在窗口中途启动，决策时刻先于任何观察到达
* 市场查询或盘口抓取太慢，把整个观察阶段耗光了
* 某次网络抖动跳过了那个窗口的观察分支

**一个 KeyError 就杀掉整个会话。**

### 问题 2：决策提前 1.2 秒
```
hold btc-updown-5m-1791696900 ... -> imbalance:book_not_lopsided
```
决策在 **T-241.2** 触发（目标 T-240）。日志里 `open_lag=0.26s` 说明观察正常，
**是抓盘口花了 ~1.2 秒**，而旧代码只留了 0.6 秒余量。

**为什么严重**：这套策略的利润**全部**来自进场价。提前 1.2 秒就是买在更早、
可能更差的价位上。**回测是按 T-240 校准的，实际成交在 T-241.2 —— 这 1.2 秒
不是舍入误差。**

### 问题 3：`spot=Nonebps`（截图里就有）
```
spot=Nonebps
```
`spot_move` 为空时，**现货闸门被静默跳过**。那道闸门本来要挡掉 15% 的窗口，
跳过之后这些窗口会被当成"合规"放过去。

---

## 二、在服务器上更新

### 第 1 步：拉取代码

```bash
cd ~/polymarket          # 换成你的实际路径
git pull
```

如果服务器上不是 git 仓库，就整体覆盖 `v6_src/` 目录。

### 第 2 步：确认版本

```bash
cd v6_src
python tools_v6/test_v6.py
```

**必须看到 `RESULT: 86/86 checks passed`。**

只看到 `75/75` 说明**代码没更新成功** —— 新加的 11 项测试就是这次三个 bug 的
回归测试，没有它们就等于没打补丁。

### 第 3 步：静态检查（可选但快）

```bash
python tools_v6/static_check.py --path .
```
期望 `0 with undefined names`。

### 第 4 步：重启

```bash
# 先停掉正在跑的
pkill -f run_live_v6.py

# 重新启动
nohup python run_live_v6.py run --stake 10 > live.log 2>&1 &
tail -f live.log
```

---

## 三、确认修复生效

启动后**检查日志里有没有这三样**：

| 检查 | 期望 | 说明 |
|---|---|---|
| 决策时刻 | `secs_to_close` ≈ **240** | 不再是 241.2 |
| 现货 | `spot=` 是**数字** | 出现 `SPOT PAIR INCOMPLETE` 说明采样缺失 |
| 崩溃 | **没有 KeyError** | 未观察的窗口会被自动创建并正常拒绝 |

**决策时刻怎么核对**：日志时间戳减去窗口收盘时间。或者直接看
`paper_v6/decisions.csv` 的 `secs_to_close` 列，**中位数应该在 240 附近**。

```bash
python -c "
import csv, statistics
rows = list(csv.DictReader(open('paper_v6/decisions.csv')))
v = [float(r['secs_to_close']) for r in rows if r.get('secs_to_close')]
print(f'决策时刻中位数 T-{statistics.median(v):.1f}s  (目标 240)')
"
```

---

## 四、这次改动的内容

| 文件 | 改动 |
|---|---|
| `engine_v6/recorder.py` | `mark_decision` 改用 `setdefault`，窗口不存在就按需创建；从 slug 反解 epoch |
| `run_live_v6.py` | ① 传**原始** imbalance（不再预先定向）② 预留 1.5 秒抓盘口预算 ③ 现货缺失时**打印警告** |
| `engine_v6/strategy.py` | 增加 range 校验；记录一个未决问题（见下） |
| `tools_v6/test_v6.py` | 新增 11 项回归测试（75 → 86）|

### 关于 imbalance 的额外发现（重要）

修复过程中发现 **live 循环对 imbalance 做了两次定向**：

```
原始值 → 第一次定向（live 循环）→ 第二次定向（strategy 内部）= 符号反了
```

**它没有改变任何决策**，因为闸门用的是 `abs()`。**但它把错误的符号写进了账本** ——
而账本就是判断实盘是否正常的依据。现在改为传原始值，由 strategy 统一定向一次。

**同时发现一个更值得注意的事实**（已记录在 `strategy.py` 末尾，未擅自改动）：

* UP 交易、原始 imbalance > 0（UP 盘口买盘重）：胜率 **71.9%**（n=114）
* UP 交易、原始 imbalance < 0（UP 盘口卖盘重）：胜率 **79.2%**（n=1555）

**符号方向和 `full_depth_bid_notional` 的自然读法相反。** 我试过加符号检验，
结果**所有窗口全被拒**，于是回退了。

**为什么回退而不是翻转符号**：翻转符号能拟合这 1,669 个观测 —— 但这**正是这个项目
已经踩过一次的过拟合**。所以闸门保持原来的量级检验，**问题记录为未决**。

**含义**：`imbalance 是核心优势` 这个说法**比之前几轮声称的要弱**。
看起来**真正在起作用的是移动幅度闸门**。

---

## 五、更新后要观察什么

纸面跑满 **至少 3 天**，然后：

```bash
python run_live_v6.py report
```

| 指标 | 判断 |
|---|---|
| **延迟漂移中位数** | **< 2 分钱** → 可以继续；**≥ 3 分钱** → 停止 |
| `secs_to_close` 中位数 | 应在 240 附近 |
| 成交频率 | 回测 ~17 笔/天；实盘若远低，对照闸门漏斗找原因 |
| `SPOT PAIR INCOMPLETE` | 频繁出现说明现货采样有问题，需再查 |

---

## 六、还没解决的事（别忽略）

1. **`depth_check.py --live` 还没跑** —— 0.30 阈值是否有效仍未知
2. **`feed.py` 从没在真实网络下验证过** —— 字段名可能不合
3. **imbalance 符号含义未决**（第四节）
4. **2σ 区间仍包含 0** —— 22 天样本不足以证明期望为正

**在跑满一周、且延迟漂移确认 < 2 分钱之前，不要投真金。**
