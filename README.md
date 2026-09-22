# 时间序列异常检测与趋势扫描 API

基于 **Python + FastAPI + SQLite** 的时间序列突变（spike / 水平偏移）检测、趋势扫描、双序列时滞关联扫描与跨检测事件聚合服务。

接收带时间戳的数值序列，使用**稳健滑动窗口**（中位数 + MAD）逐点评分，识别突变点并把相邻异常合并为异常区间，返回每段的基线、分数以及触发点的自然语言解释；检测结果持久化到 SQLite，支持按 ID / series_id 查询。趋势扫描接口以滑窗 **Mann-Kendall 检验 + Theil-Sen 斜率**识别显著上升 / 下降趋势区间（无状态，不入库）。时滞关联扫描接口对 **driver / response 双序列**按候选时滞平移并做**一对一最近邻匹配**，以 **Pearson 相关系数**评估各时滞下的关联强度并选出 `best_lag`（无状态，不入库）。事件聚合接口读取 1~50 条**已保存检测**的异常段，按**时间重叠或间隔不超过 `gap_seconds`** 跨检测（也包括同一检测的相邻段）聚合为事件，返回事件起止、覆盖序列、段数、最大绝对 z 分数与稳定排序的成员定位（只读，不修改检测记录）。

## 特性

- **稳健基线**：每个点只使用其之前 `window_size` 个已接纳点（trailing window，无前视泄漏），以中位数作为基线中心、`1.4826 × MAD` 作为稳健标准差。
- **突变评分**：稳健 z 分数 `z = (x − median) / robust_sd`，`|z| ≥ threshold` 判为异常；MAD 为 0 时自动回退到样本标准差，基线真正常量且值跳变时按最强突变处理。
- **两种基线策略 `baseline_policy`**：
  - `rolling`（默认）：所有历史点一律滑入基线窗口，行为与早期版本完全一致；
  - `guarded`：只有判为正常的点才允许进入后续基线，孤立尖刺不会污染其后的中位数与离散度；当连续异常达到 `rebase_after` 个点（默认等于 `min_window`，且不得更小）时，用该异常段最近 `min_window` 个点重建基线并递增基线代次（generation）。触发重建的点仍保持异常，原异常区间也不会被拆分。
- **区间合并**：相邻触发点间隔不超过 `merge_gap + 1` 个点时合并为一个异常区间（能把一次持续数点的水平偏移归为一段；guarded 重建不参与区间拆分）。
- **采样断档感知 `gap_reset_factor`**（可选，默认关闭）：以排序后相邻点**正时间差的中位数**估算常规采样间隔，相邻间隔超过 `gap_reset_factor` 倍即判定为断档并切分采样分段。每个分段独立执行 `min_window` 预热，rolling/guarded 的基线、guarded 的连续异常计数与重建代次都在分段起点从头开始，异常区间**绝不跨越断档合并**。响应返回断档边界、估算间隔与逐点 / 逐区间的分段编号；断档后即使水位整体迁移，也不会拿断档前的旧基线误报。
- **趋势扫描**：以 `window_size` / `step_size` 滑窗，每窗执行含**重复值方差修正**与**连续性校正**的 Mann-Kendall 检验，并按**实际时间差**计算 Theil-Sen 斜率（值/秒，非等距采样同样适用）；仅当 `p_value ≤ alpha` 且 `|slope| ≥ min_slope` 时标记上升 / 下降趋势；方向相同、时间范围重叠或首尾相接的显著窗口自动合并为趋势区间，返回区间范围、成员窗口、最小 p 值与成员斜率中位数。
- **时滞关联扫描**：对 driver / response 双序列，按候选 `lag_seconds` 逐个把 driver 时间平移 `+lag`，再按时间顺序做**贪心一对一最近邻匹配**（容差 `match_tolerance_seconds` 内取最近点，等距取较早 response，已匹配点不复用）；仅对**配对数 ≥ min_pairs 且两侧方差非零**的候选计算 Pearson 相关系数，逐候选返回配对数、相关系数、可用状态与不可用原因；`best_lag` 按 |correlation| 最大 → 配对数更多 → |lag| 更小 → lag 更小依次确定，无可用候选时为 `null`。
- **跨检测事件聚合**：`POST /api/v1/incidents/aggregate` 接收 1~50 个已保存的 `detection_id` 与非负 `gap_seconds`，把各检测的异常段统一按时间排序后顺序扫描——下一段开始时间不晚于当前事件结束时间 `+ gap_seconds`（即**时间重叠、首尾相接或间隔不超过 gap**）即并入同一事件，因此**同一检测的相邻异常段**同样会被合并，邻近关系可链式传递。每个事件返回起止时间、覆盖的 `series_id` / `detection_id`、成员段数、成员中的最大绝对 z 分数，以及按 `(start_time, detection_id, segment_index)` 稳定排序、可回溯到来源段的成员定位。接口只读，不写入或修改任何检测记录。
- **可解释结果**：每段返回基线窗口、最大/平均 |z|、全部触发点及中文原因说明。
- **持久化检索**：SQLite（WAL 模式）保存完整结果，按检测 ID 查询，可按业务 `series_id` 过滤、分页列出。
- **接口文档**：启动后访问 Swagger UI `/docs`、ReDoc `/redoc`。

## 目录结构

```
.
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI 路由与入参校验
│   ├── detection.py        # 滑动窗口稳健异常检测核心算法
│   ├── trends.py           # 滑窗 Mann-Kendall + Theil-Sen 趋势扫描
│   ├── lag_correlation.py  # 双序列时滞关联扫描（平移 + 一对一匹配 + Pearson）
│   ├── incidents.py        # 已保存检测异常段的跨检测事件聚合（只读）
│   ├── models.py           # Pydantic 请求/响应模型、时间戳解析
│   └── database.py         # SQLite 存储层
├── examples/
│   ├── request.json                # rolling 示例请求（含一次上升突变与一次下降突变）
│   ├── request_guarded.json        # guarded 示例请求（孤立尖刺 + 水平偏移重建基线）
│   ├── request_gap.json            # 断档感知示例请求（两段采样，水位迁移 + 各自尖刺）
│   ├── request_trends.json         # 趋势扫描示例请求（先升后降，方向切换）
│   └── request_lag_correlation.json # 时滞关联示例请求（response 滞后 driver 600 秒）
├── scripts/
│   └── smoke_test.py  # 算法 + API 端到端冒烟测试
├── data/              # SQLite 数据库目录（compose 挂载命名卷）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
docker compose up --build
```

服务启动在 <http://localhost:8000>，数据库保存在命名卷 `ts_data` 中。

验证：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" \
  -d @examples/request.json
curl -X POST http://localhost:8000/api/v1/trends \
  -H "Content-Type: application/json" \
  -d @examples/request_trends.json
curl -X POST http://localhost:8000/api/v1/lag-correlation \
  -H "Content-Type: application/json" \
  -d @examples/request_lag_correlation.json
```

### 方式二：本地 Python 运行

需要 Python 3.11+：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

可选环境变量：`DB_PATH`（SQLite 文件路径，默认 `./data/detections.db`）。

## 接口说明

### `POST /api/v1/detections` — 提交检测

请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `points` | `[{timestamp, value}]` | 必填（≥2） | 时间戳支持 ISO-8601 字符串（`...Z` 或带时区，无时区按 UTC）或 epoch 秒/毫秒数字；无需预先排序，服务端自动排序并拒绝重复时间戳 |
| `series_id` | string | `null` | 业务序列标识，便于过滤检索 |
| `window_size` | int | 20 | 滑动基线窗口大小（2~10000） |
| `min_window` | int | 5 | 开始评估所需的最小基线点数（前 `min_window` 个点不评分） |
| `threshold` | float | 3.0 | 稳健 z 分数阈值，`|z|` 超过即异常 |
| `merge_gap` | int | 2 | 合并区间时允许桥接的连续正常点数 |
| `baseline_policy` | string | `rolling` | 基线策略：`rolling`=所有历史点滑入基线（默认）；`guarded`=仅正常点进入基线，连续异常达到 `rebase_after` 时重建基线 |
| `rebase_after` | int \| null | `null` | guarded 模式下触发基线重建的连续异常点数，省略时取 `min_window`；必须不小于 `min_window`。rolling 模式忽略该字段 |
| `gap_reset_factor` | float \| null | `null` | 采样断档感知倍数（必须 **>1**）。设置后以相邻点正时间差的中位数估算常规采样间隔，间隔超过该倍数即切分新分段，每段独立预热并独立维护基线 / 连续异常计数 / 重建代次，异常区间不跨断档合并。启用至少需要 **3 个点**（少于 3 个点返回 422 校验错误）；省略或为 `null` 时功能关闭，结果与旧版完全兼容 |
| `include_points` | bool | false | 响应是否带逐点注解（完整结果始终入库） |

约束：`min_window ≤ window_size`；点数必须大于 `min_window`；guarded 模式下 `rebase_after ≥ min_window`；启用 `gap_reset_factor` 时点数必须 ≥ 3。

### `GET /api/v1/detections/{id}` — 按 ID 查询

查询参数 `include_points`（默认 `true`）控制是否返回逐点注解。

### `GET /api/v1/detections?series_id=&limit=&offset=` — 列表

按创建时间倒序返回检测摘要，可按 `series_id` 过滤，支持分页。

### `POST /api/v1/trends` — 趋势扫描

对序列按 `window_size` / `step_size` 滑动窗口，每窗执行 Mann-Kendall 检验（含重复值方差修正与连续性校正）并按实际时间差计算 Theil-Sen 斜率；仅当 `p_value ≤ alpha` 且 `|slope| ≥ min_slope` 时把窗口标记为显著上升 / 下降趋势，随后把方向相同、时间范围重叠或首尾相接的显著窗口合并为趋势区间。**无状态接口，结果不入库。**

请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `points` | `[{timestamp, value}]` | 必填 | 时间戳格式与检测接口一致；无需预先排序，服务端自动排序并拒绝重复时间戳（422） |
| `series_id` | string | `null` | 业务序列标识，随响应回显 |
| `window_size` | int | 10 | 滑动窗口大小（点数，3~10000） |
| `step_size` | int | 5 | 滑动步长（点数，1~10000）：小于 `window_size` 时窗口重叠，等于时首尾相接，大于时窗口间留空档（空档两侧的显著窗口不会合并） |
| `alpha` | float | 0.05 | Mann-Kendall 显著性水平（0, 1 开区间），`p_value ≤ alpha` 才可能判为趋势 |
| `min_slope` | float | 0.0 | 最小绝对斜率（**值/秒**），`|slope|` 达到该值才判为趋势；默认 0 不过滤（斜率恰为 0 的窗口恒判平稳） |
| `include_windows` | bool | false | 响应是否包含全部评估窗口（趋势区间的成员窗口始终返回） |

约束：`window_size ≥ 3`；点数必须不少于 `window_size`；`0 < alpha < 1`；`min_slope ≥ 0`；时间戳不得重复。违反任一约束返回 422。

响应（节选）：

- 摘要：`window_count`（评估窗口总数）、`significant_window_count`、`trend_segment_count` 及参数回显；
- `segments[]`：合并后的趋势区间，含 `direction`（上升 / 下降）、`start/end_index`、`start/end_time`、`window_count`、`min_p_value`（成员窗口最小 p 值）、`median_slope`（成员窗口斜率中位数，值/秒）与成员窗口明细 `windows[]`；
- 每个窗口对象含 `s`（Mann-Kendall S 统计量）、`var_s`（重复值修正后的方差）、`z`（连续性校正后 Z 值）、`p_value`（双侧）、`slope`（值/秒）、`significant` 与 `direction`（上升 / 下降 / 平稳）；
- `windows`：全部评估窗口（仅 `include_windows=true` 时返回，否则为 `null`）。

### `POST /api/v1/lag-correlation` — 双序列时滞关联扫描

对 driver / response 两条序列评估一组候选时滞：每个候选 `lag` 先把 driver 时间平移 `+lag`（lag > 0 表示 driver 的变化在 lag 秒后才体现到 response 上），再按时间顺序做**贪心一对一最近邻匹配**——每个平移后的 driver 点在**未匹配**的 response 点中取时间距离最近且不超过 `match_tolerance_seconds` 者配对，容差内左右等距时取**较早**的 response，已匹配的 response 点不得复用。仅对**配对数 ≥ min_pairs 且两侧配对值方差都非零**的候选计算 Pearson 相关系数。**无状态接口，结果不入库。**

请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `driver_points` | `[{timestamp, value}]` | 必填（≥1） | 驱动侧序列；时间戳格式与检测接口一致，无需预先排序，服务端自动排序并拒绝重复时间戳（422） |
| `response_points` | `[{timestamp, value}]` | 必填（≥1） | 响应侧序列；要求同 `driver_points` |
| `lag_seconds` | `[float]` | 必填（≥1 个） | 候选时滞（秒，可为负，必须有限）；逐个评估并按请求顺序返回结果 |
| `match_tolerance_seconds` | float | 必填 | 匹配容差（秒，≥0 且有限）：平移后两侧时间差不超过该值才允许配对 |
| `min_pairs` | int | 3 | 计算 Pearson 相关所需的最少配对数（≥2）；配对数不足时该候选不可用 |

约束：`lag_seconds` 至少 1 个且元素为有限数值；`match_tolerance_seconds ≥ 0` 且有限；`min_pairs ≥ 2`；两条序列各自时间戳不得重复。违反任一约束返回 422。

响应（节选）：

- 摘要：`driver_point_count`、`response_point_count`、`match_tolerance_seconds`、`min_pairs` 回显，以及 `candidate_count`（候选总数）与 `usable_candidate_count`（可用候选数）；
- `candidates[]`：按请求顺序逐候选返回 `lag_seconds`、`pair_count`（一对一匹配得到的配对数）、`correlation`（Pearson 相关系数，不可用时为 `null`）、`usable`（可用状态）与 `reason`（不可用原因，可用时为 `null`）；
- `best_lag`：可用候选中按 **|correlation| 最大 → 配对数更多 → |lag| 更小 → lag 更小**依次确定的最优时滞（秒）；无可用候选时为 `null`。

### `POST /api/v1/incidents/aggregate` — 跨检测事件聚合

读取 1~50 条**已保存检测**的异常段，按时间邻近关系聚合为事件。**只读接口：只查询 SQLite，不写入或修改任何检测记录；无状态，聚合结果不入库。**

聚合规则：取请求所列检测的全部异常段，统一按 `(start_time, detection_id, segment_index)` 排序后顺序扫描；只要下一段的开始时间 **不晚于当前事件结束时间 + `gap_seconds`**，即把该段并入当前事件。因此：

- 时间**重叠**（含跨检测重叠）、**首尾相接**（`gap_seconds=0` 时下一段起点等于上一段终点）或两段**间隔 ≤ `gap_seconds`** 都会合并；间隔恰好等于 `gap_seconds` 也算合并；
- **同一检测的相邻异常段**与跨检测段一视同仁，同样可以合并；
- 邻近关系可**链式传递**：A 与 B 邻近、B 与 C 邻近，即使 A 与 C 不直接邻近也归入同一事件。

请求体：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `detection_ids` | `[string]` | 必填（1~50 个） | 参与聚合的检测 ID，**不可重复**；任一 ID 不存在返回 **404**（错误详情列出未知 ID） |
| `gap_seconds` | float | 必填 | 允许桥接的最大时间间隔（秒），必须 **≥ 0 且有限**；负数 / NaN / Infinity 返回 422 |

约束：列表为空、超过 50 个、包含重复 ID 返回 422；`gap_seconds < 0` 或非有限返回 422；任一 `detection_id` 未知返回 404。检测本身**没有异常段**（`anomaly_segment_count=0`）不算错误，只是不贡献成员；全部检测均无异常时 `incident_count=0`。

响应：

- 摘要：`gap_seconds`（回显）、`detection_count`（参与检测数）、`segment_count`（参与聚合的异常段总数）、`incident_count`（事件数）；
- `incidents[]`：按事件**起始时间稳定排序**，每个事件含
  - `start_time` / `end_time`：事件起止（最早成员段开始 / 最晚成员段结束）；
  - `series_ids`：覆盖的业务 `series_id`（去重、字典序；未设置 `series_id` 的检测不计入）；
  - `detection_ids`：覆盖的检测 ID（去重、字典序）；
  - `segment_count`：成员异常段总数；
  - `max_abs_zscore`：全部成员段中的最大绝对 z 分数；
  - `members[]`：成员段定位，按 `(start_time, detection_id, segment_index)` **稳定排序**，每个成员含 `detection_id`、`series_id`、`segment_index`（该段在来源检测 `segments[]` 中的序号，从 0 开始）、`start_index` / `end_index`（来源序列中的点索引）、`start_time` / `end_time`、`point_count` / `anomaly_point_count` 与该段 `max_abs_zscore`——凭 `(detection_id, segment_index)` 可直接回溯 `GET /api/v1/detections/{id}` 返回的对应异常段。

```json
{
  "gap_seconds": 300.0,
  "detection_count": 2,
  "segment_count": 3,
  "incident_count": 1,
  "incidents": [
    {
      "start_time": "2026-09-22T09:15:00+00:00",
      "end_time": "2026-09-22T09:30:00+00:00",
      "series_ids": ["incident-series-a", "incident-series-b"],
      "detection_ids": ["a1b2...", "c3d4..."],
      "segment_count": 3,
      "max_abs_zscore": 42.7,
      "members": [
        {"detection_id": "a1b2...", "series_id": "incident-series-a", "segment_index": 0,
         "start_index": 15, "end_index": 15,
         "start_time": "2026-09-22T09:15:00+00:00", "end_time": "2026-09-22T09:15:00+00:00",
         "point_count": 1, "anomaly_point_count": 1, "max_abs_zscore": 42.7}
      ]
    }
  ]
}
```

### `GET /health` — 健康检查

## 调用示例

### rolling 模式（默认）

```bash
# 1) 发起检测（用仓库自带示例，包含上升/下降两段突变）
curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" \
  -d @examples/request.json | python -m json.tool

# 2) 记下返回的 id，查询完整结果
curl -s http://localhost:8000/api/v1/detections/<ID> | python -m json.tool

# 3) 按业务 series_id 列出历史检测
curl -s "http://localhost:8000/api/v1/detections?series_id=sensor-room-a-demo" | python -m json.tool
```

### guarded 模式（尖刺隔离 + 长异常段重建基线）

```bash
# 孤立尖刺只标记、不进基线；连续异常达到 rebase_after（此处 5）时，
# 用异常段最近 min_window 个点重建基线并递增代次
curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" \
  -d @examples/request_guarded.json | python -m json.tool
```

### 采样断档感知（gap_reset_factor）

```bash
# 两段规则 5 分钟采样，中间有约 11 小时断档；第二段水位整体迁移到 20。
# 估算常规间隔 = 300s，09:10 -> 20:00 的间隔远超 3 倍而切分为两段：
# 每段独立预热，第二段不会拿 10 的旧基线误报；两个尖刺也不会跨断档合并。
curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" \
  -d @examples/request_gap.json | python -m json.tool
```

响应中关注：

- 摘要 `gap_reset_factor`（回显倍数）、`estimated_interval_seconds`（估算常规间隔，秒）、`sampling_segment_count`（采样分段数）；
- `gap_boundaries[]`：每个断档边界包含 `before_index/after_index`、`before_time/after_time`、`gap_seconds`（断档实际时长）、`interval_seconds`（常规间隔）与 `gap_ratio`（倍数）；
- 逐点注解与异常区间上的 `sampling_segment_index`（所属采样分段编号，从 0 开始）。

### 趋势扫描（Mann-Kendall + Theil-Sen）

```bash
# 40 个点、5 分钟间隔：先以 0.25/步 上升 20 点，再对称下降 20 点（方向切换）。
# window_size=10、step_size=5 共 7 个窗口；跨峰窗口不显著，
# 上升段 3 窗、下降段 3 窗各自合并为一个趋势区间。
curl -s -X POST http://localhost:8000/api/v1/trends \
  -H "Content-Type: application/json" \
  -d @examples/request_trends.json | python -m json.tool
```

响应（节选）：

```json
{
  "series_id": "sensor-room-a-trend-demo",
  "point_count": 40,
  "window_size": 10,
  "step_size": 5,
  "alpha": 0.05,
  "min_slope": 0.0005,
  "window_count": 7,
  "significant_window_count": 6,
  "trend_segment_count": 2,
  "segments": [
    {
      "direction": "上升",
      "start_index": 0,
      "end_index": 19,
      "start_time": "2026-09-20T08:00:00+00:00",
      "end_time": "2026-09-20T09:35:00+00:00",
      "point_count": 20,
      "window_count": 3,
      "min_p_value": 8.3e-05,
      "median_slope": 0.000833,
      "windows": ["...3 个成员窗口明细..."]
    },
    {
      "direction": "下降",
      "start_index": 20,
      "end_index": 39,
      "start_time": "2026-09-20T09:40:00+00:00",
      "end_time": "2026-09-20T11:15:00+00:00",
      "point_count": 20,
      "window_count": 3,
      "min_p_value": 8.3e-05,
      "median_slope": -0.000833,
      "windows": ["...3 个成员窗口明细..."]
    }
  ],
  "windows": null
}
```

注意斜率单位为**值/秒**：示例中 5 分钟上升 0.25，即 `0.25 / 300 ≈ 8.33e-4`，因此 `min_slope` 取 `0.0005` 可以通过、取 `0.001` 则整段不判趋势。非等距采样时斜率仍按实际时间差计算。

### 时滞关联扫描（driver → response）

```bash
# driver 20 个点（5 分钟间隔）；response 22 个点、整体滞后 driver 600 秒
# 且采样时刻偏移 1 分钟（验证容差匹配）。候选时滞 -300~900 秒、容差 120 秒：
# lag=600 时 20 个 driver 全部配对，Pearson 相关 ≈ 0.9996，成为 best_lag。
curl -s -X POST http://localhost:8000/api/v1/lag-correlation \
  -H "Content-Type: application/json" \
  -d @examples/request_lag_correlation.json | python -m json.tool
```

响应（节选）：

```json
{
  "driver_point_count": 20,
  "response_point_count": 22,
  "match_tolerance_seconds": 120.0,
  "min_pairs": 3,
  "candidate_count": 5,
  "usable_candidate_count": 5,
  "candidates": [
    {"lag_seconds": -300.0, "pair_count": 19, "correlation": -0.2919, "usable": true, "reason": null},
    {"lag_seconds": 0.0,    "pair_count": 20, "correlation": 0.2939,  "usable": true, "reason": null},
    {"lag_seconds": 300.0,  "pair_count": 20, "correlation": 0.7825,  "usable": true, "reason": null},
    {"lag_seconds": 600.0,  "pair_count": 20, "correlation": 0.9996,  "usable": true, "reason": null},
    {"lag_seconds": 900.0,  "pair_count": 19, "correlation": 0.7875,  "usable": true, "reason": null}
  ],
  "best_lag": 600.0
}
```

候选不可用时 `correlation` 为 `null` 且 `reason` 给出原因，例如配对数不足（`"配对数 2 少于 min_pairs=3，未计算相关"`）或某一侧配对值恒定（`"driver 侧配对值方差为零，Pearson 相关未定义"`）；全部候选不可用时 `best_lag` 为 `null`。

### 跨检测事件聚合（incidents/aggregate）

```bash
# 先提交两条检测（可来自不同序列），记下各自返回的 id：
ID_A=$(curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" -d @examples/request.json | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')
ID_B=$(curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" -d @examples/request_gap.json | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 时间重叠或间隔不超过 300 秒的异常段聚合为同一事件（同一检测的相邻段也合并）
curl -s -X POST http://localhost:8000/api/v1/incidents/aggregate \
  -H "Content-Type: application/json" \
  -d "{\"detection_ids\": [\"$ID_A\", \"$ID_B\"], \"gap_seconds\": 300}" | python -m json.tool
```

`gap_seconds=0` 时只合并时间重叠或首尾相接的段；未知 ID 返回 404，空列表 / 超过 50 个 ID / 重复 ID / 负间隔返回 422。接口只读，聚合前后 `GET /api/v1/detections/{id}` 的结果不变。

### 内联 JSON 示例

```bash
curl -s -X POST http://localhost:8000/api/v1/detections \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "cpu-usage-guarded",
    "baseline_policy": "guarded",
    "rebase_after": 5,
    "window_size": 20,
    "min_window": 5,
    "threshold": 3.0,
    "include_points": true,
    "points": [
      {"timestamp": "2026-09-20T10:00:00Z", "value": 10.1}
    ]
  }' | python -m json.tool
```

响应中关注：

- 摘要字段 `baseline_policy`、`rebase_after`、`baseline_rebuild_count`（基线重建次数）；
- 逐点注解 `baseline_generation`（评分该点时使用的基线代次，rolling 恒为 `0`）与 `baseline_reset`（该点是否触发重建；触发点仍为异常）。

Python 调用示例：

```python
import requests

resp = requests.post("http://localhost:8000/api/v1/detections", json={
    "series_id": "cpu-usage",
    "window_size": 30,
    "threshold": 3.5,
    "merge_gap": 3,
    "points": [
        {"timestamp": "2026-09-20T10:00:00Z", "value": 12.3},
        {"timestamp": "2026-09-20T10:01:00Z", "value": 12.5},
        # ...
    ],
})
result = resp.json()
for seg in result["segments"]:
    print(seg["start_time"], "->", seg["end_time"], seg["score"]["max_abs_zscore"])
    print(seg["explanation"])

# 后续按 ID 取回
requests.get(f"http://localhost:8000/api/v1/detections/{result['id']}")
```

## 响应结构（节选）

```json
{
  "id": "b601c077...",
  "series_id": "sensor-room-a-demo",
  "created_at": "2026-09-20T12:00:00.000000+00:00",
  "point_count": 46,
  "anomaly_segment_count": 2,
  "anomaly_point_count": 6,
  "baseline_policy": "rolling",
  "rebase_after": null,
  "baseline_rebuild_count": 0,
  "gap_reset_factor": null,
  "estimated_interval_seconds": null,
  "gap_boundaries": [],
  "sampling_segment_count": 1,
  "segments": [
    {
      "start_index": 24,
      "end_index": 26,
      "start_time": "2026-09-20T10:00:00+00:00",
      "end_time": "2026-09-20T10:10:00+00:00",
      "point_count": 3,
      "anomaly_point_count": 3,
      "sampling_segment_index": 0,
      "baseline": {
        "median": 10.05,
        "mad": 0.1,
        "robust_sd": 0.22239,
        "dispersion_method": "mad",
        "scale": 1.4826,
        "window_size": 20,
        "window_start_time": "...",
        "window_end_time": "..."
      },
      "score": {
        "max_abs_zscore": 27.65,
        "max_signed_zscore": 27.65,
        "max_zscore_time": "2026-09-20T10:05:00+00:00",
        "mean_abs_zscore": 23.38
      },
      "triggers": [
        {
          "index": 24,
          "timestamp": "2026-09-20T10:00:00+00:00",
          "value": 13.8,
          "zscore": 16.86,
          "baseline_median": 10.05,
          "deviation": 3.75,
          "direction": "上升",
          "reason": "稳健 z 分数 +16.86 的绝对值超过阈值 3：观测值 13.8 相对滑动基线中位数 10.05 上升偏离 3.75（1.4826·MAD 稳健标准差 0.2224）"
        }
      ],
      "explanation": "区间 10:00 ~ 10:10（3 个点，其中 3 个点触发）：..."
    }
  ],
  "points": null
}
```

`dispersion_method` 取值：

- `mad`：常规稳健尺度 `1.4826 × MAD`；
- `std_fallback`：窗口内 MAD 为 0（如小窗口落在离散 / 多峰信号上），回退为样本标准差；
- `constant`：MAD 与标准差都为 0（基线完全恒定），此时任何偏离基线的值按 `z=999` 的最强突变处理。

逐点注解（`points[]`）额外包含：

- `baseline_generation`：评分该点时使用的基线代次。rolling 恒为 `0`；guarded 初始为 `0`，每发生一次重建递增，注意触发重建的点本身仍按旧代次基线评分；启用断档感知时代次在每个采样分段内从 `0` 重新计数。
- `baseline_reset`：该点是否为重建触发点（连续异常达到 `rebase_after`）。重建后基线整体替换为该异常段最近 `min_window` 个点，触发点仍标记为异常。
- `sampling_segment_index`：该点所属采样分段编号（从 `0` 开始）；未启用 `gap_reset_factor` 时恒为 `0`。异常区间对象上也带同名字段。

启用 `gap_reset_factor` 时摘要额外返回：

- `estimated_interval_seconds`：排序后相邻点**正时间差的中位数**（常规采样间隔估计，秒）；
- `gap_boundaries[]`：每个断档边界包含 `before_index` / `after_index`（断档前后相邻点索引）、`before_time` / `after_time`、`gap_seconds`（断档两侧实际时间差）、`interval_seconds`（常规间隔）与 `gap_ratio`（断档相对常规间隔的倍数）；
- `sampling_segment_count`：采样分段总数。

## 算法说明

1. 对每个索引 `i ≥ min_window`，取基线窗口（严格历史窗口，不使用当前点与未来点）：
   - rolling：`values[max(0, i-window_size):i]`，连续 trailing window；
   - guarded：当前已接纳点中最近的至多 `window_size` 个——初始为前 `min_window` 个种子点，之后**只有判为正常的点**才会被接纳，异常点一律排除。
2. 计算窗口中位数 `med` 与 MAD；稳健尺度 `σ = 1.4826 × MAD`，MAD=0 时回退样本标准差。
3. `z_i = (x_i − med) / σ`，`|z_i| ≥ threshold` 标记为触发点；`σ = 0` 且 `x_i ≠ med` 时 `z = 999`。
4. guarded 模式维护「连续异常计数」：
   - 出现正常点即清零，且该点滑入基线（孤立尖刺因此被永久排除在后续窗口之外，直到随窗口老化）；
   - 连续异常达到 `rebase_after`（默认 `min_window`，必须 `≥ min_window`）时，用该异常段最近 `min_window` 个点整体替换基线、代次加一、连续计数清零；触发点仍是异常，且随后的区间合并照常进行，重建不会把原异常区间拆开。
5. 扫描触发点序列：若相邻两个触发点索引差 `≤ merge_gap + 1`，则把中间正常点桥接，合并成一个异常区间（与策略无关）。
6. 每段输出：首个触发点评分时实际使用的基线窗口、段内最大/平均 |z| 与峰值时刻、全部触发点（含各自窗口基线与原因）、整段中文解释。
7. **采样断档感知（可选）**：仅当请求带 `gap_reset_factor > 1` 时启用。先以排序后相邻点正时间差的中位数估算常规间隔 `Δ`，相邻间隔 `> gap_reset_factor · Δ` 即切分采样分段；随后步骤 1~5 在每个分段内独立进行——每段前 `min_window` 个点重新预热不评分，rolling/guarded 基线从空建立，guarded 的连续异常计数与重建代次归零，区间桥接扫描被严格限制在分段内（不跨断档）。未启用时整段序列视为唯一分段，行为与旧版完全一致。

与基于均值/标准差的方案相比，中位数 + MAD 对窗口内少量离群点不敏感，能在尖刺和水平偏移共存时给出稳定基线；guarded 策略进一步保证异常点（除触发重建的长异常段末尾 `min_window` 个点外）不参与基线，适合存在真实水位迁移、又不希望偶发尖刺抬高离散度的场景。

### 趋势扫描算法（`POST /api/v1/trends`）

1. 以 `window_size` 个点为窗、`step_size` 个点为步长在排序后的序列上滑动，共 `⌊(n − window_size) / step_size⌋ + 1` 个窗口。
2. 每个窗口执行 Mann-Kendall 单调趋势检验：
   - 统计量 `S = Σ sign(x_j − x_i)`（i < j）；
   - **重复值方差修正**：`Var(S) = [n(n−1)(2n+5) − Σ t(t−1)(2t+5)] / 18`，其中 `t` 为每个并列值组的重复次数（窗口内值全部相同时 `Var(S)=0`，直接判平稳，`p=1`）；
   - **连续性校正**：`S>0` 时 `Z=(S−1)/√Var(S)`，`S<0` 时 `Z=(S+1)/√Var(S)`，`S=0` 时 `Z=0`；
   - 双侧 p 值取正态近似 `p = erfc(|Z|/√2)`。
3. 同一窗口计算 Theil-Sen 斜率：所有点对斜率 `(x_j − x_i)/(t_j − t_i)` 的中位数，分母为**实际时间差**（秒），因此非等距采样下斜率仍然无偏，单位为值/秒。
4. 仅当 `p_value ≤ alpha` 且 `|slope| ≥ min_slope`（且斜率非 0）时，窗口标记为显著趋势，方向取斜率符号（上升 / 下降）；否则判为平稳。
5. 按窗口顺序合并：方向相同且后一窗口起点不超过当前区间终点 +1（时间范围重叠或首尾相接）的显著窗口归为一段；空档（`step_size > window_size`）或方向不同的窗口自然断开。
6. 每个趋势区间输出：方向、起止索引 / 时间、覆盖点数、成员窗口数、成员窗口最小 p 值、成员窗口斜率中位数及成员窗口明细。

趋势扫描复杂度 `O(窗口数 × window_size²)`，建议 `window_size` 控制在数百以内；接口无状态，结果不写入 SQLite。

### 时滞关联扫描算法（`POST /api/v1/lag-correlation`）

1. 两条输入序列各自解析时间戳、按时间排序并拒绝重复时间戳（422）。
2. 对每个候选时滞 `lag`（秒，可为负），把 driver 时间轴整体平移：`t' = t_driver + lag`。`lag > 0` 表示 driver 的变化在 lag 秒后才体现到 response 上；`lag < 0` 表示 response 领先于 driver。
3. 在平移后的时间轴上按时间顺序做**贪心一对一最近邻匹配**：依次处理每个平移后的 driver 点，在**未匹配**的 response 点中取时间距离最近且不超过 `match_tolerance_seconds` 的点配对；容差内左右等距时取**较早**的 response；已匹配的 response 点立即锁定、不得复用，找不到容差内点的 driver 直接跳过。
4. 仅当候选的**配对数 ≥ min_pairs** 且**两侧配对值方差都非零**时，对配对序列 `(driver_value, response_value)` 计算 Pearson 相关系数 `r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)²·Σ(y−ȳ)²)`；否则该候选 `usable=false`、`correlation=null`，并在 `reason` 中说明（配对数不足 / 哪一侧方差为零）。
5. `best_lag` 在可用候选中依次按 **|correlation| 最大 → 配对数更多 → |lag| 更小 → lag 更小** 选取；无可用候选时为 `null`。

匹配为时间顺序单次扫描的贪心策略（不保证全局最优配对，但确定性强、可解释），整体复杂度约 `O(候选数 × (n_driver + n_response))` 量级；接口无状态，结果不写入 SQLite。

### 事件聚合算法（`POST /api/v1/incidents/aggregate`）

1. 按请求顺序逐个 `GET` 检测记录（只读 SQLite）；任一 ID 不存在立即返回 404，不做部分聚合。
2. 从每条检测的 `segments[]` 抽取成员段，保留其在来源检测中的 `segment_index` 与起止点索引，无异常段的检测贡献 0 个成员。
3. 全部成员按 `(start_time, detection_id, segment_index)` 稳定排序；ISO-8601 时间均为带 UTC 偏移的归一化格式，可直接做字典序比较。
4. 顺序扫描维护当前事件及其结束时间：成员 `start_time ≤ 事件结束时间 + gap_seconds` 即并入（重叠 / 相接 / 近邻），并在成员段更晚结束时顺延事件结束时间——因此邻近关系可链式传递，同一检测的相邻段也自然合并；否则关闭当前事件、开启新事件。
5. 每个事件汇总起止时间、去重排序的 `series_ids` / `detection_ids`、成员段数与成员最大绝对 z 分数；事件列表按起始时间输出（成员已按起点排序，首个成员起点即事件起点）。

整体复杂度约 `O(S log S)`（`S` 为参与聚合的异常段总数，`S ≤ 各检测段数之和`）；接口不写入数据库，聚合结果不持久化。

## 运行测试

```bash
pip install -r requirements.txt httpx
python scripts/smoke_test.py
```

测试覆盖：单点突变、持续水平偏移的识别与合并、恒定基线退化处理、无异常数据、rolling 默认行为不变、guarded 孤立尖刺隔离与连续异常重建（代次/不拆区间/自定义 `rebase_after`）、采样断档感知（间隔估计与边界、逐段预热与逐点分段编号、断档后基线从头避免水位迁移误报、区间不跨断档合并、guarded 状态逐段重置、默认关闭兼容）、趋势扫描（Mann-Kendall 重复值方差修正与连续性校正的量化校验、全同值窗口判平稳、非等距采样下 Theil-Sen 按实际时间差还原斜率、先升后降的方向切换拆分、重叠/首尾相接合并与空档不合并、`min_slope` 过滤）、时滞关联扫描（一对一最近邻匹配的容差边界 / 等距取较早 / 已匹配点不复用、正/负时滞还原、配对数不足与单侧零方差候选标记、`best_lag` 依次按 |corr| → 配对数 → |lag| → lag 决胜、无可用候选时为 null）、事件聚合（时间重叠 / 首尾相接 / 间隔等于与略大于 `gap_seconds` 的边界、同一检测相邻段合并、链式传递合并、成员按 `(start_time, detection_id, segment_index)` 稳定排序与段序号回溯、`series_id` 覆盖、无异常检测贡献 0 段与全空结果）、接口提交/查询/列表/404/参数校验（含 `rebase_after < min_window`、非法策略、启用断档但点数 < 3 与 `gap_reset_factor ≤ 1`、趋势接口的点数不足 / `window_size < 3` / `step_size < 1` / `alpha` 越界 / `min_slope` 为负 / 重复时间戳，时滞关联接口的空候选 / 非有限时滞或容差 / 负容差 / `min_pairs < 2` / 空序列 / 单侧重复时间戳，事件聚合接口的空 ID 列表 / 超过 50 个 ID / 重复 ID / 负间隔 / 非有限间隔 / 未知 ID 返回 404 / 聚合后检测记录不变的只读保证）。

## 备注

- SQLite 文件随命名卷持久化；删除卷即清空数据：`docker compose down -v`。
- 算法时间复杂度 `O(n · window_size)`，适合单次万级点数以内的离线/准实时检测。
