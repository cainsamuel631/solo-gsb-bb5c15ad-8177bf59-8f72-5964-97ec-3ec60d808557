"""端到端冒烟测试：算法正确性 + FastAPI 接口 + SQLite 持久化。

运行（需安装 requirements）：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.detection import analyze_series  # noqa: E402
from app.incidents import aggregate_incidents  # noqa: E402
from app.lag_correlation import (  # noqa: E402
    _match_pairs,
    _select_best_lag,
    scan_lag_correlation,
)
from app.main import app  # noqa: E402
from app.trends import _mann_kendall, scan_trends  # noqa: E402


def test_algorithm_spikes_and_shift() -> None:
    n = 40
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    # 单点突增
    values[25] = 20.0
    # 水平偏移：连续 5 个点明显抬升
    values[30:35] = [16.0, 16.1, 15.9, 16.2, 15.8]
    ts = [f"2026-09-20T08:{i:02d}:00Z" for i in range(n)]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2)

    assert result["anomaly_segment_count"] == 2, result["segments"]
    seg1, seg2 = result["segments"]

    # 单点突增区间
    assert seg1["start_index"] == seg1["end_index"] == 25
    assert seg1["score"]["max_abs_zscore"] >= 3.0
    assert seg1["triggers"][0]["direction"] == "上升"

    # 水平偏移区间，5 个点应被合并为一个区间
    assert seg2["start_index"] == 30 and seg2["end_index"] == 34
    assert seg2["point_count"] == 5
    assert "基线" in seg2["explanation"]

    # 逐点注解
    annotated = result["points"]
    assert len(annotated) == n - 5
    assert all("zscore" in p and "baseline_median" in p for p in annotated)
    print("✓ 算法测试通过：单点突变 + 水平偏移识别与合并正确")


def test_constant_baseline() -> None:
    values = [5.0] * 15 + [8.0, 5.0]
    ts = [f"2026-09-20T09:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=10, min_window=5, threshold=3.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 1
    trigger = result["segments"][0]["triggers"][0]
    assert trigger["zscore"] == 999.0
    assert "基线恒定" in trigger["reason"]
    print("✓ 恒定基线（MAD=0）退化为强突变哨兵处理正确")


def test_no_anomaly() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.07, 0.93]
    ts = [f"2026-09-20T10:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=5, min_window=3, threshold=5.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 0
    assert result["anomaly_point_count"] == 0
    print("✓ 无异常数据正确返回空区间")


def test_rolling_defaults_unchanged() -> None:
    """rolling（默认）结果保持原有行为，新增字段取默认值。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0
    ts = [f"2026-09-20T11:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0)
    assert result["baseline_policy"] == "rolling"
    assert result["rebase_after"] is None
    assert result["baseline_rebuild_count"] == 0
    assert all(
        p["baseline_generation"] == 0 and p["baseline_reset"] is False
        for p in result["points"]
    )
    print("✓ rolling 默认模式：结果与新增默认字段正确")


def test_guarded_isolated_spike_excluded() -> None:
    """guarded 模式：孤立尖刺不得进入后续基线，不触发重建。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0  # 孤立尖刺
    ts = [f"2026-09-20T12:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded",
    )
    assert result["baseline_policy"] == "guarded"
    assert result["rebase_after"] == 5  # 默认取 min_window
    assert result["baseline_rebuild_count"] == 0
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]] == [(10, 10)]

    by_index = {p["index"]: p for p in result["points"]}
    # 尖刺被排除：其后正常点的基线中位数仍贴近 10，离散度未被 20 放大
    assert abs(by_index[11]["baseline_median"] - 10.0) < 0.2
    assert by_index[11]["is_anomaly"] is False
    assert by_index[10]["baseline_reset"] is False
    assert all(p["baseline_generation"] == 0 for p in result["points"])

    # 区间基线窗口不得包含被排除的尖刺：成员来自 0..9（5 个种子 + 5..9 正常点），
    # 时间上界停在尖刺前一个点
    seg = result["segments"][0]
    assert seg["baseline"]["window_size"] == 10
    assert seg["baseline"]["window_end_time"] == "2026-09-20T12:09:00Z"
    print("✓ guarded 孤立尖刺：不污染后续中位数/离散度，且不触发重建")


def test_guarded_rebuild_on_consecutive_anomalies() -> None:
    """guarded 模式：连续异常达到 rebase_after 时重建基线、递增代次但不拆区间。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(40)]
    values[20:30] = [16.0] * 10  # 持续水平偏移
    ts = [f"2026-09-20T13:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2,
        baseline_policy="guarded",
    )
    by_index = {p["index"]: p for p in result["points"]}

    # 第 5 个连续异常点 i=24 触发重建；该点仍保持异常、区间不拆分
    assert by_index[24]["baseline_reset"] is True
    assert by_index[24]["is_anomaly"] is True
    assert by_index[24]["baseline_generation"] == 0  # 评分用旧代次
    assert result["baseline_rebuild_count"] >= 1
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]][0] == (20, 24)

    # 重建后 i=25..29 对新基线（全 16）恢复正常，代次为 1
    for i in range(25, 30):
        assert by_index[i]["is_anomaly"] is False
        assert by_index[i]["baseline_generation"] == 1
        assert by_index[i]["baseline_reset"] is False

    # 自定义 rebase_after（>= min_window）：首次重建延后到 i=26
    custom = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded", rebase_after=7,
    )
    assert custom["rebase_after"] == 7
    resets = [p["index"] for p in custom["points"] if p["baseline_reset"]]
    assert resets[0] == 26
    print("✓ guarded 连续异常：达 rebase_after 重建基线并递增代次，触发点仍异常、区间不拆分")


def _gap_series() -> tuple[list[str], list[float]]:
    """两段规则 5 分钟采样（各 14 点），中间断开 10 小时；基线水位 10。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 8, 0, 0, tzinfo=timezone.utc)
    timestamps: list[str] = []
    values: list[float] = []
    t = base
    for _ in range(14):
        timestamps.append(t.isoformat())
        values.append(10.0)
        t += timedelta(minutes=5)
    t = base + timedelta(hours=10)  # 大断档（常规间隔的 120 倍）
    for _ in range(14):
        timestamps.append(t.isoformat())
        values.append(10.0)
        t += timedelta(minutes=5)
    return timestamps, values


def test_gap_aware_split_and_warmup() -> None:
    """启用 gap_reset_factor：断档切分、间隔估计、逐段预热与逐点分段编号。"""
    ts, values = _gap_series()
    values[13] = 20.0  # 第一段末点尖刺

    result = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0,
        merge_gap=2, gap_reset_factor=3.0,
    )
    assert result["gap_reset_factor"] == 3.0
    assert result["estimated_interval_seconds"] == 300.0
    assert result["sampling_segment_count"] == 2
    assert len(result["gap_boundaries"]) == 1
    gap = result["gap_boundaries"][0]
    assert gap["before_index"] == 13 and gap["after_index"] == 14
    # 段 0 末点 = base+65min，段 1 首点 = base+10h，差 535min
    assert gap["gap_seconds"] == 10 * 3600 - 65 * 60
    assert gap["gap_ratio"] == gap["gap_seconds"] / 300.0

    # 两段各自预热 5 个点：评估点为 5..13 与 19..27，共 18 个
    evaluated = [p["index"] for p in result["points"]]
    assert evaluated == list(range(5, 14)) + list(range(19, 28))
    assert result["evaluated_point_count"] == 18
    seg_of = {p["index"]: p["sampling_segment_index"] for p in result["points"]}
    assert all(seg_of[i] == 0 for i in range(5, 14))
    assert all(seg_of[i] == 1 for i in range(19, 28))

    # 第一段尖刺被标记，且区间带分段编号
    assert [(s["start_index"], s["sampling_segment_index"]) for s in result["segments"]] == [(13, 0)]
    print("✓ 断档感知：间隔估计、断档边界、逐段预热与逐点分段编号正确")


def test_gap_aware_baseline_restart_and_no_merge_across_gap() -> None:
    """断档后基线从头：新水位不误报；merge_gap 再大，异常区间也不跨断档合并。"""
    from datetime import datetime, timedelta, timezone

    # 场景一：断档后水位迁移到 20，独立预热后不应报异常
    ts, values = _gap_series()
    values[14:] = [20.0] * 14
    gapped = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0, gap_reset_factor=3.0
    )
    plain = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0
    )
    assert gapped["anomaly_segment_count"] == 0
    # 不启用时，断档后早期点仍拿 10 的旧基线评分，会误报一段
    assert plain["anomaly_segment_count"] >= 1

    # 场景二：断档两侧各有异常，merge_gap=10 也不得合并
    ts2, values2 = _gap_series()
    values2[13] = 30.0  # 段 0 末点
    values2[19] = 30.0  # 段 1 首个可评估点（14..18 预热，19 起评分）
    merged = analyze_series(
        ts2, values2, window_size=10, min_window=5, threshold=3.0,
        merge_gap=10, gap_reset_factor=3.0,
    )
    assert [(s["start_index"], s["end_index"]) for s in merged["segments"]] == [(13, 13), (19, 19)]
    assert [s["sampling_segment_index"] for s in merged["segments"]] == [0, 1]
    # 同参数不启用断档时会被桥接成一个跨断档区间
    merged_plain = analyze_series(
        ts2, values2, window_size=10, min_window=5, threshold=3.0, merge_gap=10
    )
    assert [(s["start_index"], s["end_index"]) for s in merged_plain["segments"]] == [(13, 19)]
    print("✓ 断档感知：断档后基线从头避免水位迁移误报，且异常区间不跨断档合并")


def test_gap_aware_guarded_resets_state() -> None:
    """guarded + 断档：第二段连续异常计数与重建代次从头开始。"""
    ts, values = _gap_series()
    # 第一段放一个长水平偏移触发一次重建；第二段同样偏移
    values[10:14] = [30.0] * 4
    values[23:27] = [30.0] * 4

    result = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        merge_gap=2, baseline_policy="guarded", gap_reset_factor=3.0,
    )
    by_index = {p["index"]: p for p in result["points"]}
    # 第一段的重建点 i=12（10,11,12 连续 3 异常，rebase_after 默认=min_window=3）
    assert by_index[12]["baseline_reset"] is True
    assert by_index[12]["baseline_generation"] == 0  # 触发点按旧代次评分
    # 第二段同样在 i=25 触发重建，且代次从 0 重新计数（未受第一段重建影响）
    assert by_index[25]["baseline_reset"] is True
    assert by_index[25]["baseline_generation"] == 0
    assert by_index[26]["baseline_generation"] == 1  # 重建后首个点用新代次
    # 两段各重建一次
    assert result["baseline_rebuild_count"] == 2
    print("✓ 断档感知 guarded：连续异常计数与重建代次在每段起点重置")


def test_gap_aware_defaults_compatible() -> None:
    """未启用 gap_reset_factor 时响应字段取兼容默认值。"""
    ts, values = _gap_series()
    result = analyze_series(
        ts, values, window_size=10, min_window=5, threshold=3.0, merge_gap=2
    )
    assert result["gap_reset_factor"] is None
    assert result["estimated_interval_seconds"] is None
    assert result["gap_boundaries"] == []
    assert result["sampling_segment_count"] == 1
    assert all(p["sampling_segment_index"] == 0 for p in result["points"])
    assert all(s["sampling_segment_index"] == 0 for s in result["segments"])
    print("✓ 断档感知默认关闭：新增字段全部取兼容默认值")


def test_mann_kendall_ties_quantitative() -> None:
    """重复值方差修正 + 连续性校正的量化校验；全同值窗口退化为无趋势。"""
    # 含并列组 [1,1,2,2,3]：S=8，tie_term=2·1·9×2=36，Var=(300−36)/18=14.6667
    s, var_s, z, p = _mann_kendall([1.0, 1.0, 2.0, 2.0, 3.0])
    assert s == 8
    assert abs(var_s - 264.0 / 18.0) < 1e-9
    assert abs(z - 7.0 / var_s**0.5) < 1e-9  # 连续性校正：(S−1)/√Var
    assert abs(z - 1.8278) < 1e-3
    assert abs(p - 0.0676) < 5e-4

    # 无重复值对照 [1,2,3,4,5]：S=10，Var=300/18=16.6667，修正后方差更小
    s2, var_s2, z2, p2 = _mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s2 == 10
    assert abs(var_s2 - 300.0 / 18.0) < 1e-9
    assert var_s < var_s2  # 重复值修正使方差减小
    assert abs(p2 - 0.0275) < 5e-4

    # 全部相同：Var(S)=0，Z=0，p=1，无趋势
    s3, var_s3, z3, p3 = _mann_kendall([7.0] * 6)
    assert s3 == 0 and var_s3 == 0.0 and z3 == 0.0 and p3 == 1.0

    # 全同值窗口经完整扫描判为平稳、不产生区间
    ts = [f"2026-09-20T07:{i:02d}:00Z" for i in range(6)]
    result = scan_trends(ts, [7.0] * 6, window_size=6, step_size=1)
    assert result["windows"][0]["direction"] == "平稳"
    assert result["windows"][0]["significant"] is False
    assert result["trend_segment_count"] == 0
    print("✓ Mann-Kendall 重复值方差修正与连续性校正量化正确，全同值窗口判平稳")


def test_theil_sen_irregular_sampling() -> None:
    """非等距采样：Theil-Sen 按实际时间差还原真实斜率（值/秒）。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 6, 0, 0, tzinfo=timezone.utc)
    offsets = [0, 7, 13, 42, 55, 90, 121, 200, 260, 301]  # 非等距秒偏移
    ts = [(base + timedelta(seconds=o)).isoformat() for o in offsets]
    values = [3.0 + 0.02 * o for o in offsets]  # 真实斜率 0.02 /秒

    result = scan_trends(ts, values, window_size=10, step_size=1, min_slope=0.001)
    assert result["window_count"] == 1
    w = result["windows"][0]
    assert abs(w["slope"] - 0.02) < 1e-12  # 按实际时间差，非按点序号
    assert w["significant"] is True and w["direction"] == "上升"
    assert result["segments"][0]["median_slope"] == w["slope"]

    # 同样的值若按等距点序号理解斜率会失真：换一组非等距时间戳斜率随之改变
    offsets2 = [0, 100, 101, 300, 301, 500, 600, 800, 900, 1000]
    ts2 = [(base + timedelta(seconds=o)).isoformat() for o in offsets2]
    values2 = [1.0 + 0.005 * o for o in offsets2]
    result2 = scan_trends(ts2, values2, window_size=10, step_size=1, min_slope=0.0001)
    assert abs(result2["windows"][0]["slope"] - 0.005) < 1e-12
    print("✓ 非等距采样：Theil-Sen 按实际时间差还原斜率（值/秒）")


def test_trend_direction_switch() -> None:
    """方向切换：先升后降的序列拆分为方向相反的两个趋势区间。"""
    n = 40
    values = [float(i) if i < 20 else float(38 - i) for i in range(n)]
    ts = [f"2026-09-20T05:{i // 60:02d}:{i % 60:02d}Z" for i in range(n)]  # 1 秒间隔

    result = scan_trends(ts, values, window_size=10, step_size=5, alpha=0.05, min_slope=0.5)
    assert result["window_count"] == 7
    assert result["trend_segment_count"] == 2

    up, down = result["segments"]
    assert up["direction"] == "上升" and (up["start_index"], up["end_index"]) == (0, 19)
    assert down["direction"] == "下降" and (down["start_index"], down["end_index"]) == (20, 39)
    assert up["window_count"] == 3 and down["window_count"] == 3
    assert abs(up["median_slope"] - 1.0) < 1e-9  # 1 秒间隔下斜率 ±1 /秒
    assert abs(down["median_slope"] + 1.0) < 1e-9
    assert up["min_p_value"] < 1e-3 and down["min_p_value"] < 1e-3
    # 区间成员窗口方向一致，且跨峰窗口 [15..24] 不显著、不属于任何区间
    assert all(w["direction"] == "上升" for w in up["windows"])
    assert all(w["direction"] == "下降" for w in down["windows"])
    middle = result["windows"][3]
    assert (middle["start_index"], middle["end_index"]) == (15, 24)
    assert middle["significant"] is False and middle["direction"] == "平稳"
    print("✓ 方向切换：上升/下降显著窗口各自合并，跨峰窗口判平稳")


def test_trend_merge_adjacent_and_gap() -> None:
    """合并规则：重叠或首尾相接的同向显著窗口合并；留空档的不合并。"""
    values = [float(i) for i in range(25)]
    ts = [f"2026-09-20T04:{i // 60:02d}:{i % 60:02d}Z" for i in range(25)]

    # step == window_size：窗口首尾相接，同向显著窗口合并为一段
    joined = scan_trends(ts, values[:20], window_size=10, step_size=10, min_slope=0.1)
    assert joined["window_count"] == 2
    assert joined["trend_segment_count"] == 1
    seg = joined["segments"][0]
    assert (seg["start_index"], seg["end_index"]) == (0, 19)
    assert seg["window_count"] == 2 and seg["point_count"] == 20

    # step > window_size：窗口间留空档，即使同向显著也不合并
    gapped = scan_trends(ts, values, window_size=10, step_size=15, min_slope=0.1)
    assert gapped["window_count"] == 2
    assert gapped["trend_segment_count"] == 2
    assert [(s["start_index"], s["end_index"]) for s in gapped["segments"]] == [(0, 9), (15, 24)]
    print("✓ 合并规则：重叠/首尾相接合并，空档不合并")


def test_trend_min_slope_filter() -> None:
    """min_slope 过滤：p 值达标但绝对斜率不足时不判趋势。"""
    values = [0.001 * i for i in range(20)]  # 斜率 0.001 /秒（1 秒间隔）
    ts = [f"2026-09-20T03:{i // 60:02d}:{i % 60:02d}Z" for i in range(20)]

    blocked = scan_trends(ts, values, window_size=10, step_size=5, min_slope=0.01)
    assert blocked["significant_window_count"] == 0
    assert blocked["trend_segment_count"] == 0
    assert all(w["p_value"] <= 0.05 for w in blocked["windows"])  # p 值其实达标

    allowed = scan_trends(ts, values, window_size=10, step_size=5, min_slope=0.0001)
    assert allowed["significant_window_count"] == allowed["window_count"]
    assert allowed["trend_segment_count"] == 1
    print("✓ min_slope 过滤：斜率不达标的显著 p 值窗口不判趋势")


def test_lag_matching_rules() -> None:
    """一对一最近邻匹配：容差边界、等距取较早、已匹配点不复用。"""
    # 容差边界：距离 == 容差可配对，超过则不可
    assert _match_pairs([0.0], [120.0], 120.0) == [(0, 0)]
    assert _match_pairs([0.0], [120.0], 119.9) == []

    # 容差内左右等距：取较早的 response
    assert _match_pairs([10.0], [5.0, 15.0], 5.0) == [(0, 0)]

    # 一对一：单个 response 不得被两个 driver 复用（先到的 driver 锁定）
    assert _match_pairs([0.0, 10.0], [5.0], 6.0) == [(0, 0)]

    # 贪心按时间顺序：每个 driver 取容差内最近的未用点
    assert _match_pairs([0.0, 4.0], [3.0, 5.0], 3.0) == [(0, 0), (1, 1)]

    # response 全部用尽且 driver 时间越过末尾：不得越界、不得复用
    assert _match_pairs([0.0, 100.0], [0.0], 1.0) == [(0, 0)]
    print("✓ 时滞匹配：容差边界、等距取较早、一对一不复用、越界防护正确")


def _lag_demo_series() -> tuple[list[str], list[float], list[str], list[float]]:
    """构造 response 滞后 driver 600 秒的双序列（response 采样整体偏移 60 秒）。"""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 21, 8, 0, 0, tzinfo=timezone.utc)
    dvals = [10.0, 11.0, 12.5, 13.5, 12.5, 11.0, 10.0, 9.0, 8.0, 9.0,
             10.5, 12.0, 13.5, 14.5, 13.5, 12.0, 10.5, 9.5, 9.0, 10.0]
    driver_ts = [(base + timedelta(minutes=5 * i)).isoformat() for i in range(len(dvals))]
    rvals = [10.05, 10.95] + [
        dvals[j - 2] + (0.05 if j % 2 == 0 else -0.05) for j in range(2, 22)
    ]
    response_ts = [
        (base + timedelta(minutes=1) + timedelta(minutes=5 * j)).isoformat()
        for j in range(len(rvals))
    ]
    return driver_ts, dvals, response_ts, rvals


def test_lag_correlation_scan_basic() -> None:
    """候选时滞扫描：还原真实时滞（含负时滞），配对数与相关系数符合构造。"""
    driver_ts, dvals, response_ts, rvals = _lag_demo_series()
    result = scan_lag_correlation(
        driver_ts, dvals, response_ts, rvals,
        lag_candidates=[-300.0, 0.0, 300.0, 600.0, 900.0],
        match_tolerance_seconds=120.0,
        min_pairs=3,
    )
    assert result["driver_point_count"] == 20 and result["response_point_count"] == 22
    assert result["candidate_count"] == 5
    assert result["best_lag"] == 600.0
    by_lag = {c["lag_seconds"]: c for c in result["candidates"]}
    best = by_lag[600.0]
    assert best["usable"] is True and best["reason"] is None
    assert best["pair_count"] == 20  # 20 个 driver 全部配对（response 共 22 个）
    assert best["correlation"] > 0.99
    # 真实时滞的 |correlation| 严格高于其余候选
    assert all(
        abs(best["correlation"]) > abs(c["correlation"])
        for lag, c in by_lag.items()
        if lag != 600.0
    )

    # 负时滞：response 领先 driver 120 秒，best_lag 应为 -120
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)
    n = 12
    vals = [5.0, 6.2, 5.8, 7.1, 8.0, 6.4, 5.5, 7.7, 8.9, 6.1, 5.2, 7.0]
    dts = [(base + timedelta(minutes=5 * i)).isoformat() for i in range(n)]
    rts = [
        (base + timedelta(seconds=-110) + timedelta(minutes=5 * i)).isoformat()
        for i in range(n)
    ]
    rvals2 = [v + 0.01 for v in vals]
    neg = scan_lag_correlation(
        dts, vals, rts, rvals2,
        lag_candidates=[0.0, -60.0, -120.0, -180.0],
        match_tolerance_seconds=30.0,
        min_pairs=3,
    )
    assert neg["best_lag"] == -120.0
    neg_by_lag = {c["lag_seconds"]: c for c in neg["candidates"]}
    assert neg_by_lag[-120.0]["pair_count"] == n
    assert neg_by_lag[-120.0]["correlation"] > 0.9999
    # 其余时滞在 30 秒容差内无任何配对
    assert all(
        c["pair_count"] == 0 and c["usable"] is False
        for lag, c in neg_by_lag.items()
        if lag != -120.0
    )
    print("✓ 时滞关联扫描：正/负时滞还原、配对数与 |correlation| 排序正确")


def test_lag_correlation_unusable_candidates() -> None:
    """不可用候选：配对数不足、单侧方差为零、无配对；全不可用时 best_lag 为 null。"""
    driver_ts, dvals, response_ts, rvals = _lag_demo_series()

    # min_pairs 超过可能配对数：全部候选不可用，best_lag 为 null
    result = scan_lag_correlation(
        driver_ts, dvals, response_ts, rvals,
        lag_candidates=[0.0, 600.0], match_tolerance_seconds=120.0, min_pairs=25,
    )
    assert result["best_lag"] is None
    assert result["usable_candidate_count"] == 0
    assert all(c["correlation"] is None for c in result["candidates"])
    assert all("配对数" in c["reason"] for c in result["candidates"])

    # driver 侧值恒定：方差为零，候选不可用
    const_driver = scan_lag_correlation(
        driver_ts, [7.0] * len(dvals), response_ts, rvals,
        lag_candidates=[600.0], match_tolerance_seconds=120.0, min_pairs=3,
    )
    cand = const_driver["candidates"][0]
    assert cand["pair_count"] == 20  # 配对照常进行，仅相关不可计算
    assert cand["usable"] is False and cand["correlation"] is None
    assert "driver" in cand["reason"] and "方差" in cand["reason"]
    assert const_driver["best_lag"] is None

    # response 侧值恒定：方差为零，候选不可用
    const_resp = scan_lag_correlation(
        driver_ts, dvals, response_ts, [3.0] * len(rvals),
        lag_candidates=[600.0], match_tolerance_seconds=120.0, min_pairs=3,
    )
    cand = const_resp["candidates"][0]
    assert cand["usable"] is False
    assert "response" in cand["reason"] and "方差" in cand["reason"]

    # 时滞过大：容差内无任何配对
    far = scan_lag_correlation(
        driver_ts, dvals, response_ts, rvals,
        lag_candidates=[10_000_000.0], match_tolerance_seconds=120.0, min_pairs=3,
    )
    assert far["candidates"][0]["pair_count"] == 0
    assert far["candidates"][0]["usable"] is False
    print("✓ 时滞关联：配对数不足 / 单侧零方差 / 无配对候选均正确标记不可用")


def test_lag_best_lag_tiebreak() -> None:
    """best_lag 决胜规则：|corr| 最大 → 配对数更多 → |lag| 更小 → lag 更小。"""

    def cand(lag: float, pairs: int, corr: float, usable: bool = True) -> dict:
        return {
            "lag_seconds": float(lag),
            "pair_count": pairs,
            "correlation": corr,
            "usable": usable,
            "reason": None,
        }

    # |correlation| 最大优先（与符号无关）
    assert _select_best_lag([cand(100, 5, 0.9), cand(-200, 3, -0.95)]) == -200.0
    # |corr| 相同：配对数更多者胜
    assert _select_best_lag([cand(100, 5, 0.9), cand(50, 8, -0.9)]) == 50.0
    # 配对数也相同：|lag| 更小者胜
    assert _select_best_lag([cand(100, 5, 0.9), cand(-50, 5, -0.9)]) == -50.0
    # |lag| 也相同：lag 更小（更负）者胜
    assert _select_best_lag([cand(50, 5, 0.9), cand(-50, 5, -0.9)]) == -50.0
    # 不可用候选不参与；全部不可用或空列表时返回 None
    assert _select_best_lag([cand(100, 5, 0.9, usable=False), cand(50, 5, 0.8)]) == 50.0
    assert _select_best_lag([cand(100, 5, 0.9, usable=False)]) is None
    assert _select_best_lag([]) is None
    print("✓ best_lag 决胜规则：|corr| → 配对数 → |lag| → lag，无可用候选为 null")


def _detection_record(
    detection_id: str, segments: list[tuple[str, str, float]], series_id: str | None = None
) -> dict:
    """构造供 aggregate_incidents 使用的最小检测记录（段：起、止、最大|z|）。"""
    return {
        "id": detection_id,
        "series_id": series_id,
        "created_at": "2026-09-22T00:00:00+00:00",
        "request_params": {},
        "result": {
            "segments": [
                {
                    "start_index": 0,
                    "end_index": 0,
                    "start_time": start,
                    "end_time": end,
                    "point_count": 1,
                    "anomaly_point_count": 1,
                    "score": {"max_abs_zscore": z},
                }
                for start, end, z in segments
            ]
        },
    }


def test_incident_aggregation_overlap_and_gap_boundary() -> None:
    """跨检测聚合：时间重叠合并；间隔 == gap 合并，超过 gap 不合并。"""
    records = [
        _detection_record("a", [("2026-09-22T00:00:00+00:00", "2026-09-22T00:10:00+00:00", 5.0)]),
        _detection_record("b", [("2026-09-22T00:05:00+00:00", "2026-09-22T00:15:00+00:00", 7.0)]),
        _detection_record("c", [("2026-09-22T00:20:00+00:00", "2026-09-22T00:25:00+00:00", 3.0)]),
    ]

    # a[00:00-00:10] 与 b[00:05-00:15] 重叠，始终同事件；
    # c 从 00:20 开始：距 b 结束（00:15）恰好 300s，gap=300 合并、gap=299 不合并
    merged = aggregate_incidents(records, gap_seconds=300)
    assert merged["detection_count"] == 3 and merged["segment_count"] == 3
    assert merged["incident_count"] == 1
    inc = merged["incidents"][0]
    assert inc["start_time"] == "2026-09-22T00:00:00+00:00"
    assert inc["end_time"] == "2026-09-22T00:25:00+00:00"
    assert inc["segment_count"] == 3
    assert inc["detection_ids"] == ["a", "b", "c"]
    assert inc["max_abs_zscore"] == 7.0
    # 成员按 start_time 稳定排序
    assert [(m["detection_id"], m["segment_index"]) for m in inc["members"]] == [
        ("a", 0), ("b", 0), ("c", 0)
    ]

    split = aggregate_incidents(records, gap_seconds=299)
    assert split["incident_count"] == 2
    assert split["incidents"][0]["detection_ids"] == ["a", "b"]
    assert split["incidents"][1]["detection_ids"] == ["c"]
    # 事件按起始时间排序
    assert split["incidents"][0]["start_time"] < split["incidents"][1]["start_time"]

    # gap=0：首尾相接（下一段起点 == 上一段终点）也合并
    touching = [
        _detection_record("a", [("2026-09-22T01:00:00+00:00", "2026-09-22T01:05:00+00:00", 4.0)]),
        _detection_record("b", [("2026-09-22T01:05:00+00:00", "2026-09-22T01:10:00+00:00", 6.0)]),
    ]
    assert aggregate_incidents(touching, gap_seconds=0)["incident_count"] == 1
    print("✓ 事件聚合：时间重叠 / gap 边界（含 == 与首尾相接）合并规则正确")


def test_incident_aggregation_same_detection_and_chaining() -> None:
    """同一检测相邻段可合并；链式邻近传递合并；成员保留原始段序号。"""
    records = [
        _detection_record(
            "a",
            [
                ("2026-09-22T02:00:00+00:00", "2026-09-22T02:01:00+00:00", 5.0),
                ("2026-09-22T02:05:00+00:00", "2026-09-22T02:06:00+00:00", 4.0),
            ],
        ),
        _detection_record("b", [("2026-09-22T02:03:00+00:00", "2026-09-22T02:03:30+00:00", 6.0)]),
    ]

    # gap=0：a 的两段相隔 4 分钟不合并；b 夹在中间且与 a#0 相隔 2 分钟也不合并 → 3 个事件
    assert aggregate_incidents(records, gap_seconds=0)["incident_count"] == 3

    # gap=120：a#0 与 b（间隔 120s）合并，桥到 b 后与 a#1（b 结束→a#1 开始 90s）链式合并
    chained = aggregate_incidents(records, gap_seconds=120)
    assert chained["incident_count"] == 1
    inc = chained["incidents"][0]
    assert inc["segment_count"] == 3
    assert inc["detection_ids"] == ["a", "b"]
    assert inc["max_abs_zscore"] == 6.0
    # 同一检测的两个段保留各自在来源检测中的原始段序号
    assert [(m["detection_id"], m["segment_index"]) for m in inc["members"]] == [
        ("a", 0), ("b", 0), ("a", 1)
    ]
    print("✓ 事件聚合：同检测相邻段合并、链式传递合并、成员段序号定位正确")


def test_incident_aggregation_sorting_and_empty() -> None:
    """成员稳定排序（同起点按 detection_id）、无异常检测、全部无异常时 0 事件。"""
    records = [
        _detection_record(
            "zzz",
            [("2026-09-22T03:00:00+00:00", "2026-09-22T03:05:00+00:00", 3.0)],
            series_id="series-z",
        ),
        _detection_record(
            "aaa",
            [
                ("2026-09-22T03:00:00+00:00", "2026-09-22T03:02:00+00:00", 8.0),
                ("2026-09-22T03:10:00+00:00", "2026-09-22T03:11:00+00:00", 2.0),
            ],
            series_id="series-a",
        ),
        _detection_record("empty", [], series_id=None),  # 无异常段，不贡献成员
    ]
    result = aggregate_incidents(records, gap_seconds=0)
    assert result["detection_count"] == 3 and result["segment_count"] == 3
    assert result["incident_count"] == 2
    first, second = result["incidents"]
    # 同一起点：按 detection_id 排序，aaa 段在 zzz 段之前
    assert [(m["detection_id"], m["segment_index"]) for m in first["members"]] == [
        ("aaa", 0), ("zzz", 0)
    ]
    assert first["max_abs_zscore"] == 8.0
    assert first["series_ids"] == ["series-a", "series-z"]  # null series_id 不列入
    assert [(m["detection_id"], m["segment_index"]) for m in second["members"]] == [("aaa", 1)]

    no_segments = aggregate_incidents(
        [_detection_record("e1", []), _detection_record("e2", [])], gap_seconds=60
    )
    assert no_segments == {
        "gap_seconds": 60.0,
        "detection_count": 2,
        "segment_count": 0,
        "incident_count": 0,
        "incidents": [],
    }
    print("✓ 事件聚合：成员稳定排序、series 覆盖、无异常检测与空事件列表正确")


def test_api_end_to_end(tmp_db: str = "/tmp/tsad_smoke.db") -> None:
    import os

    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DB_PATH"] = tmp_db

    # 重新初始化数据库（app 模块已导入，直接调用 init_db）
    from app import database

    database.DB_PATH = tmp_db
    database.init_db(tmp_db)

    with TestClient(app) as client:
        _exercise_api(client)


def _exercise_api(client: "TestClient") -> None:
    # 健康检查
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    # 提交检测
    payload = {
        "series_id": "test-series",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "include_points": True,
        "points": [
            {"timestamp": f"2026-09-20T08:{i:02d}:00", "value": 1.0 + (i % 2) * 0.05}
            for i in range(15)
        ]
        + [{"timestamp": "2026-09-20T08:15:00", "value": 9.0}],
    }
    r = client.post("/api/v1/detections", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    detection_id = body["id"]
    assert body["anomaly_segment_count"] == 1
    assert body["segments"][0]["triggers"][0]["timestamp"] == "2026-09-20T08:15:00+00:00"
    assert len(body["points"]) > 0

    # 按 ID 查询
    r = client.get(f"/api/v1/detections/{detection_id}")
    assert r.status_code == 200
    assert r.json()["id"] == detection_id
    assert r.json()["segments"][0]["baseline"]["window_size"] == 10

    # include_points=false 不返回逐点
    r = client.get(f"/api/v1/detections/{detection_id}", params={"include_points": False})
    assert r.json()["points"] is None

    # 列表 + series_id 过滤
    r = client.get("/api/v1/detections", params={"series_id": "test-series"})
    assert r.status_code == 200 and r.json()["total"] == 1

    # 404
    r = client.get("/api/v1/detections/nonexistent")
    assert r.status_code == 404

    # 参数校验：min_window > window_size
    bad = dict(payload)
    bad["min_window"], bad["window_size"] = 20, 10
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # guarded 模式：回显策略与重建统计，逐点注解带代次/重建标记
    guarded_payload = dict(payload)
    guarded_payload["series_id"] = "test-series-guarded"
    guarded_payload["baseline_policy"] = "guarded"
    guarded_payload["rebase_after"] = 3
    r = client.post("/api/v1/detections", json=guarded_payload)
    assert r.status_code == 201, r.text
    gbody = r.json()
    assert gbody["baseline_policy"] == "guarded"
    assert gbody["rebase_after"] == 3
    assert "baseline_rebuild_count" in gbody
    assert all(
        "baseline_generation" in p and "baseline_reset" in p
        for p in gbody["points"]
    )
    guarded_id = gbody["id"]

    # 历史结果取回时策略字段持久化保留
    r = client.get(f"/api/v1/detections/{guarded_id}")
    assert r.status_code == 200
    assert r.json()["baseline_policy"] == "guarded"
    assert r.json()["baseline_rebuild_count"] == gbody["baseline_rebuild_count"]

    # 参数校验：rebase_after < min_window
    bad = dict(payload)
    bad["baseline_policy"] = "guarded"
    bad["min_window"], bad["rebase_after"] = 3, 2
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 参数校验：非法 baseline_policy
    bad = dict(payload)
    bad["baseline_policy"] = "strict"
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 断档感知：两段规则采样 + 中间大断档，返回边界 / 间隔 / 逐点分段编号
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 9, 20, 20, 0, 0, tzinfo=timezone.utc)
    gap_points = []
    t = base
    for i in range(12):
        gap_points.append({"timestamp": t.isoformat(), "value": 10.0 + 0.05 * (i % 2)})
        t += timedelta(minutes=5)
    t = base + timedelta(hours=10)
    for i in range(12):
        gap_points.append({"timestamp": t.isoformat(), "value": 20.0 + 0.05 * (i % 2)})
        t += timedelta(minutes=5)
    gap_payload = {
        "series_id": "test-series-gap",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "gap_reset_factor": 3.0,
        "include_points": True,
        "points": gap_points,
    }
    r = client.post("/api/v1/detections", json=gap_payload)
    assert r.status_code == 201, r.text
    gp = r.json()
    assert gp["gap_reset_factor"] == 3.0
    assert gp["estimated_interval_seconds"] == 300.0
    assert gp["sampling_segment_count"] == 2
    assert len(gp["gap_boundaries"]) == 1
    gb = gp["gap_boundaries"][0]
    assert gb["before_index"] == 11 and gb["after_index"] == 12
    # 断档后水位 20 独立预热，不产生误报
    assert gp["anomaly_segment_count"] == 0
    assert all("sampling_segment_index" in p for p in gp["points"])
    assert {p["sampling_segment_index"] for p in gp["points"]} == {0, 1}

    # 历史结果取回时间断档字段持久化保留
    gap_id = gp["id"]
    r = client.get(f"/api/v1/detections/{gap_id}")
    assert r.status_code == 200
    assert r.json()["sampling_segment_count"] == 2
    assert r.json()["gap_boundaries"][0]["gap_seconds"] == gb["gap_seconds"]

    # 参数校验：启用断档但点数 < 3
    bad = dict(gap_payload)
    bad["points"] = gap_points[:2]
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422
    assert "gap_reset_factor" in r.text

    # 参数校验：gap_reset_factor 必须 > 1
    bad = dict(gap_payload)
    bad["gap_reset_factor"] = 1.0
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 趋势扫描：先升后降序列，返回方向相反的两个区间
    trend_points = []
    for i in range(40):
        v = 10 + 0.25 * i if i < 20 else 10 + 0.25 * (38 - i)
        trend_points.append(
            {"timestamp": f"2026-09-20T22:{i // 60:02d}:{i % 60:02d}Z", "value": v}
        )
    trend_payload = {
        "series_id": "test-series-trend",
        "window_size": 10,
        "step_size": 5,
        "alpha": 0.05,
        "min_slope": 0.1,
        "points": trend_points,
    }
    r = client.post("/api/v1/trends", json=trend_payload)
    assert r.status_code == 200, r.text
    tb = r.json()
    assert tb["series_id"] == "test-series-trend"
    assert tb["window_count"] == 7
    assert tb["significant_window_count"] == 6
    assert tb["trend_segment_count"] == 2
    assert [s["direction"] for s in tb["segments"]] == ["上升", "下降"]
    up_seg, down_seg = tb["segments"]
    assert (up_seg["start_index"], up_seg["end_index"]) == (0, 19)
    assert (down_seg["start_index"], down_seg["end_index"]) == (20, 39)
    assert up_seg["window_count"] == 3 and up_seg["min_p_value"] < 1e-3
    assert abs(up_seg["median_slope"] - 0.25) < 1e-9  # 1 秒间隔，0.25/秒
    assert abs(down_seg["median_slope"] + 0.25) < 1e-9
    assert len(up_seg["windows"]) == 3  # 成员窗口随区间返回
    assert {"s", "var_s", "z", "p_value", "slope", "significant"} <= set(up_seg["windows"][0])
    assert tb["windows"] is None  # 默认不返回全部窗口

    # include_windows=true 返回全部评估窗口（含不显著窗口）
    r = client.post("/api/v1/trends", json={**trend_payload, "include_windows": True})
    assert r.status_code == 200, r.text
    tw = r.json()["windows"]
    assert len(tw) == 7
    assert tw[3]["significant"] is False and tw[3]["direction"] == "平稳"

    # 参数校验：点数少于 window_size
    bad = dict(trend_payload)
    bad["points"] = trend_points[:5]
    r = client.post("/api/v1/trends", json=bad)
    assert r.status_code == 422

    # 参数校验：window_size < 3 / step_size < 1 / alpha 越界 / min_slope 为负
    for field, value in [
        ("window_size", 2),
        ("step_size", 0),
        ("alpha", 0),
        ("alpha", 1),
        ("min_slope", -0.1),
    ]:
        bad = dict(trend_payload)
        bad[field] = value
        r = client.post("/api/v1/trends", json=bad)
        assert r.status_code == 422, (field, value, r.text)

    # 参数校验：重复时间戳
    bad = dict(trend_payload)
    bad["points"] = trend_points[:10] + [trend_points[9]]
    r = client.post("/api/v1/trends", json=bad)
    assert r.status_code == 422
    assert "重复时间戳" in r.text

    # 时滞关联扫描：response 滞后 driver 600 秒，best_lag 应还原该时滞
    lag_base = datetime(2026, 9, 21, 8, 0, 0, tzinfo=timezone.utc)
    lag_dvals = [10.0, 11.0, 12.5, 13.5, 12.5, 11.0, 10.0, 9.0, 8.0, 9.0,
                 10.5, 12.0, 13.5, 14.5, 13.5, 12.0, 10.5, 9.5, 9.0, 10.0]
    lag_driver = [
        {"timestamp": (lag_base + timedelta(minutes=5 * i)).isoformat(), "value": v}
        for i, v in enumerate(lag_dvals)
    ]
    lag_response = [
        {"timestamp": (lag_base + timedelta(minutes=1)).isoformat(), "value": 10.05},
        {"timestamp": (lag_base + timedelta(minutes=6)).isoformat(), "value": 10.95},
    ] + [
        {
            "timestamp": (lag_base + timedelta(minutes=1) + timedelta(minutes=5 * j)).isoformat(),
            "value": round(lag_dvals[j - 2] + (0.05 if j % 2 == 0 else -0.05), 2),
        }
        for j in range(2, 22)
    ]
    lag_payload = {
        "driver_points": lag_driver,
        "response_points": lag_response,
        "lag_seconds": [-300, 0, 300, 600, 900],
        "match_tolerance_seconds": 120,
        "min_pairs": 3,
    }
    r = client.post("/api/v1/lag-correlation", json=lag_payload)
    assert r.status_code == 200, r.text
    lb = r.json()
    assert lb["driver_point_count"] == 20 and lb["response_point_count"] == 22
    assert lb["match_tolerance_seconds"] == 120 and lb["min_pairs"] == 3
    assert lb["candidate_count"] == 5 and lb["usable_candidate_count"] == 5
    assert lb["best_lag"] == 600.0
    by_lag = {c["lag_seconds"]: c for c in lb["candidates"]}
    assert by_lag[600.0]["pair_count"] == 20
    assert by_lag[600.0]["correlation"] > 0.99
    assert by_lag[600.0]["usable"] is True and by_lag[600.0]["reason"] is None
    assert {"lag_seconds", "pair_count", "correlation", "usable", "reason"} == set(
        lb["candidates"][0]
    )

    # 输入无需预先排序：乱序提交结果一致
    shuffled = dict(lag_payload)
    shuffled["driver_points"] = list(reversed(lag_driver))
    r = client.post("/api/v1/lag-correlation", json=shuffled)
    assert r.status_code == 200 and r.json()["best_lag"] == 600.0

    # min_pairs 超过配对数：候选全部不可用、best_lag 为 null（仍为 200）
    high = dict(lag_payload)
    high["min_pairs"] = 25
    r = client.post("/api/v1/lag-correlation", json=high)
    assert r.status_code == 200, r.text
    hb = r.json()
    assert hb["best_lag"] is None and hb["usable_candidate_count"] == 0
    assert all(c["correlation"] is None and c["reason"] for c in hb["candidates"])

    # driver 值恒定：方差为零，候选不可用
    const = dict(lag_payload)
    const["driver_points"] = [dict(p, value=5.0) for p in lag_driver]
    r = client.post("/api/v1/lag-correlation", json=const)
    assert r.status_code == 200
    assert r.json()["best_lag"] is None
    assert "driver" in r.json()["candidates"][0]["reason"]

    # 参数校验：lag_seconds 为空 / 含非有限值
    bad = dict(lag_payload)
    bad["lag_seconds"] = []
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422
    # httpx 的 json= 序列化拒绝非有限浮点，需手工构造含 NaN 的原始请求体
    import json as _json

    bad = dict(lag_payload)
    bad["lag_seconds"] = [0, float("nan")]
    r = client.post(
        "/api/v1/lag-correlation",
        content=_json.dumps(bad),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422

    # 参数校验：match_tolerance_seconds 为负 / 非有限
    bad = dict(lag_payload)
    bad["match_tolerance_seconds"] = -1
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422
    bad = dict(lag_payload)
    bad["match_tolerance_seconds"] = float("inf")
    r = client.post(
        "/api/v1/lag-correlation",
        content=_json.dumps(bad),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422

    # 参数校验：min_pairs < 2、points 为空
    bad = dict(lag_payload)
    bad["min_pairs"] = 1
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422
    bad = dict(lag_payload)
    bad["driver_points"] = []
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422

    # 参数校验：driver / response 序列内部重复时间戳（错误消息带序列标识）
    bad = dict(lag_payload)
    bad["driver_points"] = lag_driver + [lag_driver[0]]
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422 and "driver_points" in r.text
    bad = dict(lag_payload)
    bad["response_points"] = lag_response + [lag_response[0]]
    r = client.post("/api/v1/lag-correlation", json=bad)
    assert r.status_code == 422 and "response_points" in r.text

    # 事件聚合：构造 3 条已知异常段的检测（服务端按时间重排）。
    # 公共基线 09:00~09:14（15 个点）：
    # A：09:15 单点尖刺 → 1 段 [09:15]；
    # B：09:25 与 09:30 两个尖刺，中间夹 09:26~09:29 四个正常点
    #    （merge_gap=1 不桥接）→ 2 段 [09:25]、[09:30]，间隔恰好 300s；
    # C：无异常段。A 段结束（09:15）距 B 首段开始（09:25）恰好 600s。
    def spike_payload(series: str, extra_points: list[dict]) -> dict:
        points = [
            {"timestamp": f"2026-09-22T09:{i:02d}:00", "value": 1.0 + (i % 2) * 0.05}
            for i in range(15)
        ]
        return {
            "series_id": series,
            "window_size": 10,
            "min_window": 3,
            "threshold": 3.0,
            "merge_gap": 1,
            "points": points + extra_points,
        }

    def create_detection(series: str, extra_points: list[dict]) -> tuple[str, list[dict]]:
        resp = client.post("/api/v1/detections", json=spike_payload(series, extra_points))
        assert resp.status_code == 201, resp.text
        return resp.json()["id"], resp.json()["segments"]

    det_a, segs_a = create_detection(
        "incident-series-a", [{"timestamp": "2026-09-22T09:15:00", "value": 9.0}]
    )
    assert len(segs_a) == 1 and segs_a[0]["start_time"] == "2026-09-22T09:15:00+00:00"

    det_b, segs_b = create_detection(
        "incident-series-b",
        [
            {"timestamp": "2026-09-22T09:25:00", "value": 9.0},
            {"timestamp": "2026-09-22T09:26:00", "value": 1.0},
            {"timestamp": "2026-09-22T09:27:00", "value": 1.05},
            {"timestamp": "2026-09-22T09:28:00", "value": 1.0},
            {"timestamp": "2026-09-22T09:29:00", "value": 1.05},
            {"timestamp": "2026-09-22T09:30:00", "value": 9.0},
        ],
    )
    assert [s["start_time"] for s in segs_b] == [
        "2026-09-22T09:25:00+00:00",
        "2026-09-22T09:30:00+00:00",
    ]

    det_c, segs_c = create_detection("incident-series-c", [])  # 无异常段
    assert segs_c == []

    # gap=300：B 的两段间隔恰好 300s（同检测相邻段）→ 合并为一个事件；
    # A(09:15) 距 B#0(09:25) 600s > 300 → 独立事件。事件按起始时间排序。
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_b, det_a, det_c], "gap_seconds": 300},
    )
    assert r.status_code == 200, r.text
    ib = r.json()
    assert ib["gap_seconds"] == 300.0
    assert ib["detection_count"] == 3 and ib["segment_count"] == 3
    assert ib["incident_count"] == 2
    inc_a, inc_b = ib["incidents"]
    assert inc_a["start_time"] == segs_a[0]["start_time"]
    assert inc_a["end_time"] == segs_a[0]["end_time"]
    assert inc_a["detection_ids"] == [det_a] and inc_a["segment_count"] == 1
    assert inc_b["start_time"] == segs_b[0]["start_time"]
    assert inc_b["end_time"] == segs_b[1]["end_time"]
    assert inc_b["detection_ids"] == [det_b]
    assert inc_b["series_ids"] == ["incident-series-b"]
    assert inc_b["segment_count"] == 2
    assert inc_b["max_abs_zscore"] == max(
        s["score"]["max_abs_zscore"] for s in segs_b
    )
    # 成员定位：同检测内按段序号稳定排序
    assert [
        (m["detection_id"], m["segment_index"], m["start_time"]) for m in inc_b["members"]
    ] == [
        (det_b, 0, segs_b[0]["start_time"]),
        (det_b, 1, segs_b[1]["start_time"]),
    ]
    # 成员定位可回溯到来源段
    m = inc_a["members"][0]
    assert (m["start_index"], m["end_index"]) == (
        segs_a[0]["start_index"], segs_a[0]["end_index"]
    )
    assert m["max_abs_zscore"] == segs_a[0]["score"]["max_abs_zscore"]
    assert m["series_id"] == "incident-series-a"

    # gap=600：A→B#0 间隔恰好 600s，再链式桥到 B#1，三段合并为一个事件
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_a, det_b], "gap_seconds": 600},
    )
    assert r.status_code == 200, r.text
    assert r.json()["incident_count"] == 1
    merged = r.json()["incidents"][0]
    assert merged["start_time"] == segs_a[0]["start_time"]
    assert merged["end_time"] == segs_b[1]["end_time"]
    assert merged["detection_ids"] == sorted([det_a, det_b])
    assert merged["segment_count"] == 3
    # 成员按 start_time 排序：A(09:15)、B#0(09:25)、B#1(09:30)
    assert [
        (m["detection_id"], m["segment_index"]) for m in merged["members"]
    ] == [(det_a, 0), (det_b, 0), (det_b, 1)]

    # 所有检测均无异常段：incident_count=0
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_c], "gap_seconds": 0},
    )
    assert r.status_code == 200
    assert r.json() == {
        "gap_seconds": 0.0,
        "detection_count": 1,
        "segment_count": 0,
        "incident_count": 0,
        "incidents": [],
    }

    # 404：未知 detection_id（同时混入已知 ID，错误详情包含未知 ID）
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_a, "nonexistent-id"], "gap_seconds": 0},
    )
    assert r.status_code == 404 and "nonexistent-id" in r.text

    # 422：空列表 / 超过 50 个 / 重复 ID / 负间隔 / 非有限间隔
    r = client.post("/api/v1/incidents/aggregate", json={"detection_ids": [], "gap_seconds": 0})
    assert r.status_code == 422
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [f"id{i}" for i in range(51)], "gap_seconds": 0},
    )
    assert r.status_code == 422
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_a, det_a], "gap_seconds": 0},
    )
    assert r.status_code == 422 and "重复" in r.text
    r = client.post(
        "/api/v1/incidents/aggregate",
        json={"detection_ids": [det_a], "gap_seconds": -1},
    )
    assert r.status_code == 422
    r = client.post(
        "/api/v1/incidents/aggregate",
        content=_json.dumps({"detection_ids": [det_a], "gap_seconds": float("nan")}),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422

    # 只读保证：聚合之后原检测记录仍可原样取回、段内容不变
    r = client.get(f"/api/v1/detections/{det_b}")
    assert r.status_code == 200
    assert [
        (s["start_index"], s["end_index"]) for s in r.json()["segments"]
    ] == [(s["start_index"], s["end_index"]) for s in segs_b]

    print(
        "✓ API 端到端测试通过：提交 / 查询 / 列表 / 404 / 参数校验 / guarded 策略 / "
        "趋势扫描 / 时滞关联 / 事件聚合"
    )


if __name__ == "__main__":
    test_algorithm_spikes_and_shift()
    test_constant_baseline()
    test_no_anomaly()
    test_rolling_defaults_unchanged()
    test_guarded_isolated_spike_excluded()
    test_guarded_rebuild_on_consecutive_anomalies()
    test_gap_aware_split_and_warmup()
    test_gap_aware_baseline_restart_and_no_merge_across_gap()
    test_gap_aware_guarded_resets_state()
    test_gap_aware_defaults_compatible()
    test_mann_kendall_ties_quantitative()
    test_theil_sen_irregular_sampling()
    test_trend_direction_switch()
    test_trend_merge_adjacent_and_gap()
    test_trend_min_slope_filter()
    test_lag_matching_rules()
    test_lag_correlation_scan_basic()
    test_lag_correlation_unusable_candidates()
    test_lag_best_lag_tiebreak()
    test_incident_aggregation_overlap_and_gap_boundary()
    test_incident_aggregation_same_detection_and_chaining()
    test_incident_aggregation_sorting_and_empty()
    test_api_end_to_end()
    print("\n全部冒烟测试通过 ✅")
