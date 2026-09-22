"""稳健滑动窗口异常检测核心算法。

思路：
1. 对每个点，取其之前 ``window_size`` 个点（严格 trailing window）计算
   稳健基线：中位数 median 与 MAD（median absolute deviation）。
2. 以 MAD * 1.4826 作为稳健标准差估计，计算稳健 z 分数：
       z = (x - median) / (1.4826 * MAD)
   当 |z| >= threshold 时标记为异常点（突变 / 水平偏移）。
3. 允许把相邻异常点之间至多 ``merge_gap`` 个正常点桥接，合并成异常区间。
4. 每个异常区间给出基线、分数与触发点的人类可读解释。

基线策略 ``baseline_policy``：
- ``rolling``（默认，保持原始行为）：所有历史点一律滑入基线窗口；
- ``guarded``：只有判为正常的点才允许进入后续基线，孤立尖刺不会污染
  其后的中位数与离散度；当连续异常点数达到 ``rebase_after`` 时，用该异常
  段最近 ``min_window`` 个点重建基线并递增代次（generation），触发点本身
  仍保持异常，且区间不因此被拆分。

采样断档感知 ``gap_reset_factor``（默认关闭）：
- 启用（>1）时，以排序后相邻点**正时间差的中位数**估算常规采样间隔，
  相邻间隔超过该倍数即视为断档并划分新采样分段；
- 每个分段独立执行 ``min_window`` 预热，rolling/guarded 的基线、
  guarded 的连续异常计数与重建代次均在分段起点从头开始；
- 异常区间严格不跨越断档合并；响应返回断档边界、估算间隔与逐点分段编号。
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Any

from .models import parse_timestamp

# 正态分布下 MAD 与标准差的换算系数
MAD_SCALE = 1.4826
# 基线恒定时（MAD=0 且标准差=0）又与基线不一致，使用的有限大 z 分数哨兵值
ZERO_MAD_ZSCORE = 999.0
# 启用断档感知所需的最少点数（少于 3 个点无法估算相邻间隔）
MIN_POINTS_FOR_GAP_AWARE = 3


def _sample_std(values: list[float], mean: float) -> float:
    """样本标准差（n-1 修正）；点数不足 2 时返回 0。"""
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _window_stats(history: list[float]) -> tuple[float, float, float, str]:
    """返回 (中位数, 原始MAD, 稳健离散度, 离散度方法)。

    优先使用 1.4826*MAD；当 MAD=0（小窗口落在多峰 / 离散信号上会误判）时，
    回退到样本标准差；两者都为 0 才视为真正退化的恒定基线。
    """
    med = median(history)
    mad = median([abs(x - med) for x in history])
    robust_sd = mad * MAD_SCALE
    if robust_sd > 0:
        return med, mad, robust_sd, "mad"
    mean = sum(history) / len(history)
    std = _sample_std(history, mean)
    if std > 0:
        return med, mad, std, "std_fallback"
    return med, mad, 0.0, "constant"


def _zscore(value: float, med: float, robust_sd: float) -> tuple[float, bool]:
    """计算稳健 z 分数；返回 (zscore, 基线是否退化)。"""
    degenerate = robust_sd == 0.0
    if degenerate:
        return (0.0 if value == med else ZERO_MAD_ZSCORE), True
    return (value - med) / robust_sd, False


def _compute_gap_split(
    timestamps: list[str],
    gap_reset_factor: float,
) -> tuple[list[tuple[int, int]], float, list[dict[str, Any]]]:
    """按相邻时间差检测采样断档，返回 (分段边界, 估算常规间隔秒数, 断档边界)。

    以排序后相邻点**正时间差**的中位数估算常规采样间隔；
    某个间隔超过 ``gap_reset_factor`` 倍常规间隔即视为断档，
    断档两侧划分为不同分段。需要至少 3 个点。
    """
    n = len(timestamps)
    if n < MIN_POINTS_FOR_GAP_AWARE:
        raise ValueError(
            f"启用 gap_reset_factor 至少需要 {MIN_POINTS_FOR_GAP_AWARE} 个点，当前只有 {n} 个"
        )

    dts = [parse_timestamp(ts) for ts in timestamps]
    deltas = [
        (dts[i] - dts[i - 1]).total_seconds()
        for i in range(1, n)
        if (dts[i] - dts[i - 1]).total_seconds() > 0
    ]
    if not deltas:
        # 理论上不会出现（时间戳已去重排序），兜底为单一分段
        return [(0, n)], 0.0, []
    median_interval = median(deltas)

    bounds: list[tuple[int, int]] = []
    boundaries: list[dict[str, Any]] = []
    seg_start = 0
    for i in range(1, n):
        delta = (dts[i] - dts[i - 1]).total_seconds()
        if delta > 0 and median_interval > 0 and delta > gap_reset_factor * median_interval:
            bounds.append((seg_start, i))
            boundaries.append(
                {
                    "before_index": i - 1,
                    "after_index": i,
                    "before_time": timestamps[i - 1],
                    "after_time": timestamps[i],
                    "gap_seconds": delta,
                    "interval_seconds": median_interval,
                    "gap_ratio": delta / median_interval,
                }
            )
            seg_start = i
    bounds.append((seg_start, n))
    return bounds, median_interval, boundaries


def analyze_series(
    timestamps: list[str],
    values: list[float],
    window_size: int = 20,
    min_window: int = 5,
    threshold: float = 3.0,
    merge_gap: int = 2,
    baseline_policy: str = "rolling",
    rebase_after: int | None = None,
    gap_reset_factor: float | None = None,
) -> dict[str, Any]:
    """执行检测并返回可 JSON 序列化的完整结果。

    :param timestamps: 已排序、已去重的 ISO-8601 时间戳字符串
    :param values: 与 timestamps 对齐的数值
    :param baseline_policy: ``rolling``（默认，所有点滑入基线）或
        ``guarded``（仅正常点进入基线，连续异常达到 ``rebase_after`` 时重建）
    :param rebase_after: guarded 模式下触发基线重建的连续异常点数，
        默认取 ``min_window``，且必须不小于 ``min_window``；
        rolling 模式忽略该参数
    :param gap_reset_factor: 大于 1 时启用采样断档感知：以排序后相邻点
        正时间差的中位数估算常规采样间隔，间隔超过该倍数即划分新分段；
        每个分段独立预热（min_window）、独立维护基线 / 连续异常计数 /
        重建代次，异常区间不跨越断档合并。至少需要 3 个点
    """
    n = len(values)
    # 每个被评估点的注解；未评估点（预热段）保持 None
    annotations: list[dict[str, Any] | None] = [None] * n
    is_anomaly = [False] * n
    zscores: list[float | None] = [None] * n
    # guarded 模式下各异常点评分时实际使用的基线成员（rolling 不需要）
    baseline_windows: list[tuple[int, ...] | None] = [None] * n
    baseline_rebuild_count = 0

    gap_enabled = gap_reset_factor is not None
    if gap_enabled:
        segment_bounds, median_interval, gap_boundaries = _compute_gap_split(
            timestamps, gap_reset_factor  # type: ignore[arg-type]
        )
        point_segments = [0] * n
        for seg_idx, (lo, hi) in enumerate(segment_bounds):
            for k in range(lo, hi):
                point_segments[k] = seg_idx
        # 首个可评估点索引：第一个足以完成预热的分段起点 + min_window
        evaluated_start = next(
            (lo + min_window for lo, hi in segment_bounds if hi - lo > min_window), n
        )
    else:
        segment_bounds = [(0, n)]
        point_segments = [0] * n
        median_interval = None
        gap_boundaries = []
        evaluated_start = min_window

    if baseline_policy == "guarded":
        if rebase_after is None:
            rebase_after = min_window
        baseline_rebuild_count = _run_guarded(
            timestamps=timestamps,
            values=values,
            window_size=window_size,
            min_window=min_window,
            threshold=threshold,
            rebase_after=rebase_after,
            segment_bounds=segment_bounds,
            point_segments=point_segments,
            annotations=annotations,
            is_anomaly=is_anomaly,
            zscores=zscores,
            baseline_windows=baseline_windows,
        )
    else:
        _run_rolling(
            timestamps=timestamps,
            values=values,
            window_size=window_size,
            min_window=min_window,
            threshold=threshold,
            segment_bounds=segment_bounds,
            point_segments=point_segments,
            annotations=annotations,
            is_anomaly=is_anomaly,
            zscores=zscores,
        )

    segments = _build_segments(
        timestamps=timestamps,
        values=values,
        is_anomaly=is_anomaly,
        zscores=zscores,
        annotations=annotations,
        baseline_windows=baseline_windows,
        window_size=window_size,
        threshold=threshold,
        merge_gap=merge_gap,
        segment_bounds=segment_bounds,
        point_segments=point_segments,
    )

    evaluated_indices = [i for i, ann in enumerate(annotations) if ann is not None]
    return {
        "point_count": n,
        "window_size": window_size,
        "min_window": min_window,
        "threshold": threshold,
        "merge_gap": merge_gap,
        "baseline_policy": baseline_policy,
        # rolling 不使用 rebase_after，统一回显 null；guarded 回显生效值
        "rebase_after": rebase_after if baseline_policy == "guarded" else None,
        "baseline_rebuild_count": baseline_rebuild_count,
        # 采样断档感知（默认关闭）；关闭时以下字段取兼容默认值
        "gap_reset_factor": gap_reset_factor,
        "estimated_interval_seconds": median_interval,
        "gap_boundaries": gap_boundaries,
        "sampling_segment_count": len(segment_bounds),
        "evaluated_start_index": evaluated_start,
        "evaluated_point_count": len(evaluated_indices),
        "anomaly_point_count": sum(1 for i in evaluated_indices if is_anomaly[i]),
        "anomaly_segment_count": len(segments),
        "segments": segments,
        "points": [annotations[i] for i in evaluated_indices],
    }


def _run_rolling(
    *,
    timestamps: list[str],
    values: list[float],
    window_size: int,
    min_window: int,
    threshold: float,
    segment_bounds: list[tuple[int, int]],
    point_segments: list[int],
    annotations: list[dict[str, Any] | None],
    is_anomaly: list[bool],
    zscores: list[float | None],
) -> None:
    """rolling 策略：所有历史点一律滑入 trailing window。

    每个采样分段独立预热（前 min_window 个点不评分）、基线从空开始；
    窗口绝不跨越断档（只取当前分段内的历史点）。
    """
    for seg_lo, seg_hi in segment_bounds:
        for i in range(seg_lo + min_window, seg_hi):
            lo = max(seg_lo, i - window_size)
            history = values[lo:i]
            med, mad, robust_sd, dispersion_method = _window_stats(history)
            z, degenerate = _zscore(values[i], med, robust_sd)
            zscores[i] = z
            is_anomaly[i] = abs(z) >= threshold

            annotations[i] = {
                "index": i,
                "timestamp": timestamps[i],
                "value": values[i],
                "baseline_median": med,
                "baseline_mad": mad,
                "baseline_robust_sd": robust_sd,
                "dispersion_method": dispersion_method,
                "zscore": z,
                "degenerate_baseline": degenerate,
                "is_anomaly": is_anomaly[i],
                # rolling 不区分代次，也不发生重建
                "baseline_generation": 0,
                "baseline_reset": False,
                "sampling_segment_index": point_segments[i],
            }


def _run_guarded(
    *,
    timestamps: list[str],
    values: list[float],
    window_size: int,
    min_window: int,
    threshold: float,
    rebase_after: int,
    segment_bounds: list[tuple[int, int]],
    point_segments: list[int],
    annotations: list[dict[str, Any] | None],
    is_anomaly: list[bool],
    zscores: list[float | None],
    baseline_windows: list[tuple[int, ...] | None],
) -> int:
    """guarded 策略：只有正常点进入基线，长异常段触发重建。

    返回基线重建（代次递增）次数。每个采样分段从头开始：种子窗口、
    连续异常计数、基线代次均在分段起点重置，基线成员不跨越断档。
    """
    rebuild_count = 0

    for seg_lo, seg_hi in segment_bounds:
        # 分段点数不足以完成预热：整段不评分
        if seg_hi - seg_lo <= min_window:
            continue

        # 基线只容纳已接纳点的索引（重建后为异常段末尾的 min_window 个点）；
        # 正常点滑入、异常点排除，因此成员索引在原序列上可能不连续。
        accepted: deque[int] = deque(range(seg_lo, seg_lo + min_window))
        generation = 0
        consecutive_anomalies = 0
        # 连续异常期间 accepted 不变，多个异常点可共享同一份窗口成员快照
        window_snapshot: tuple[int, ...] | None = tuple(accepted)

        for i in range(seg_lo + min_window, seg_hi):
            members = tuple(accepted)
            history = [values[k] for k in members]
            med, mad, robust_sd, dispersion_method = _window_stats(history)
            z, degenerate = _zscore(values[i], med, robust_sd)
            zscores[i] = z
            anomaly = abs(z) >= threshold
            is_anomaly[i] = anomaly
            # 该点评分所使用的代次（重建发生在评分之后，不影响本点评分）
            scoring_generation = generation

            reset = False
            if anomaly:
                # 记录该点评分时实际使用的基线成员（供异常区间展示，正常点无需记录）
                if window_snapshot is None:
                    window_snapshot = members
                baseline_windows[i] = window_snapshot
                consecutive_anomalies += 1
                if consecutive_anomalies >= rebase_after:
                    # 用该异常段最近 min_window 个点重建基线并递增代次；
                    # 当前点仍是异常，且原异常区间不被拆分。
                    accepted = deque(range(i - min_window + 1, i + 1))
                    generation += 1
                    rebuild_count += 1
                    consecutive_anomalies = 0
                    reset = True
                    window_snapshot = None
            else:
                # 只有判为正常的点允许进入后续基线，孤立尖刺被排除在外
                consecutive_anomalies = 0
                accepted.append(i)
                if len(accepted) > window_size:
                    accepted.popleft()
                window_snapshot = None

            annotations[i] = {
                "index": i,
                "timestamp": timestamps[i],
                "value": values[i],
                "baseline_median": med,
                "baseline_mad": mad,
                "baseline_robust_sd": robust_sd,
                "dispersion_method": dispersion_method,
                "zscore": z,
                "degenerate_baseline": degenerate,
                "is_anomaly": anomaly,
                "baseline_generation": scoring_generation,
                "baseline_reset": reset,
                "sampling_segment_index": point_segments[i],
            }

    return rebuild_count


def _build_segments(
    *,
    timestamps: list[str],
    values: list[float],
    is_anomaly: list[bool],
    zscores: list[float | None],
    annotations: list[dict[str, Any] | None],
    window_size: int,
    threshold: float,
    merge_gap: int,
    segment_bounds: list[tuple[int, int]],
    point_segments: list[int],
    baseline_windows: list[tuple[int, ...] | None] | None = None,
) -> list[dict[str, Any]]:
    """扫描异常布尔序列，按 merge_gap 合并相邻异常点为区间。

    扫描严格限制在每个采样分段内部：桥接不会跨过断档，异常区间
    不可能把断档两侧的异常合并为同一段。
    """
    segments: list[dict[str, Any]] = []

    for seg_lo, seg_hi in segment_bounds:
        i = seg_lo
        while i < seg_hi:
            if not is_anomaly[i]:
                i += 1
                continue

            seg_start = i
            last_anomaly = i
            j = i + 1
            # 相邻两个异常点索引差 <= merge_gap + 1 即允许桥接，且不得越过 seg_hi
            while j < seg_hi and (j - last_anomaly) <= merge_gap + 1:
                if is_anomaly[j]:
                    last_anomaly = j
                j += 1
            seg_end = last_anomaly

            described = _describe_segment(
                seg_start=seg_start,
                seg_end=seg_end,
                timestamps=timestamps,
                values=values,
                is_anomaly=is_anomaly,
                zscores=zscores,
                annotations=annotations,
                window_size=window_size,
                threshold=threshold,
                segment_lo=seg_lo,
                baseline_windows=baseline_windows,
            )
            described["sampling_segment_index"] = point_segments[seg_start]
            segments.append(described)
            i = seg_end + 1

    return segments


def _describe_segment(
    *,
    seg_start: int,
    seg_end: int,
    timestamps: list[str],
    values: list[float],
    is_anomaly: list[bool],
    zscores: list[float | None],
    annotations: list[dict[str, Any] | None],
    window_size: int,
    threshold: float,
    segment_lo: int = 0,
    baseline_windows: list[tuple[int, ...] | None] | None = None,
) -> dict[str, Any]:
    """构造单个异常区间：基线、分数、触发点与解释。"""
    # 区间基线取第一个触发点评分时实际使用的基线窗口（突变发生前的基线）。
    # rolling 下即 seg_start 之前的连续 trailing window（不早于所在分段起点）；
    # guarded 下只含已接纳点，成员索引在原序列上可能不连续（被排除的尖刺
    # 不会出现在这里），且天然局限在分段内部。
    if baseline_windows is not None and baseline_windows[seg_start] is not None:
        member_indices = baseline_windows[seg_start]  # type: ignore[assignment]
        base_lo, base_hi = member_indices[0], member_indices[-1] + 1
        baseline = [values[k] for k in member_indices]
    else:
        base_lo = max(segment_lo, seg_start - window_size)
        base_hi = seg_start
        baseline = [values[k] for k in range(base_lo, base_hi)]
    med, mad, robust_sd, dispersion_method = _window_stats(baseline)

    trigger_indices = [k for k in range(seg_start, seg_end + 1) if is_anomaly[k]]
    triggers: list[dict[str, Any]] = []
    max_idx = trigger_indices[0]
    for k in trigger_indices:
        z = zscores[k]
        assert z is not None
        ann = annotations[k]
        assert ann is not None
        point_med = ann["baseline_median"]
        point_sd = ann["baseline_robust_sd"]
        point_method = ann["dispersion_method"]
        deviation = values[k] - point_med
        direction = "上升" if deviation > 0 else "下降"
        if ann["degenerate_baseline"]:
            reason = (
                f"基线窗口内 MAD=0 且标准差=0（基线恒定为 {point_med:g}），观测值 {values[k]:g} "
                f"与基线不一致，按最强突变处理（z={z:g}，阈值 {threshold:g}）"
            )
        elif point_method == "std_fallback":
            reason = (
                f"基线窗口 MAD=0，回退为样本标准差尺度：z 分数 {z:+.2f} 的绝对值"
                f"超过阈值 {threshold:g}：观测值 {values[k]:g} 相对滑动基线中位数 {point_med:g} "
                f"{direction}偏离 {abs(deviation):g}（样本标准差 {point_sd:.4g}）"
            )
        else:
            reason = (
                f"稳健 z 分数 {z:+.2f} 的绝对值超过阈值 {threshold:g}："
                f"观测值 {values[k]:g} 相对滑动基线中位数 {point_med:g} "
                f"{direction}偏离 {abs(deviation):g}（1.4826·MAD 稳健标准差 {point_sd:.4g}）"
            )
        triggers.append(
            {
                "index": k,
                "timestamp": timestamps[k],
                "value": values[k],
                "zscore": z,
                "baseline_median": point_med,
                "baseline_mad": ann["baseline_mad"],
                "dispersion_method": point_method,
                "deviation": deviation,
                "direction": direction,
                "reason": reason,
            }
        )
        if abs(z) > abs(zscores[max_idx] or 0.0):
            max_idx = k

    seg_z = [zscores[k] for k in range(seg_start, seg_end + 1) if zscores[k] is not None]
    mean_abs_z = sum(abs(z) for z in seg_z) / len(seg_z)
    max_z = zscores[max_idx]
    length = seg_end - seg_start + 1
    span_direction = triggers[0]["direction"]

    explanation = (
        f"区间 {timestamps[seg_start]} ~ {timestamps[seg_end]}（{length} 个点，"
        f"其中 {len(triggers)} 个点触发）：首个触发点为 {timestamps[seg_start]}，"
        f"值 {values[seg_start]:g} 相对基线中位数 {med:g} {span_direction}突变；"
        f"区间最大 |z|={abs(max_z):.2f}（{timestamps[max_idx]}，值 {values[max_idx]:g}），"
        f"平均 |z|={mean_abs_z:.2f}，判定阈值 {threshold:g}。"
    )

    return {
        "start_index": seg_start,
        "end_index": seg_end,
        "start_time": timestamps[seg_start],
        "end_time": timestamps[seg_end],
        "point_count": length,
        "anomaly_point_count": len(triggers),
        "baseline": {
            "median": med,
            "mad": mad,
            "robust_sd": robust_sd,
            "dispersion_method": dispersion_method,
            "scale": MAD_SCALE,
            "window_size": len(baseline),
            "window_start_time": timestamps[base_lo],
            "window_end_time": timestamps[base_hi - 1],
        },
        "score": {
            "max_abs_zscore": abs(max_z),
            "max_signed_zscore": max_z,
            "max_zscore_index": max_idx,
            "max_zscore_time": timestamps[max_idx],
            "mean_abs_zscore": mean_abs_z,
        },
        "triggers": triggers,
        "explanation": explanation,
    }
