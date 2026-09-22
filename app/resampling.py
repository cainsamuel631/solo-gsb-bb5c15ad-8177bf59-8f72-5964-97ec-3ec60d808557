"""无状态等间隔重采样：anchor 对齐网格 + 缺口感知补值。

输入任意非等距采样序列，输出位于**首末样本闭区间**内、与 ``anchor_time``
对齐的等间隔网格点。网格时间一律用**整数微秒**计算，避免浮点累加漂移：

- 网格时刻为 ``t_k = anchor + k · interval_us``（``k`` 为整数）；
- 网格点恰与某原始样本同时刻时标 ``observed``，直接取样本值；
- 其余点仅在其前后两个样本的间隔 ``≤ max_gap_seconds`` 时补值——
  ``linear`` 线性插值，``previous`` 取前一样本；
- 前后样本间隔超过 ``max_gap_seconds`` 时 ``value=null``、``source=missing``，
  **禁止跨断档补值，也不向首末样本之外外推**。
"""

from __future__ import annotations

import calendar
import math
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
from typing import Literal

from .models import parse_timestamp

# 网格点数上限
MAX_RESAMPLE_POINTS = 10_000
_MICROSECONDS_PER_SECOND = 1_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def epoch_microseconds(dt: datetime) -> int:
    """aware datetime -> epoch 整数微秒（纯整数日历运算，无浮点精度损失）。"""
    return calendar.timegm(dt.utctimetuple()) * _MICROSECONDS_PER_SECOND + dt.microsecond


def whole_microseconds(seconds: float) -> int | None:
    """把秒数换算为整数微秒；不能表示为整微秒（含合理浮点容差）时返回 None。

    容差取乘法结果若干 ULP：例如 0.3 秒的二进制近似
    （300000.00000000006 微秒）仍视为整微秒，而 0.3000005 秒
    （300000.5 微秒）被拒绝。
    """
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    us = seconds * _MICROSECONDS_PER_SECOND
    nearest = round(us)
    if nearest <= 0:
        return None
    # 允许 4 个 ULP 的浮点表示 / 乘法误差（随数量级自适应）
    ulp = math.nextafter(us, math.inf) - us
    if abs(us - nearest) <= 4 * ulp:
        return nearest
    return None


def inclusive_gap_microseconds(seconds: float) -> int:
    """把 max_gap_seconds 换算为「间隔 ≤ 阈值」判定用的整数微秒上界。

    与 :func:`whole_microseconds` 不同，阈值不要求恰为整微秒：向最近整微秒
    取整并保留与 whole_microseconds 相同的 ULP 容差，使浮点传入的边界
    （如 0.1 秒）按语义精确比较（100000us 断档在 max_gap=0.1 时允许补值）。
    """
    return max(0, round(seconds * _MICROSECONDS_PER_SECOND))


def _ceildiv(a: int, b: int) -> int:
    """整除向上取整：ceil(a / b) = -((-a) // b) = -floor(-a/b)。"""
    return -((-a) // b)


def _format_us(t_us: int) -> str:
    """整数微秒 -> UTC ISO-8601（纯整数日历运算，与其他接口的归一化格式一致）。"""
    return (_EPOCH + timedelta(microseconds=t_us)).isoformat()


def resample_series(
    timestamps: list[str],
    values: list[float],
    *,
    interval_us: int,
    anchor_us: int,
    max_gap_seconds: float,
    method: Literal["linear", "previous"],
) -> dict:
    """在 anchor 对齐的等间隔微秒网格上重采样。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证）。
    ``max_gap_seconds`` 原样回显；内部比较换算为整数微秒上界
    （见 :func:`inclusive_gap_microseconds`，含等号语义）。
    """
    max_gap_us = inclusive_gap_microseconds(max_gap_seconds)
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    n_samples = len(sample_us)
    first_us, last_us = sample_us[0], sample_us[-1]

    # 位于 [first, last] 的网格下标范围：anchor + k·Δ ∈ [first, last]
    k_min = _ceildiv(first_us - anchor_us, interval_us)  # ceil((first - anchor)/Δ)
    k_max = (last_us - anchor_us) // interval_us  # floor((last - anchor)/Δ)
    if k_max < k_min:
        grid_us: list[int] = []
    else:
        count = k_max - k_min + 1
        if count > MAX_RESAMPLE_POINTS:
            raise ValueError(
                f"网格点数 {count} 超过上限 {MAX_RESAMPLE_POINTS}，"
                f"请增大 interval_seconds 或缩短样本时间范围"
            )
        grid_us = [anchor_us + k * interval_us for k in range(k_min, k_max + 1)]

    # 微秒时刻 -> 原始样本值
    observed = dict(zip(sample_us, values))

    points: list[dict] = []
    observed_count = interpolated_count = missing_count = 0

    # 当前连续缺口（missing run）的跟踪状态
    run_start_idx: int | None = None
    run_end_idx: int | None = None
    run_start_time: str | None = None
    run_end_time: str | None = None
    missing_ranges: list[dict] = []

    for idx, t_us in enumerate(grid_us):
        if t_us in observed:
            source = "observed"
            value: float | None = observed[t_us]
            observed_count += 1
        else:
            # 严格位于 t 左 / 右两侧的最近样本（网格在首末闭区间内，二者必存在）
            i_before = bisect_right(sample_us, t_us) - 1
            i_after = bisect_left(sample_us, t_us)
            t0, t1 = sample_us[i_before], sample_us[i_after]
            if t1 - t0 <= max_gap_us:
                if method == "linear":
                    value = values[i_before] + (values[i_after] - values[i_before]) * (
                        (t_us - t0) / (t1 - t0)
                    )
                else:  # previous
                    value = values[i_before]
                source = "interpolated"
                interpolated_count += 1
            else:
                value = None
                source = "missing"
                missing_count += 1

        iso_time = _format_us(t_us)
        points.append({"index": idx, "timestamp": iso_time, "value": value, "source": source})

        # 连续缺口：missing 连续出现即并入当前 run，observed / interpolated 断开
        if source == "missing":
            if run_start_idx is None:
                run_start_idx = run_end_idx = idx
                run_start_time = run_end_time = iso_time
            else:
                run_end_idx = idx
                run_end_time = iso_time
        elif run_start_idx is not None:
            missing_ranges.append(
                _close_missing_range(
                    sample_us, values, grid_us, run_start_idx, run_end_idx,
                    run_start_time, run_end_time, max_gap_seconds,
                )
            )
            run_start_idx = run_end_idx = None
            run_start_time = run_end_time = None

    if run_start_idx is not None:
        missing_ranges.append(
            _close_missing_range(
                sample_us, values, grid_us, run_start_idx, run_end_idx,
                run_start_time, run_end_time, max_gap_seconds,
            )
        )

    return {
        "anchor_time": _format_us(anchor_us),
        "interval_seconds": interval_us / _MICROSECONDS_PER_SECOND,
        "method": method,
        "max_gap_seconds": max_gap_seconds,
        "grid_start_time": _format_us(grid_us[0]) if grid_us else None,
        "grid_end_time": _format_us(grid_us[-1]) if grid_us else None,
        "point_count": len(grid_us),
        "observed_count": observed_count,
        "interpolated_count": interpolated_count,
        "missing_count": missing_count,
        "points": points,
        "missing_ranges": missing_ranges,
        "input_point_count": n_samples,
        "range_start_time": _format_us(first_us),
        "range_end_time": _format_us(last_us),
    }


def _close_missing_range(
    sample_us: list[int],
    values: list[float],
    grid_us: list[int],
    run_start_idx: int,
    run_end_idx: int,
    run_start_time: str,
    run_end_time: str,
    max_gap_seconds: float,
) -> dict:
    """收尾一个连续缺口：定位包围该 run 的前后样本与断档时长。"""
    t_start = grid_us[run_start_idx]
    t_end = grid_us[run_end_idx]
    i_before = bisect_right(sample_us, t_start) - 1
    i_after = bisect_left(sample_us, t_end)
    gap_us = sample_us[i_after] - sample_us[i_before]
    return {
        "start_index": run_start_idx,
        "end_index": run_end_idx,
        "start_time": run_start_time,
        "end_time": run_end_time,
        "point_count": run_end_idx - run_start_idx + 1,
        "before_time": _format_us(sample_us[i_before]),
        "after_time": _format_us(sample_us[i_after]),
        "sample_gap_seconds": gap_us / _MICROSECONDS_PER_SECOND,
        "max_gap_seconds": max_gap_seconds,
        "before_value": values[i_before],
        "after_value": values[i_after],
    }
