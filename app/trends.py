"""滑动窗口趋势扫描：Mann-Kendall 检验 + Theil-Sen 斜率。

思路：
1. 以 ``window_size`` / ``step_size`` 在按时间排序的序列上滑动窗口；
2. 每个窗口执行 Mann-Kendall 单调趋势检验：
   - 统计量 ``S = Σ sign(x_j − x_i)``（i < j）；
   - **重复值方差修正**：``Var(S) = [n(n−1)(2n+5) − Σ t(t−1)(2t+5)] / 18``，
     其中 t 为每个并列值组的重复次数；
   - **连续性校正**：S>0 时 Z=(S−1)/√Var(S)，S<0 时 Z=(S+1)/√Var(S)，S=0 时 Z=0；
   - 双侧 p 值取正态近似 ``p = erfc(|Z|/√2)``；
3. 同一窗口按**实际时间差**（秒）计算 Theil-Sen 斜率：
   所有点对斜率 ``(x_j − x_i)/(t_j − t_i)`` 的中位数，非等距采样同样适用；
4. 仅当 ``p_value ≤ alpha`` 且 ``|slope| ≥ min_slope``（且 slope ≠ 0）时，
   把窗口标记为显著的上升 / 下降趋势，否则视为平稳；
5. 方向相同且时间范围重叠或首尾相接的显著窗口合并为趋势区间，
   输出区间范围、成员窗口、最小 p 值与成员斜率中位数。

复杂度 ``O(窗口数 × window_size²)``，建议 window_size 控制在数百以内。
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any

from .models import parse_timestamp


def _mann_kendall(values: list[float]) -> tuple[int, float, float, float]:
    """Mann-Kendall 检验，返回 (S, Var(S), Z, 双侧 p 值)。

    含重复值方差修正与连续性校正；窗口内值全部相同（Var(S)=0）时
    无趋势可言，返回 Z=0、p=1。
    """
    n = len(values)
    s = 0
    for i in range(n - 1):
        xi = values[i]
        for j in range(i + 1, n):
            d = values[j] - xi
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1

    # 重复值（并列组）方差修正：Σ t(t−1)(2t+5)，t 为各并列组大小
    tie_term = sum(
        t * (t - 1) * (2 * t + 5) for t in Counter(values).values() if t > 1
    )
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0.0:
        return s, 0.0, 0.0, 1.0

    # 连续性校正
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2.0))  # 双侧正态近似
    return s, var_s, z, p_value


def _theil_sen(times: list[float], values: list[float]) -> float:
    """Theil-Sen 斜率：所有点对斜率的中位数（单位：值/秒，按实际时间差）。"""
    n = len(values)
    slopes = []
    for i in range(n - 1):
        xi, ti = values[i], times[i]
        for j in range(i + 1, n):
            slopes.append((values[j] - xi) / (times[j] - ti))
    return median(slopes)


def scan_trends(
    timestamps: list[str],
    values: list[float],
    window_size: int = 10,
    step_size: int = 5,
    alpha: float = 0.05,
    min_slope: float = 0.0,
) -> dict[str, Any]:
    """执行趋势扫描并返回可 JSON 序列化的完整结果。

    :param timestamps: 已排序、已去重的 ISO-8601 时间戳字符串
    :param values: 与 timestamps 对齐的数值
    :param window_size: 滑动窗口大小（点数，≥3）
    :param step_size: 滑动步长（点数，≥1）；等于 window_size 时窗口首尾相接，
        大于 window_size 时窗口之间留有空档（空档窗口不会合并）
    :param alpha: Mann-Kendall 显著性水平，p_value ≤ alpha 才可能判为趋势
    :param min_slope: 最小绝对斜率（值/秒），|slope| 达到该值才判为趋势
    """
    n = len(values)
    if n < window_size:
        raise ValueError(
            f"点数（{n}）必须不少于 window_size（{window_size}），否则无窗可滑"
        )

    times = [parse_timestamp(ts).timestamp() for ts in timestamps]

    windows: list[dict[str, Any]] = []
    start = 0
    window_index = 0
    while start + window_size <= n:
        end = start + window_size - 1
        w_values = values[start : end + 1]
        w_times = times[start : end + 1]

        s, var_s, z, p_value = _mann_kendall(w_values)
        slope = _theil_sen(w_times, w_values)

        # 显著趋势：p 值达标且绝对斜率达标；slope 恰为 0 恒判平稳
        significant = p_value <= alpha and abs(slope) >= min_slope and slope != 0.0
        if significant:
            direction = "上升" if slope > 0 else "下降"
        else:
            direction = "平稳"

        windows.append(
            {
                "window_index": window_index,
                "start_index": start,
                "end_index": end,
                "start_time": timestamps[start],
                "end_time": timestamps[end],
                "point_count": window_size,
                "s": s,
                "var_s": var_s,
                "z": z,
                "p_value": p_value,
                "slope": slope,
                "significant": significant,
                "direction": direction,
            }
        )
        window_index += 1
        start += step_size

    segments = _merge_trend_segments(windows, timestamps)
    return {
        "point_count": n,
        "window_size": window_size,
        "step_size": step_size,
        "alpha": alpha,
        "min_slope": min_slope,
        "window_count": len(windows),
        "significant_window_count": sum(1 for w in windows if w["significant"]),
        "trend_segment_count": len(segments),
        "segments": segments,
        "windows": windows,
    }


def _merge_trend_segments(
    windows: list[dict[str, Any]], timestamps: list[str]
) -> list[dict[str, Any]]:
    """合并显著窗口：方向相同且时间范围重叠或首尾相接的窗口归为一段。

    窗口按 start_index 升序生成；后一窗口起点不超过当前区间终点 +1
    （索引相邻即时间范围首尾相接）且方向相同即并入当前区间，
    否则结算当前区间并开启新区间。非显著窗口直接跳过。
    """
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for w in windows:
        if not w["significant"]:
            continue
        if (
            current is not None
            and w["direction"] == current["direction"]
            and w["start_index"] <= current["end_index"] + 1
        ):
            current["windows"].append(w)
            current["end_index"] = w["end_index"]
        else:
            if current is not None:
                segments.append(_finalize_segment(current, timestamps))
            current = {
                "direction": w["direction"],
                "start_index": w["start_index"],
                "end_index": w["end_index"],
                "windows": [w],
            }
    if current is not None:
        segments.append(_finalize_segment(current, timestamps))
    return segments


def _finalize_segment(
    segment: dict[str, Any], timestamps: list[str]
) -> dict[str, Any]:
    """结算一个趋势区间：范围、成员窗口、最小 p 值与成员斜率中位数。"""
    members = segment["windows"]
    return {
        "direction": segment["direction"],
        "start_index": segment["start_index"],
        "end_index": segment["end_index"],
        "start_time": timestamps[segment["start_index"]],
        "end_time": timestamps[segment["end_index"]],
        "point_count": segment["end_index"] - segment["start_index"] + 1,
        "window_count": len(members),
        "min_p_value": min(w["p_value"] for w in members),
        "median_slope": median(w["slope"] for w in members),
        "windows": members,
    }
