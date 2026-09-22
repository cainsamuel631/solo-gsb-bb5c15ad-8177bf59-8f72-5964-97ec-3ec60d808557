"""无状态等间隔重采样：anchor 对齐网格 + 线性 / 前值填充 + 断档缺失标记。

思路：
1. 输入序列按时间排序、无重复时间戳。所有时间一律换算为**整数微秒**
   （epoch microseconds）后再参与运算，避免浮点时间轴带来的对齐误差；
2. 网格时刻 ``g_k = anchor_time + k · interval``（interval 必须恰好是整数
   微秒，否则返回 422），只保留落在**首末样本闭区间**
   ``[first_time, last_time]`` 内的网格点，最多 10000 个（超过返回 422），
   因此天然不存在跨断档外推——网格两侧永远有样本，或网格点本身就是样本；
3. 每个网格点按来源分三类：
   - ``observed``：网格时刻与某个原始样本时刻（整数微秒）完全相等，
     直接采用样本值；
   - ``interpolated``：非观测点，但夹住它的前后两个相邻样本的时间间隔
     ``gap`` **不超过** ``max_gap_seconds`` 时，按 ``method`` 补值——
     ``linear`` 为按时间比例的线性插值，``previous`` 为前值（阶梯）填充；
   - ``missing``：前后样本间隔超过 ``max_gap_seconds``（采样断档），
     ``value`` 返回 ``null``，绝不跨断档插值或外推；
4. 顺序扫描网格，把连续的 ``missing`` 点汇总为缺口范围（连续缺段）。

整体为单次线性扫描（样本定位用 bisect 均摊 ``O(log n)``），
适合万级网格点以内的离线 / 准实时分析；接口无状态，结果不写入 SQLite。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import parse_timestamp

# 网格点数硬上限
MAX_GRID_POINTS = 10_000
# 微秒级整点判定的容差（秒）：远小于 1 微秒
_MICRO_EPS = 1e-9

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)


def epoch_us(dt: datetime) -> int:
    """把 aware datetime 换算为整数 epoch 微秒（纯整数运算）。"""
    return (dt - _EPOCH) // _MICROSECOND


def epoch_us_to_iso(value_us: int) -> str:
    """整数 epoch 微秒转回 UTC ISO-8601 字符串。"""
    seconds, micros = divmod(value_us, 1_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=micros
    ).isoformat()


def interval_to_microseconds(interval_seconds: float) -> int:
    """把秒级间隔换算为整数微秒；不能恰好表示为整数微秒时抛出 ValueError。"""
    us = int(round(interval_seconds * 1_000_000))
    if not math.isclose(
        us / 1_000_000, interval_seconds, rel_tol=0.0, abs_tol=_MICRO_EPS
    ):
        raise ValueError(
            f"interval_seconds 必须能表示为整数微秒，收到 {interval_seconds!r} 秒"
        )
    return us


def resample(
    timestamps: list[str],
    values: list[float],
    anchor_time: object,
    interval_seconds: float,
    method: str,
    max_gap_seconds: float,
) -> dict[str, Any]:
    """执行无状态重采样并返回可 JSON 序列化的完整结果。

    :param timestamps: 已排序、已去重的 ISO-8601 时间戳
    :param values: 与 timestamps 对齐的数值
    :param anchor_time: 网格锚点时间（ISO-8601 字符串或 epoch 秒/毫秒数字）
    :param interval_seconds: 网格间隔（秒），调用方需保证为正且有限
    :param method: 补值方法：``linear``（线性插值）或 ``previous``（前值填充）
    :param max_gap_seconds: 允许补值的最大前后样本间隔（秒，非负）：
        网格点两侧相邻样本间隔不超过该值才补值，否则标记 missing
    """
    sample_us = [epoch_us(parse_timestamp(ts)) for ts in timestamps]
    value_by_us = dict(zip(sample_us, values))
    anchor_us = epoch_us(parse_timestamp(anchor_time))
    interval_us = interval_to_microseconds(interval_seconds)
    max_gap_us = int(round(max_gap_seconds * 1_000_000))

    first_us, last_us = sample_us[0], sample_us[-1]

    # 首末样本闭区间内、与 anchor 对齐的网格下标范围
    k_start = math.ceil((first_us - anchor_us) / interval_us)
    k_end = math.floor((last_us - anchor_us) / interval_us)

    grid_count = k_end - k_start + 1 if k_end >= k_start else 0
    if grid_count > MAX_GRID_POINTS:
        raise ValueError(
            f"网格点数 {grid_count} 超过上限 {MAX_GRID_POINTS}，"
            f"请增大 interval_seconds 或缩小首末样本时间范围"
        )

    points: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    observed_count = 0
    interpolated_count = 0
    missing_count = 0

    # 连续 missing 缺口的扫描状态（记录网格下标与整数微秒时刻）
    missing_start_idx: int | None = None
    missing_start_us = 0
    missing_end_us = 0

    def close_gap(end_idx: int, end_us: int) -> None:
        """关闭一段连续缺口并写入 gaps。"""
        assert missing_start_idx is not None
        gaps.append(
            {
                "start_index": missing_start_idx,
                "end_index": end_idx,
                "start_time": epoch_us_to_iso(missing_start_us),
                "end_time": epoch_us_to_iso(end_us),
                "point_count": end_idx - missing_start_idx + 1,
            }
        )

    for k in range(k_start, k_end + 1):
        grid_us = anchor_us + k * interval_us
        grid_idx = len(points)

        # 原始样本恰落网格：observed
        if grid_us in value_by_us:
            if missing_start_idx is not None:
                close_gap(grid_idx - 1, missing_end_us)
                missing_start_idx = None
            points.append(
                {
                    "timestamp": epoch_us_to_iso(grid_us),
                    "value": value_by_us[grid_us],
                    "source": "observed",
                }
            )
            observed_count += 1
            continue

        # 夹住网格点的前后相邻样本（网格在首末闭区间内，必同时存在两侧）
        right_idx = bisect_left(sample_us, grid_us)
        left_idx = right_idx - 1
        left_us, right_us = sample_us[left_idx], sample_us[right_idx]
        gap_us = right_us - left_us

        if gap_us > max_gap_us:
            # 采样断档：不跨档插值，也不外推
            if missing_start_idx is None:
                missing_start_idx = grid_idx
                missing_start_us = grid_us
            missing_end_us = grid_us
            points.append(
                {"timestamp": epoch_us_to_iso(grid_us), "value": None, "source": "missing"}
            )
            missing_count += 1
            continue

        if missing_start_idx is not None:
            close_gap(grid_idx - 1, missing_end_us)
            missing_start_idx = None

        if method == "linear":
            ratio = (grid_us - left_us) / gap_us
            value = values[left_idx] + (values[right_idx] - values[left_idx]) * ratio
        else:  # previous
            value = values[left_idx]

        points.append(
            {
                "timestamp": epoch_us_to_iso(grid_us),
                "value": value,
                "source": "interpolated",
            }
        )
        interpolated_count += 1

    if missing_start_idx is not None:
        close_gap(len(points) - 1, missing_end_us)

    return {
        "anchor_time": epoch_us_to_iso(anchor_us),
        "interval_seconds": interval_seconds,
        "method": method,
        "max_gap_seconds": max_gap_seconds,
        "sample_point_count": len(values),
        "grid_point_count": len(points),
        "observed_count": observed_count,
        "interpolated_count": interpolated_count,
        "missing_count": missing_count,
        "points": points,
        "missing_gaps": gaps,
    }
