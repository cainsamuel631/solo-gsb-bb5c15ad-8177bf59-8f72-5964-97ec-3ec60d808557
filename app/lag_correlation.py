"""双序列时滞关联扫描：候选时滞平移 + 一对一最近邻匹配 + Pearson 相关。

思路：
1. 输入两条按时间排序、各自无重复时间戳的序列：driver（驱动侧）与
   response（响应侧），以及一组候选时滞 ``lag_seconds``（秒，可为负）。
2. 对每个候选时滞 ``lag``，先把 driver 的时间轴整体平移：
   ``t' = t_driver + lag``（lag > 0 表示 driver 的变化在 lag 秒后才体现到
   response 上）；
3. 在平移后的时间轴上按时间顺序做**贪心一对一最近邻匹配**：
   依次处理每个平移后的 driver 点，在未匹配的 response 点中取时间距离
   最近且不超过 ``match_tolerance_seconds`` 的点配对；容差内距离相同
   （左右各一个等距点）时取**较早**的 response；已匹配的 response 点
   不得再被复用，匹配不上的 driver 点直接跳过；
4. 仅当候选的**配对数不少于 min_pairs** 且**两侧配对值方差都非零**时，
   才对配对序列 (driver_value, response_value) 计算 Pearson 相关系数，
   否则该候选标记为不可用并给出原因；
5. ``best_lag`` 在可用候选中依次按 |correlation| 最大、配对数更多、
   |lag| 更小、lag 更小选取；无可用候选时为 null。

匹配为贪心策略（时间顺序单次扫描），复杂度约
``O(候选数 × (n_driver + n_response) × 跳过已匹配点的均摊开销)``，
适合万级点数以内的离线 / 准实时分析；接口无状态，结果不写入 SQLite。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import Any

from .models import parse_timestamp


def _match_pairs(
    shifted_driver_times: list[float],
    response_times: list[float],
    tolerance: float,
) -> list[tuple[int, int]]:
    """贪心一对一最近邻匹配，返回 (driver_index, response_index) 配对列表。

    按平移后 driver 时间升序处理；每个 driver 点在**未使用**的 response
    点中取时间距离最近且不超过 ``tolerance`` 者；左右等距时取较早的
    response（左侧）。已配对的 response 点立即标记为已用，不再参与后续
    匹配。
    """
    used = [False] * len(response_times)
    pairs: list[tuple[int, int]] = []

    for driver_idx, t in enumerate(shifted_driver_times):
        pos = bisect_left(response_times, t)

        # 右侧（不早于 t）最近的未用 response 点
        right = pos
        while right < len(response_times) and used[right]:
            right += 1
        if right >= len(response_times) or response_times[right] - t > tolerance:
            right = -1  # 右侧无未用点，或最近的右侧点已超容差（更远的必然也超）

        # 左侧（早于 t）最近的未用 response 点
        left = pos - 1
        while left >= 0 and used[left]:
            left -= 1
        if left >= 0 and t - response_times[left] > tolerance:
            left = -1

        # 取距离更近者；距离相同取较早的 response（左侧）
        if left >= 0 and right >= 0:
            if t - response_times[left] <= response_times[right] - t:
                chosen = left
            else:
                chosen = right
        elif left >= 0:
            chosen = left
        elif right >= 0:
            chosen = right
        else:
            continue  # 容差内无可用 response 点，该 driver 点不配对

        used[chosen] = True
        pairs.append((driver_idx, chosen))

    return pairs


def _pearson(xs: list[float], ys: list[float]) -> tuple[float | None, str | None]:
    """Pearson 相关系数；任一侧方差为零时返回 (None, 原因)。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx == 0.0:
        return None, "driver 侧配对值方差为零，Pearson 相关未定义"
    if syy == 0.0:
        return None, "response 侧配对值方差为零，Pearson 相关未定义"
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy)
    # 浮点误差可能使 |r| 略微超过 1，截断回合法区间
    return max(-1.0, min(1.0, r)), None


def _select_best_lag(candidates: list[dict[str, Any]]) -> float | None:
    """在可用候选中选取 best_lag；无可用候选时返回 None。

    优先级依次：|correlation| 最大 → 配对数更多 → |lag| 更小 → lag 更小。
    """
    usable = [c for c in candidates if c["usable"]]
    if not usable:
        return None
    best = min(
        usable,
        key=lambda c: (
            -abs(c["correlation"]),
            -c["pair_count"],
            abs(c["lag_seconds"]),
            c["lag_seconds"],
        ),
    )
    return best["lag_seconds"]


def scan_lag_correlation(
    driver_timestamps: list[str],
    driver_values: list[float],
    response_timestamps: list[str],
    response_values: list[float],
    lag_candidates: list[float],
    match_tolerance_seconds: float,
    min_pairs: int = 3,
) -> dict[str, Any]:
    """执行时滞关联扫描并返回可 JSON 序列化的完整结果。

    :param driver_timestamps: 已排序、已去重的 driver 侧 ISO-8601 时间戳
    :param driver_values: 与 driver_timestamps 对齐的数值
    :param response_timestamps: 已排序、已去重的 response 侧 ISO-8601 时间戳
    :param response_values: 与 response_timestamps 对齐的数值
    :param lag_candidates: 候选时滞（秒，可为负）；对每个候选先把 driver
        时间平移 ``+lag`` 再做一对一最近邻匹配
    :param match_tolerance_seconds: 匹配容差（秒），平移后两侧时间差不超过
        该值才允许配对
    :param min_pairs: 计算 Pearson 相关所需的最少配对数
    """
    driver_times = [parse_timestamp(ts).timestamp() for ts in driver_timestamps]
    response_times = [parse_timestamp(ts).timestamp() for ts in response_timestamps]

    candidates: list[dict[str, Any]] = []
    for lag in lag_candidates:
        shifted = [t + lag for t in driver_times]
        pairs = _match_pairs(shifted, response_times, match_tolerance_seconds)
        pair_count = len(pairs)

        correlation: float | None = None
        usable = False
        reason: str | None = None
        if pair_count < min_pairs:
            reason = f"配对数 {pair_count} 少于 min_pairs={min_pairs}，未计算相关"
        else:
            xs = [driver_values[i] for i, _ in pairs]
            ys = [response_values[j] for _, j in pairs]
            correlation, reason = _pearson(xs, ys)
            usable = correlation is not None

        candidates.append(
            {
                "lag_seconds": lag,
                "pair_count": pair_count,
                "correlation": correlation,
                "usable": usable,
                "reason": reason,
            }
        )

    return {
        "driver_point_count": len(driver_values),
        "response_point_count": len(response_values),
        "match_tolerance_seconds": match_tolerance_seconds,
        "min_pairs": min_pairs,
        "candidate_count": len(candidates),
        "usable_candidate_count": sum(1 for c in candidates if c["usable"]),
        "candidates": candidates,
        "best_lag": _select_best_lag(candidates),
    }
