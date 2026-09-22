"""事件聚合：把多条已保存检测的异常段按时间邻近关系聚合为事件。

规则：
- 取请求所列检测的全部异常段，统一按（起始时间, detection_id, 段序号）排序；
- 顺序扫描，只要下一段的开始时间不晚于当前事件结束时间 + ``gap_seconds``
  （即时间重叠、首尾相接或两段间隔不超过 ``gap_seconds``）即并入同一事件，
  因此同一检测的相邻异常段同样会被合并；
- 事件输出起止时间、覆盖的 series_id / detection_id、成员段数、成员中的
  最大绝对 z 分数；成员定位按（起始时间, detection_id, 段序号）稳定排序，
  事件列表按事件起始时间稳定排序。

本模块为只读计算，不触碰检测记录。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .models import parse_timestamp


def _member_sort_key(member: dict[str, Any]) -> tuple[str, str, int]:
    return member["start_time"], member["detection_id"], member["segment_index"]


def _extract_members(
    detection_id: str,
    series_id: str | None,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """从一条已保存检测结果中抽取可参与聚合的异常段成员。"""
    members: list[dict[str, Any]] = []
    for idx, seg in enumerate(result.get("segments", [])):
        members.append(
            {
                "detection_id": detection_id,
                "series_id": series_id,
                "segment_index": idx,
                "start_index": seg["start_index"],
                "end_index": seg["end_index"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "point_count": seg["point_count"],
                "anomaly_point_count": seg["anomaly_point_count"],
                "max_abs_zscore": float(seg["score"]["max_abs_zscore"]),
            }
        )
    return members


def aggregate_incidents(
    records: list[dict[str, Any]],
    gap_seconds: float,
) -> dict[str, Any]:
    """把多条检测记录的异常段聚合为事件。

    :param records: database.get_detection 返回的记录列表（顺序与请求一致）
    :param gap_seconds: 允许桥接的最大时间间隔（秒，非负）；
        下一段开始时间 ≤ 当前事件结束时间 + gap_seconds 即并入同一事件
    """
    members: list[dict[str, Any]] = []
    for record in records:
        members.extend(
            _extract_members(record["id"], record.get("series_id"), record["result"])
        )
    members.sort(key=_member_sort_key)

    incidents: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_end: str | None = None
    bridge_limit: str | None = None  # 当前事件结束时间 + gap_seconds

    for member in members:
        if not current or member["start_time"] <= bridge_limit:
            # 与当前事件重叠或间隔不超过 gap_seconds（首个成员直接开启事件）
            current.append(member)
            if current_end is None or member["end_time"] > current_end:
                current_end = member["end_time"]
                bridge_limit = _shift_iso(current_end, gap_seconds)
        else:
            incidents.append(_build_incident(current))
            current = [member]
            current_end = member["end_time"]
            bridge_limit = _shift_iso(current_end, gap_seconds)
    if current:
        incidents.append(_build_incident(current))

    return {
        "gap_seconds": gap_seconds,
        "detection_count": len(records),
        "segment_count": len(members),
        "incident_count": len(incidents),
        "incidents": incidents,
    }


def _shift_iso(iso_time: str | None, gap_seconds: float) -> str:
    """事件结束时间加上 gap_seconds，返回 ISO-8601 字符串用于时间比较。

    所有入库时间均为带 UTC 偏移的 ISO-8601，解析后做秒级平移；
    gap_seconds 为 0 时直接返回原字符串（零成本的重叠/相接判断）。
    """
    assert iso_time is not None
    if gap_seconds == 0:
        return iso_time
    return (parse_timestamp(iso_time) + timedelta(seconds=gap_seconds)).isoformat()


def _build_incident(members: list[dict[str, Any]]) -> dict[str, Any]:
    """构造单个事件：成员已按 _member_sort_key 排好序。"""
    start_time = members[0]["start_time"]
    end_time = max(m["end_time"] for m in members)
    detection_ids = sorted({m["detection_id"] for m in members})
    series_ids = sorted({m["series_id"] for m in members if m["series_id"] is not None})
    return {
        "start_time": start_time,
        "end_time": end_time,
        "series_ids": series_ids,
        "detection_ids": detection_ids,
        "segment_count": len(members),
        "max_abs_zscore": max(m["max_abs_zscore"] for m in members),
        "members": members,
    }
