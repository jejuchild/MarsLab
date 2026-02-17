"""
Prebuilt workflow: Terraced Crater → ε (dielectric) Inversion.

Identifies concentric terraces in a crater using HiRISE DTM profiles,
finds SHARAD tracks crossing the crater, and computes the dielectric
constant (εr) from terrace depth + SHARAD two-way travel time.
"""

from __future__ import annotations

import uuid
from typing import Optional

from agent.schemas import WorkflowSession, WorkflowStep


def create_terrace_epsilon_workflow(
    region_name: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    user_goal: Optional[str] = None,
    language: str = "en",
) -> WorkflowSession:
    """Create a Terraced Crater → εr inversion workflow."""
    session_id = str(uuid.uuid4())

    # Build region query from name or coordinates
    if region_name:
        region_query = region_name
    elif lat is not None and lon is not None:
        region_query = f"{lat}, {lon}"
    else:
        region_query = "Arcadia Planitia"

    if language == "ko":
        descriptions = [
            f"지역 검색: {region_query}",
            "HiRISE DTM 데이터 검색",
            "SHARAD 고해상도 데이터 검색",
            "계단형 지형 탐지 (방사상 프로파일)",
            "SHARAD 트랙과 크레이터 교차 분석",
            "분석 데이터 다운로드",
            "유전율 (εr) 역추정",
            "분석 보고서 생성",
        ]
        narrative = (
            f"{region_query} 지역에서 HiRISE DTM으로 계단형 크레이터를 탐지하고, "
            "SHARAD 레이더 데이터와 교차 분석하여 유전율(εr)을 역추정합니다. "
            "계단 깊이와 SHARAD 왕복 시간으로부터 지하 물질의 유전 상수를 계산합니다."
        )
        approval_msg = "SHARAD 레이더그램 데이터를 다운로드하시겠습니까?"
    else:
        descriptions = [
            f"Resolve region: {region_query}",
            "Search for HiRISE DTM coverage",
            "Search for SHARAD high-res radargrams",
            "Detect concentric terraces (radial profiles)",
            "Find SHARAD track × crater intersections",
            "Download data for detailed analysis",
            "Run εr dielectric inversion",
            "Generate analysis report",
        ]
        narrative = (
            f"This workflow searches for HiRISE DTMs and SHARAD radargrams "
            f"near {region_query}, detects concentric terraces in crater walls, "
            f"identifies SHARAD tracks crossing the crater, and computes the "
            f"dielectric constant (εr) from terrace depth + radar travel time."
        )
        approval_msg = "Download SHARAD and HiRISE DTM data for analysis?"

    # Build step params — terrace detection needs coordinates if provided
    terrace_params: dict = {}
    if lat is not None:
        terrace_params["lat"] = lat
    if lon is not None:
        terrace_params["lon"] = lon

    steps = [
        WorkflowStep(
            step_id="1", tool_name="resolve_region",
            description=descriptions[0],
            params={"query": region_query},
        ),
        WorkflowStep(
            step_id="2", tool_name="search_products",
            description=descriptions[1],
            params={"instrument": "HIRISE_DTM"},
        ),
        WorkflowStep(
            step_id="3", tool_name="search_products",
            description=descriptions[2],
            params={"instrument": "SHARAD_HIGHRES"},
        ),
        WorkflowStep(
            step_id="4", tool_name="detect_terraces",
            description=descriptions[3],
            params=terrace_params,
        ),
        WorkflowStep(
            step_id="5", tool_name="find_sharad_intersections",
            description=descriptions[4],
            params={},
        ),
        WorkflowStep(
            step_id="6", tool_name="download_products",
            description=descriptions[5],
            params={"instruments": ["SHARAD_HIGHRES", "HIRISE_DTM"]},
            requires_approval=True,
            approval_message=approval_msg,
        ),
        WorkflowStep(
            step_id="7", tool_name="epsilon_inversion",
            description=descriptions[6],
            params={},
        ),
        WorkflowStep(
            step_id="8", tool_name="generate_report",
            description=descriptions[7],
            params={"title": f"Terraced Crater εr Inversion: {region_query}"},
        ),
    ]

    return WorkflowSession(
        session_id=session_id,
        user_goal=user_goal or f"Compute dielectric constant from terraced crater near {region_query}",
        language=language,
        agent_type="terrace_epsilon",
        steps=steps,
        plan_narrative=narrative,
    )
