"""
Prebuilt workflow: Multi-Criteria Ice Evidence Search.

Searches a region for CRISM mineral ice signatures, SHARAD subsurface
reflectors, and CNN mineral classifications, then fuses all evidence
into an ice probability score.
"""

from __future__ import annotations

import uuid
from typing import Optional

from agent.schemas import WorkflowSession, WorkflowStep


def create_ice_search_workflow(
    region_name: str = "Arcadia Planitia",
    user_goal: Optional[str] = None,
    language: str = "en",
) -> WorkflowSession:
    """Create a multi-criteria ice evidence search workflow."""
    session_id = str(uuid.uuid4())

    if language == "ko":
        descriptions = [
            f"지역 검색: {region_name}",
            "CRISM 분광 데이터 검색",
            "SHARAD 고해상도 데이터 검색",
            "CRISM 얼음/수화 광물 분석",
            "SHARAD 지하 반사체 탐지",
            "SHARAD × CRISM 교차 분석",
            "얼음 위치에서 SHARAD 반사체 집중 분석",
            "다중 기준 얼음 증거 종합",
            "분석 보고서 생성",
        ]
        narrative = (
            f"{region_name} 지역에서 CRISM 분광 데이터와 SHARAD 레이더 데이터를 검색하고, "
            "광물 분석과 지하 반사체 탐지를 수행한 후, 다중 기준 증거를 종합하여 "
            "얼음 존재 확률을 산출합니다."
        )
    else:
        descriptions = [
            f"Resolve region: {region_name}",
            "Search for CRISM spectral observations",
            "Search for SHARAD high-res radargrams",
            "Analyze CRISM ice/hydration minerals",
            "Detect SHARAD subsurface reflectors",
            "Find SHARAD × CRISM geometric intersections",
            "Targeted subsurface analysis at ice locations",
            "Fuse multi-criteria ice evidence",
            "Generate analysis report",
        ]
        narrative = (
            f"This workflow searches {region_name} for CRISM spectral data and SHARAD "
            f"radargrams, analyzes mineral ice signatures and subsurface reflectors, "
            f"finds co-located detections, and fuses all evidence into an ice probability score."
        )

    steps = [
        WorkflowStep(
            step_id="1", tool_name="resolve_region",
            description=descriptions[0],
            params={"query": region_name},
        ),
        WorkflowStep(
            step_id="2", tool_name="search_products",
            description=descriptions[1],
            params={"instrument": "CRISM"},
        ),
        WorkflowStep(
            step_id="3", tool_name="search_products",
            description=descriptions[2],
            params={"instrument": "SHARAD_HIGHRES"},
        ),
        WorkflowStep(
            step_id="4", tool_name="analyze_minerals",
            description=descriptions[3],
            params={},
        ),
        WorkflowStep(
            step_id="5", tool_name="analyze_subsurface",
            description=descriptions[4],
            params={},
        ),
        WorkflowStep(
            step_id="6", tool_name="find_sharad_intersections",
            description=descriptions[5],
            params={},
        ),
        WorkflowStep(
            step_id="7", tool_name="targeted_subsurface_at_ice",
            description=descriptions[6],
            params={},
        ),
        WorkflowStep(
            step_id="8", tool_name="evaluate_ice_evidence",
            description=descriptions[7],
            params={},
        ),
        WorkflowStep(
            step_id="9", tool_name="generate_report",
            description=descriptions[8],
            params={"title": f"Ice Evidence Search: {region_name}"},
        ),
    ]

    return WorkflowSession(
        session_id=session_id,
        user_goal=user_goal or f"Search for ice evidence in {region_name}",
        language=language,
        agent_type="ice_search",
        steps=steps,
        plan_narrative=narrative,
    )
