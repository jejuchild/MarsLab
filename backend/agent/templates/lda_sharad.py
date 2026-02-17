"""
Prebuilt workflow: LDA → SHARAD High-Res Subsurface Interface Analysis.

Finds Lobate Debris Aprons in a region, identifies SHARAD track
intersections, downloads data, and analyzes subsurface interfaces
for ice evidence.
"""

from __future__ import annotations

import uuid
from typing import Optional

from agent.schemas import WorkflowSession, WorkflowStep


def create_lda_sharad_workflow(
    region_name: str = "Arcadia Planitia",
    user_goal: Optional[str] = None,
    language: str = "en",
) -> WorkflowSession:
    """Create an LDA → SHARAD subsurface analysis workflow."""
    session_id = str(uuid.uuid4())

    if language == "ko":
        descriptions = [
            f"지역 검색: {region_name}",
            "SHARAD 고해상도 데이터 검색",
            "HiRISE DTM 데이터 검색",
            "SHARAD 트랙과 DTM 교차 분석",
            "SHARAD 지하 반사체 탐지",
            "분석 데이터 다운로드",
            "물리 기반 유전율 역추정",
            "분석 보고서 생성",
        ]
        narrative = (
            f"{region_name} 지역에서 SHARAD 고해상도 레이더 데이터와 HiRISE DTM을 "
            "교차 분석하여 지하 얼음 경계면을 탐지하고, 유전율을 역추정합니다."
        )
        approval_msg = "SHARAD 레이더그램 데이터를 다운로드하시겠습니까?"
    else:
        descriptions = [
            f"Resolve region: {region_name}",
            "Search for SHARAD high-res radargrams",
            "Search for HiRISE DTM coverage",
            "Find SHARAD track × DTM geometric intersections",
            "Detect SHARAD subsurface reflectors",
            "Download data for detailed analysis",
            "Run physics-based dielectric inversion",
            "Generate analysis report",
        ]
        narrative = (
            f"This workflow searches for SHARAD high-resolution radargrams and HiRISE DTMs "
            f"in {region_name}, finds geometric intersections, detects subsurface interfaces, "
            f"and runs physics-based dielectric inversion."
        )
        approval_msg = "Download SHARAD radargram data for analysis?"

    steps = [
        WorkflowStep(
            step_id="1", tool_name="resolve_region",
            description=descriptions[0],
            params={"query": region_name},
        ),
        WorkflowStep(
            step_id="2", tool_name="search_products",
            description=descriptions[1],
            params={"instrument": "SHARAD_HIGHRES"},
        ),
        WorkflowStep(
            step_id="3", tool_name="search_products",
            description=descriptions[2],
            params={"instrument": "HIRISE_DTM"},
        ),
        WorkflowStep(
            step_id="4", tool_name="find_sharad_intersections",
            description=descriptions[3],
            params={},
        ),
        WorkflowStep(
            step_id="5", tool_name="analyze_subsurface",
            description=descriptions[4],
            params={},
        ),
        WorkflowStep(
            step_id="6", tool_name="download_products",
            description=descriptions[5],
            params={"instruments": ["SHARAD_HIGHRES"]},
            requires_approval=True,
            approval_message=approval_msg,
        ),
        WorkflowStep(
            step_id="7", tool_name="run_sharad_inversion",
            description=descriptions[6],
            params={},
        ),
        WorkflowStep(
            step_id="8", tool_name="generate_report",
            description=descriptions[7],
            params={"title": f"LDA–SHARAD Analysis: {region_name}"},
        ),
    ]

    return WorkflowSession(
        session_id=session_id,
        user_goal=user_goal or f"Analyze subsurface interfaces beneath LDA features in {region_name}",
        language=language,
        agent_type="lda_sharad",
        steps=steps,
        plan_narrative=narrative,
    )
