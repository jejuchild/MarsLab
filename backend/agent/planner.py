"""
LLM-based workflow planner.

Generates a structured workflow plan from a natural-language goal.
Uses Ollama (llama3.1) first, falls back to Gemini Flash.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.intent_router import detect_language, route_intent
from agent.schemas import WorkflowSession, WorkflowStep
from agent.tool_registry import workflow_tools

logger = logging.getLogger(__name__)

# Cache Ollama availability to avoid repeated connection timeouts
_ollama_last_fail: float = 0.0
_OLLAMA_BACKOFF_S = 60.0  # Skip Ollama for 60s after a failure


class WorkflowPlanner:
    """Create workflow plans from natural-language goals."""

    async def create_plan(
        self,
        user_goal: str,
        language: Optional[str] = None,
        map_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowSession:
        """Generate a workflow session from a user goal.

        1. Detect language
        2. Route intent to template or generic planner
        3. For templates: instantiate directly
        4. For generic: call LLM to generate plan
        """
        if language is None:
            language = detect_language(user_goal)

        agent_type, region_hint = route_intent(user_goal)

        # Try template first
        if agent_type != "generic":
            session = self._try_template(agent_type, region_hint, user_goal, language)
            if session:
                # Inject map context
                if map_context:
                    session.context["map_viewport"] = map_context.get("viewport")
                    session.context["selected_features"] = map_context.get("selected_features", [])
                return session

        # Generic: use LLM planner
        session = await self._plan_with_llm(user_goal, language, map_context)
        return session

    def _try_template(
        self,
        agent_type: str,
        region_hint: Optional[str],
        user_goal: str,
        language: str,
    ) -> Optional[WorkflowSession]:
        """Try to instantiate a prebuilt workflow template."""
        try:
            if agent_type == "lda_sharad":
                from agent.templates.lda_sharad import create_lda_sharad_workflow
                return create_lda_sharad_workflow(
                    region_name=region_hint or "Arcadia Planitia",
                    user_goal=user_goal,
                    language=language,
                )
            elif agent_type == "terrace_epsilon":
                from agent.templates.terrace_epsilon import create_terrace_epsilon_workflow
                return create_terrace_epsilon_workflow(
                    region_name=region_hint,
                    user_goal=user_goal,
                    language=language,
                )
            elif agent_type == "ice_search":
                from agent.templates.ice_search import create_ice_search_workflow
                return create_ice_search_workflow(
                    region_name=region_hint or "Arcadia Planitia",
                    user_goal=user_goal,
                    language=language,
                )
        except ImportError as e:
            logger.warning(f"Template {agent_type} not available: {e}")
        return None

    async def _plan_with_llm(
        self,
        user_goal: str,
        language: str,
        map_context: Optional[Dict] = None,
    ) -> WorkflowSession:
        """Use LLM to generate a workflow plan.

        Fires Ollama and Gemini concurrently; uses whichever responds first.
        Falls back to a hardcoded default plan if both fail.
        """
        import asyncio

        tool_catalog = self._build_tool_catalog()
        prompt = self._build_prompt(user_goal, tool_catalog, language, map_context)

        # Race Ollama vs Gemini — use first successful response
        async def _race_llm() -> Optional[Dict]:
            tasks = [
                asyncio.create_task(self._try_ollama(prompt)),
                asyncio.create_task(self._try_gemini(prompt)),
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    # Cancel remaining tasks
                    for t in tasks:
                        t.cancel()
                    return result
            return None

        llm_response = await _race_llm()
        if llm_response is None:
            llm_response = self._default_plan(user_goal, language)

        steps = self._parse_steps(llm_response)
        narrative = llm_response.get("narrative", "")

        session = WorkflowSession(
            session_id=str(uuid.uuid4()),
            user_goal=user_goal,
            language=language,
            agent_type="generic",
            steps=steps,
            plan_narrative=narrative,
        )

        if map_context:
            session.context["map_viewport"] = map_context.get("viewport")
            session.context["selected_features"] = map_context.get("selected_features", [])

        return session

    def _build_tool_catalog(self) -> str:
        lines = []
        for spec in workflow_tools.get_all_specs():
            approval = " [REQUIRES APPROVAL]" if spec.requires_approval else ""
            lines.append(f"- {spec.name}: {spec.description}{approval}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        goal: str,
        tool_catalog: str,
        language: str,
        map_context: Optional[Dict] = None,
    ) -> str:
        lang_note = "Write step descriptions in Korean." if language == "ko" else ""
        viewport_note = ""
        if map_context and map_context.get("viewport"):
            vp = map_context["viewport"]
            viewport_note = f"\nCurrent map viewport: lat [{vp.get('south', '?')}, {vp.get('north', '?')}], lon [{vp.get('west', '?')}, {vp.get('east', '?')}]"

        return f"""You are a Mars science research workflow planner.
Given a user goal, create a step-by-step analysis plan using available tools.

USER GOAL: {goal}
{viewport_note}

AVAILABLE TOOLS:
{tool_catalog}

RULES:
1. Always start with resolve_region if a region is mentioned
2. Search for required instrument data before analysis
3. Mark download steps with requires_approval=true
4. Keep plans under 8 steps
5. End with generate_report
6. {lang_note}
7. Return ONLY valid JSON

OUTPUT FORMAT:
{{
  "narrative": "2-3 sentence plan summary",
  "steps": [
    {{
      "tool_name": "tool_name",
      "description": "What this step does",
      "params": {{}},
      "requires_approval": false
    }}
  ]
}}"""

    async def _try_ollama(self, prompt: str) -> Optional[Dict]:
        global _ollama_last_fail
        # Skip if Ollama failed recently (avoids repeated connect timeouts)
        if time.time() - _ollama_last_fail < _OLLAMA_BACKOFF_S:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=1.5)) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.1",
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return json.loads(data.get("response", "{}"))
        except Exception as e:
            _ollama_last_fail = time.time()
            logger.info(f"Ollama planner failed: {e}")
        return None

    async def _try_gemini(self, prompt: str) -> Optional[Dict]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"responseMimeType": "application/json"},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text)
        except Exception as e:
            logger.info(f"Gemini planner failed: {e}")
        return None

    def _default_plan(self, goal: str, language: str) -> Dict:
        """Fallback when both LLMs are unavailable."""
        return {
            "narrative": f"Analyzing: {goal}" if language == "en" else f"분석 중: {goal}",
            "steps": [
                {"tool_name": "resolve_region", "description": "Resolve target region", "params": {"query": goal}},
                {"tool_name": "search_products", "description": "Search SHARAD data", "params": {"instrument": "SHARAD_HIGHRES"}},
                {"tool_name": "search_products", "description": "Search CRISM data", "params": {"instrument": "CRISM"}},
                {"tool_name": "analyze_subsurface", "description": "Analyze subsurface", "params": {}},
                {"tool_name": "generate_report", "description": "Generate report", "params": {}},
            ],
        }

    def _parse_steps(self, llm_response: Dict) -> List[WorkflowStep]:
        """Convert LLM JSON response to WorkflowStep objects."""
        steps = []
        raw_steps = llm_response.get("steps", [])

        for i, raw in enumerate(raw_steps):
            tool_name = raw.get("tool_name", "")
            # Validate tool exists
            spec = workflow_tools.get_spec(tool_name)
            if spec is None:
                logger.warning(f"Planner suggested unknown tool: {tool_name}, skipping")
                continue

            # Determine approval requirement
            requires_approval = raw.get("requires_approval", spec.requires_approval)
            approval_message = None
            if requires_approval and spec.approval_template:
                try:
                    approval_message = spec.approval_template.format(**raw.get("params", {}))
                except (KeyError, IndexError):
                    approval_message = f"Approve: {raw.get('description', tool_name)}?"
            elif requires_approval:
                approval_message = f"Approve: {raw.get('description', tool_name)}?"

            steps.append(WorkflowStep(
                step_id=str(i + 1),
                tool_name=tool_name,
                description=raw.get("description", tool_name),
                params=raw.get("params", {}),
                requires_approval=requires_approval,
                approval_message=approval_message,
            ))

        return steps


# Singleton
planner = WorkflowPlanner()
