from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config import AgentConfig, CLASS_ORDER
from .prompts.react_template import REACT_TEMPLATE, format_react_prompt
from .prompts.system import SYSTEM_PROMPT
from .tools.analyze_mola import AnalyzeMOLATool
from .tools.classify import ClassifyTool
from .tools.query_rag import QueryRAGTool
from .tools.regional_context import RegionalContextTool
from .tools.zoom_tile import ZoomTileTool

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentResult:
    landform_class: str
    confidence: float
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    num_steps: int = 0
    mode: str = "fast"
    error: Optional[str] = None

    @property
    def reasoning_chain_json(self) -> str:
        return json.dumps(self.reasoning_chain, ensure_ascii=True, indent=2)


class BaseVLM:
    def generate(self, prompt: str, system_prompt: str) -> str:
        raise NotImplementedError

    async def agenerate(self, prompt: str, system_prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt, system_prompt)


class MockVLM(BaseVLM):
    """Deterministic mock for offline testing."""

    _CLASS_RE = re.compile(r'"class"\s*:\s*"([A-Z_]+)"')
    _CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')

    def generate(self, prompt: str, system_prompt: str) -> str:
        detected_class = "BACKGROUND"
        confidence = 0.55

        class_match = self._CLASS_RE.search(prompt)
        if class_match:
            candidate = class_match.group(1).strip()
            if candidate in CLASS_ORDER:
                detected_class = candidate

        conf_match = self._CONF_RE.search(prompt)
        if conf_match:
            confidence = max(0.01, min(0.99, float(conf_match.group(1))))

        improved_conf = max(confidence, 0.72)
        return (
            "Thought: The classifier and contextual observations are sufficient to decide.\n"
            "Final Answer:\n"
            f"Class: {detected_class}\n"
            f"Confidence: {improved_conf:.2f}\n"
            "Reasoning: Combined classifier evidence and terrain context support this label."
        )


class ClaudeVLM(BaseVLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, system_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing for ClaudeVLM.")

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parts: List[str] = []
            for block in response.content:
                if getattr(block, "type", "") == "text":
                    text_chunk = getattr(block, "text", "")
                    if isinstance(text_chunk, str):
                        parts.append(text_chunk)
            if parts:
                return "\n".join(parts)
        except ImportError:
            pass

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        chunks = [item.get("text", "") for item in data.get("content", []) if item.get("type") == "text"]
        return "\n".join([chunk for chunk in chunks if chunk]).strip()


class LocalTransformerVLM(BaseVLM):
    """Placeholder local VLM adapter. Swap with project-specific implementation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str) -> str:
        return (
            "Thought: Local VLM placeholder active; using available classifier evidence.\n"
            "Final Answer:\n"
            "Class: BACKGROUND\n"
            "Confidence: 0.50\n"
            "Reasoning: Replace LocalTransformerVLM with a deployed local model for full reasoning quality."
        )


class MarsLandformAgent:
    def __init__(
        self,
        config: AgentConfig | Any,
        classifier: Any,
        rag: Any,
        mola_features: Dict[str, Dict[str, Any]],
        metadata: Any,
        vlm: Optional[BaseVLM] = None,
    ) -> None:
        self.config = getattr(config, "agent", config)
        if not isinstance(self.config, AgentConfig):
            self.config = AgentConfig(**asdict(self.config)) if hasattr(self.config, "__dataclass_fields__") else AgentConfig()

        self.tools = {
            "classify": ClassifyTool(classifier),
            "query_rag": QueryRAGTool(rag),
            "analyze_mola": AnalyzeMOLATool(mola_features),
            "zoom_tile": ZoomTileTool(metadata),
            "regional_context": RegionalContextTool(metadata, classifier),
        }
        self.vlm = vlm or self._build_default_vlm(self.config)

    def _build_default_vlm(self, agent_cfg: AgentConfig) -> BaseVLM:
        model_name = str(agent_cfg.vlm_model or "").lower()
        if "claude" in model_name:
            return ClaudeVLM(
                model=agent_cfg.vlm_model,
                temperature=agent_cfg.vlm_temperature,
                max_tokens=agent_cfg.vlm_max_tokens,
            )
        if model_name:
            return LocalTransformerVLM(model_name=agent_cfg.vlm_model)
        return MockVLM()

    async def classify_image(self, image_id: str) -> AgentResult:
        reasoning_chain: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        initial = self._safe_tool_call("classify", {"image_id": image_id})
        tools_used.append("classify")
        reasoning_chain.append({"step": 0, "action": "classify", "action_input": {"image_id": image_id}, "observation": initial})

        initial_class = str(initial.get("class", "BACKGROUND"))
        initial_conf = float(initial.get("confidence", 0.0) or 0.0)

        needs_agent_loop = (
            self.config.mode == "agent"
            or initial_conf < float(self.config.confidence_threshold)
            or bool(initial.get("error"))
        )

        if not needs_agent_loop:
            return AgentResult(
                landform_class=initial_class,
                confidence=initial_conf,
                reasoning_chain=reasoning_chain,
                tools_used=tools_used,
                num_steps=0,
                mode="fast",
                error=initial.get("error"),
            )

        scratchpad: List[str] = [f"Observation: {json.dumps(initial, ensure_ascii=True)}"]
        for step in range(1, int(self.config.max_steps) + 1):
            prompt = format_react_prompt(
                image_id=image_id,
                tool_names=list(self.tools.keys()),
                react_template=REACT_TEMPLATE,
                scratchpad="\n".join(scratchpad),
            )

            try:
                vlm_response = await self._invoke_vlm(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            except Exception as exc:
                msg = f"VLM call failed at step {step}: {exc}"
                LOGGER.exception(msg)
                reasoning_chain.append({"step": step, "error": msg})
                break

            parsed = self._parse_react_response(vlm_response)
            reasoning_chain.append({"step": step, "vlm_response": vlm_response, "parsed": parsed})

            if parsed.get("final"):
                final_class, final_conf = self._extract_final_answer(parsed["final"], fallback_class=initial_class, fallback_conf=initial_conf)
                return AgentResult(
                    landform_class=final_class,
                    confidence=final_conf,
                    reasoning_chain=reasoning_chain,
                    tools_used=tools_used,
                    num_steps=step,
                    mode="agent",
                )

            action = parsed.get("action")
            action_input = parsed.get("action_input", {})
            if action not in self.tools:
                observation = {
                    "error": f"Unknown action '{action}'. Available tools: {sorted(self.tools.keys())}",
                    "requested_action": action,
                }
            else:
                observation = self._safe_tool_call(action, action_input)
                tools_used.append(action)

            scratchpad.append(f"Thought: {parsed.get('thought', '').strip()}")
            scratchpad.append(f"Action: {action}")
            scratchpad.append(f"Action Input: {json.dumps(action_input, ensure_ascii=True)}")
            scratchpad.append(f"Observation: {json.dumps(observation, ensure_ascii=True)}")

            reasoning_chain.append(
                {
                    "step": step,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
            )

        final_class = initial_class if initial_class in CLASS_ORDER else "BACKGROUND"
        final_conf = max(initial_conf, 0.35)
        reasoning_chain.append(
            {
                "step": int(self.config.max_steps),
                "forced_final": True,
                "class": final_class,
                "confidence": final_conf,
            }
        )
        return AgentResult(
            landform_class=final_class,
            confidence=final_conf,
            reasoning_chain=reasoning_chain,
            tools_used=tools_used,
            num_steps=int(self.config.max_steps),
            mode="agent",
            error="Reached max_steps without final answer.",
        )

    async def _invoke_vlm(self, prompt: str, system_prompt: str) -> str:
        if hasattr(self.vlm, "agenerate"):
            response = self.vlm.agenerate(prompt, system_prompt)
            if inspect.isawaitable(response):
                return str(await response)
            return str(response)

        response = self.vlm.generate(prompt, system_prompt)
        if inspect.isawaitable(response):
            return str(await response)
        return str(response)

    def _safe_tool_call(self, tool_name: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools[tool_name]
        action_input = action_input or {}

        try:
            if hasattr(tool, "run"):
                output = tool.run(**action_input)
            elif callable(tool):
                output = tool(**action_input)
            else:
                raise RuntimeError(f"Tool {tool_name} is not callable.")

            if inspect.isawaitable(output):
                async def _await_output(awaitable: Any) -> Any:
                    return await awaitable

                output = asyncio.run(_await_output(output))

            if isinstance(output, dict):
                return output
            return {"result": output}
        except Exception as exc:
            LOGGER.exception("Tool call failed for %s", tool_name)
            return {"error": str(exc), "tool": tool_name, "input": action_input}

    def _parse_react_response(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        final_match = re.search(r"Final Answer\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if final_match:
            return {"final": final_match.group(1).strip()}

        thought = self._capture_block(text, "Thought")
        action = self._capture_block(text, "Action")
        action_input_text = self._capture_block(text, "Action Input")
        action_input = self._parse_action_input(action_input_text)

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    @staticmethod
    def _capture_block(text: str, key: str) -> str:
        pattern = rf"{key}\s*:\s*(.*?)(?:\n[A-Za-z ]+\s*:|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _parse_action_input(action_input_text: str) -> Dict[str, Any]:
        if not action_input_text:
            return {}

        cleaned = action_input_text.strip().strip("`")
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            if "=" in cleaned and "," in cleaned:
                pairs: Dict[str, Any] = {}
                for token in cleaned.split(","):
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    pairs[key.strip()] = value.strip().strip('"')
                if pairs:
                    return pairs
            return {"input": cleaned}

    @staticmethod
    def _extract_final_answer(final_text: str, fallback_class: str, fallback_conf: float) -> tuple[str, float]:
        class_match = re.search(r"Class\s*:\s*([A-Z_]+)", final_text, flags=re.IGNORECASE)
        conf_match = re.search(r"Confidence\s*:\s*([0-9]*\.?[0-9]+)", final_text, flags=re.IGNORECASE)

        detected_class = class_match.group(1).upper() if class_match else fallback_class
        if detected_class not in CLASS_ORDER:
            detected_class = fallback_class if fallback_class in CLASS_ORDER else "BACKGROUND"

        confidence = fallback_conf
        if conf_match:
            try:
                confidence = max(0.0, min(1.0, float(conf_match.group(1))))
            except ValueError:
                confidence = fallback_conf

        return detected_class, confidence
