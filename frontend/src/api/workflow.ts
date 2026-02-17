/**
 * API client for the Workflow Research Assistant.
 *
 * Communicates with /api/workflow/* endpoints and provides
 * typed helpers for SSE streaming.
 */

const BASE = "/api/workflow";

export type WorkflowStepData = {
  step_id: string;
  tool_name: string;
  description: string;
  params: Record<string, unknown>;
  status: "pending" | "awaiting_approval" | "running" | "completed" | "failed" | "skipped";
  result: Record<string, unknown> | null;
  error: string | null;
  requires_approval: boolean;
  approval_message: string | null;
  is_step_gate?: boolean;
  started_at: string | null;
  completed_at: string | null;
};

export type WorkflowSessionData = {
  session_id: string;
  user_goal: string;
  language: string;
  state: string;
  agent_type: string | null;
  plan_narrative: string | null;
  steps: WorkflowStepData[];
  current_step_index?: number;
  context_keys?: string[];
  report_markdown?: string | null;
};

export type WorkflowEvent = {
  event: string;
  data: Record<string, unknown>;
};

/** Create a workflow plan from a natural-language goal. */
export async function createWorkflow(
  goal: string,
  language?: string,
  mapContext?: Record<string, unknown>,
): Promise<WorkflowSessionData> {
  const res = await fetch(`${BASE}/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, language, map_context: mapContext }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Instantiate a prebuilt workflow template. */
export async function createFromTemplate(
  templateName: string,
  regionName?: string,
  lat?: number,
  lon?: number,
): Promise<WorkflowSessionData> {
  const res = await fetch(`${BASE}/template/${templateName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ region_name: regionName, lat, lon }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Start executing a workflow. Returns a Response to consume as SSE stream. */
export async function startWorkflow(sessionId: string): Promise<Response> {
  const res = await fetch(`${BASE}/${sessionId}/start`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res;
}

/** Approve a pending step. */
export async function approveStep(sessionId: string, stepId: string): Promise<void> {
  const res = await fetch(`${BASE}/${sessionId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ step_id: stepId }),
  });
  if (!res.ok) throw new Error(await res.text());
}

/** Deny (skip) a pending step. */
export async function denyStep(sessionId: string, stepId: string): Promise<void> {
  const res = await fetch(`${BASE}/${sessionId}/deny`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ step_id: stepId }),
  });
  if (!res.ok) throw new Error(await res.text());
}

/** Resume a paused workflow. Returns a Response to consume as SSE stream. */
export async function resumeWorkflow(sessionId: string): Promise<Response> {
  const res = await fetch(`${BASE}/${sessionId}/resume`);
  if (!res.ok) throw new Error(await res.text());
  return res;
}

/** Get current session state (polling fallback). */
export async function getWorkflowState(sessionId: string): Promise<WorkflowSessionData> {
  const res = await fetch(`${BASE}/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** List all workflow sessions. */
export async function listWorkflows(): Promise<WorkflowSessionData[]> {
  const res = await fetch(`${BASE}/sessions/list`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Cancel a running workflow. */
export async function cancelWorkflow(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/${sessionId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
}

/**
 * Consume an SSE response stream, calling onEvent for each parsed event.
 * Returns when the stream ends.
 */
export async function consumeSSE(
  response: Response,
  onEvent: (event: WorkflowEvent) => void,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event: WorkflowEvent = JSON.parse(line.slice(6));
          onEvent(event);
        } catch {
          // Skip malformed JSON
        }
      }
    }
  }

  // Process any remaining buffer
  if (buffer.startsWith("data: ")) {
    try {
      const event: WorkflowEvent = JSON.parse(buffer.slice(6));
      onEvent(event);
    } catch {
      // Skip
    }
  }
}
