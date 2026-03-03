/**
 * API client for HiRISE landform classification endpoints.
 */

/* =========================================================
 * Types
 * =======================================================*/

export type ModelType = "v3" | "v2";

export interface ClassifyRequest {
  product_id: string;
  model: ModelType;
  include_heatmap: boolean;
}

export interface TilePrediction {
  x: number;
  y: number;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  lat: number;
  lon: number;
}

export interface ClassSummary {
  class_name: string;
  tile_count: number;
  percentage: number;
  mean_confidence: number;
}

export interface AgentReasoningStep {
  step: number;
  action?: string | null;
  action_input?: Record<string, unknown> | null;
  observation?: Record<string, unknown> | string | null;
  thought?: string | null;
  vlm_response?: string | null;
  error?: string | null;
  forced_final?: boolean;
}

export interface AgentReasoning {
  enabled: boolean;
  mode: string;
  landform_class?: string | null;
  confidence?: number | null;
  reasoning_chain: AgentReasoningStep[];
  tools_used: string[];
  num_steps: number;
  error?: string | null;
}

export interface ClassifyResult {
  product_id: string;
  model_used: string;
  tile_predictions: TilePrediction[];
  class_summary: ClassSummary[];
  dominant_class: string;
  dominant_confidence: number;
  heatmap_url: string | null;
  processing_time_s: number;
  agent_reasoning?: AgentReasoning | null;
  num_tiles?: number;
  device?: string;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  result?: ClassifyResult;
  error?: string;
}

export interface ClassificationServiceStatus {
  models_loaded: string[];
  device: string;
  queue_length: number;
}

/* =========================================================
 * Landform type metadata
 * =======================================================*/

export const LANDFORM_TYPES: Record<string, { label: string; icon: string; color: string }> = {
  LDA: { label: "Lobate Debris Apron", icon: "landslide", color: "bg-blue-500" },
  LVF: { label: "Lineated Valley Fill", icon: "timeline", color: "bg-emerald-500" },
  CCF: { label: "Concentric Crater Fill", icon: "target", color: "bg-amber-500" },
  OTHER: { label: "Other Terrain", icon: "landscape", color: "bg-slate-500" },
};

/* =========================================================
 * Fetchers
 * =======================================================*/

export async function submitClassification(
  req: ClassifyRequest,
): Promise<{ job_id: string }> {
  const res = await fetch("/api/hirise-landforms/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Classification submission failed (${res.status})`);
  }
  return res.json();
}

export async function pollJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/hirise-landforms/jobs/${jobId}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Job status query failed (${res.status})`);
  }
  return res.json();
}

export async function getClassificationStatus(): Promise<ClassificationServiceStatus> {
  const res = await fetch("/api/hirise-landforms/status");
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Service status query failed (${res.status})`);
  }
  return res.json();
}
