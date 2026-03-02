/**
 * API client for HiRISE landform classification endpoints.
 */

/* =========================================================
 * Types
 * =======================================================*/

export type ModelType = "v2" | "mars-bench";

export interface ClassifyRequest {
  product_id: string;
  model: ModelType;
  include_heatmap: boolean;
}

export interface LandformClass {
  class_name: string;
  class_code: string;
  probability: number;
}

export interface ClassifyResult {
  product_id: string;
  model: ModelType;
  top_class: string;
  confidence: number;
  classes: LandformClass[];
  heatmap_url: string | null;
  processing_time_s: number;
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
  LDA: { label: "Lobate Debris Apron", icon: "landslide", color: "bg-amber-500" },
  LVF: { label: "Lineated Valley Fill", icon: "timeline", color: "bg-cyan-500" },
  CCF: { label: "Concentric Crater Fill", icon: "target", color: "bg-violet-500" },
  GLF: { label: "Glacier-Like Form", icon: "ac_unit", color: "bg-blue-500" },
  OTHER: { label: "Other", icon: "help_outline", color: "bg-slate-500" },
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
