import { useState, useCallback, useEffect, useRef } from "react";
import LandformClassCard from "./LandformClassCard";
import {
  submitClassification,
  pollJobStatus,
  type ModelType,
  type ClassifyResult,
  type JobStatus,
} from "../api/hirise_landforms";

/* =========================================================
 * Props
 * =======================================================*/
export interface HiriseLandformPanelProps {
  productId: string;
  onClose: () => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function HiriseLandformPanel({
  productId,
  onClose,
}: HiriseLandformPanelProps) {
  const [model, setModel] = useState<ModelType>("v2");
  const [includeHeatmap, setIncludeHeatmap] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ── Cleanup polling on unmount ──────────────────────── */
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  /* ── Submit classification ───────────────────────────── */
  const handleClassify = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setJobStatus(null);

    try {
      const { job_id } = await submitClassification({
        product_id: productId,
        model,
        include_heatmap: includeHeatmap,
      });
      setJobId(job_id);

      // Start polling every 2 seconds
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollJobStatus(job_id);
          setJobStatus(status);

          if (status.status === "completed" && status.result) {
            setResult(status.result);
            setLoading(false);
            if (pollRef.current) clearInterval(pollRef.current);
          } else if (status.status === "failed") {
            setError(status.error ?? "Classification failed");
            setLoading(false);
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch (e: unknown) {
          const msg = e instanceof Error ? e.message : "Polling failed";
          setError(msg);
          setLoading(false);
          if (pollRef.current) clearInterval(pollRef.current);
        }
      }, 2000);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Submission failed";
      setError(msg);
      setLoading(false);
    }
  }, [productId, model, includeHeatmap]);

  /* ── Progress percentage ─────────────────────────────── */
  const progress = jobStatus?.progress ?? 0;

  /* ── Render ──────────────────────────────────────────── */
  return (
    <section className="flex flex-col border-b border-[#232f48]">
      {/* Header */}
      <div className="flex items-center bg-[#0d1520] px-3 py-2">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex flex-1 items-center gap-1.5 text-left"
        >
          <span className="material-symbols-outlined text-[16px] text-violet-400">
            image_search
          </span>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
            Landform Classification
          </span>
          <span className="material-symbols-outlined ml-auto text-[14px] text-[#92a4c9]">
            {collapsed ? "expand_more" : "expand_less"}
          </span>
        </button>
        <button
          onClick={onClose}
          className="ml-2 text-[#92a4c9] hover:text-slate-200"
        >
          <span className="material-symbols-outlined text-[14px]">close</span>
        </button>
      </div>

      {!collapsed && (
        <div className="flex flex-col gap-2 bg-[#0d1520] px-3 pb-3 pt-1">
          {/* Product ID */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#92a4c9]">Product:</span>
            <span className="truncate font-mono text-[10px] text-slate-200">
              {productId}
            </span>
          </div>

          {/* Model selector */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-[#92a4c9]">
              Model
            </span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as ModelType)}
              className="rounded border border-[#232f48] bg-[#111b2a] px-2 py-0.5 text-[11px] text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500/50"
            >
              <option value="v2">V2 (Standard)</option>
              <option value="mars-bench">Mars-Bench</option>
            </select>
          </div>

          {/* Options row */}
          <div className="flex items-center gap-3">
            <label className="flex cursor-pointer items-center gap-1">
              <input
                type="checkbox"
                checked={includeHeatmap}
                onChange={(e) => setIncludeHeatmap(e.target.checked)}
                className="h-3 w-3 rounded border-[#232f48] bg-[#0d1520] accent-blue-500"
              />
              <span className="text-[10px] text-[#92a4c9]">Attention Heatmap</span>
            </label>

            <button
              onClick={handleClassify}
              disabled={loading}
              className="ml-auto rounded bg-violet-600/20 px-3 py-1 text-[10px] font-medium text-violet-300 transition-colors hover:bg-violet-600/30 disabled:opacity-50"
            >
              {loading ? "Processing…" : "Classify"}
            </button>
          </div>

          {/* Progress bar */}
          {loading && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-[9px] text-[#92a4c9]">
                <span>
                  {jobStatus?.status === "queued"
                    ? "Queued…"
                    : jobStatus?.status === "processing"
                      ? "Processing…"
                      : "Submitting…"}
                </span>
                <span className="font-mono">{Math.round(progress * 100)}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[#1a2744]">
                <div
                  className="h-full rounded-full bg-violet-500 transition-all"
                  style={{ width: `${Math.max(progress * 100, 2)}%` }}
                />
              </div>
              {jobId && (
                <span className="font-mono text-[8px] text-slate-500">
                  Job: {jobId}
                </span>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="rounded border border-red-500/20 bg-red-500/5 px-2 py-1 text-[10px] text-red-400">
              {error}
            </div>
          )}

          {/* Results */}
          {result && !error && (
            <div className="flex flex-col gap-2">
              {/* Summary */}
              <div className="flex items-center gap-2 rounded border border-[#232f48] bg-[#111b2a] px-2 py-1.5">
                <span className="material-symbols-outlined text-[14px] text-green-400">
                  check_circle
                </span>
                <div className="flex flex-col">
                  <span className="text-[11px] font-medium text-slate-200">
                    {result.top_class}
                  </span>
                  <span className="text-[9px] text-[#92a4c9]">
                    Confidence: {Math.round(result.confidence * 100)}% ·{" "}
                    {result.processing_time_s.toFixed(1)}s
                  </span>
                </div>
              </div>

              {/* Class cards */}
              <div className="flex flex-col gap-1">
                <span className="text-[9px] uppercase tracking-wider text-[#92a4c9]">
                  All Classes
                </span>
                {result.classes
                  .sort((a, b) => b.probability - a.probability)
                  .map((cls) => (
                    <LandformClassCard
                      key={cls.class_code}
                      classCode={cls.class_code}
                      className_={cls.class_name}
                      probability={cls.probability}
                      isTopClass={cls.class_name === result.top_class}
                    />
                  ))}
              </div>

              {/* Attention heatmap */}
              {result.heatmap_url && (
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-[#92a4c9]">
                    Attention Heatmap
                  </span>
                  <div className="overflow-hidden rounded border border-[#232f48]">
                    <img
                      src={result.heatmap_url}
                      alt="Attention heatmap"
                      className="h-auto w-full"
                      loading="lazy"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
