import { useState, useCallback, useEffect, useRef } from "react";
import {
  submitClassification,
  pollJobStatus,
  type ModelType,
  type ClassifyResult,
  type JobStatus,
  type AgentReasoning,
} from "../api/hirise_landforms";
import { registerLandform } from "../api/accessibility";

/* =========================================================
 * Props
 * =======================================================*/
export interface HiriseLandformPanelProps {
  productId: string;
  lat?: number;
  lon?: number;
  onClose: () => void;
}

/* =========================================================
 * Agent Reasoning Sub-component
 * =======================================================*/
function AgentReasoningPanel({ reasoning }: { reasoning: AgentReasoning }) {
  const [expanded, setExpanded] = useState(false);

  if (!reasoning.enabled) return null;

  const hasError = !!reasoning.error;
  const hasChain = reasoning.reasoning_chain.length > 0;

  return (
    <div className="flex flex-col gap-1.5 rounded border border-amber-500/20 bg-amber-500/5 px-2 py-1.5">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-left"
      >
        <span className="material-symbols-outlined text-[14px] text-amber-400">
          psychology
        </span>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-300">
          VLM Agent Reasoning
        </span>
        <span className="ml-auto flex items-center gap-1">
          {reasoning.num_steps > 0 && (
            <span className="rounded bg-amber-600/20 px-1 py-0.5 text-[8px] text-amber-400">
              {reasoning.num_steps} step{reasoning.num_steps !== 1 ? "s" : ""}
            </span>
          )}
          <span className="material-symbols-outlined text-[12px] text-[#92a4c9]">
            {expanded ? "expand_less" : "expand_more"}
          </span>
        </span>
      </button>

      {/* Summary line */}
      <div className="flex items-center gap-2 text-[9px] text-[#92a4c9]">
        <span>Mode: {reasoning.mode}</span>
        {reasoning.tools_used.length > 0 && (
          <span>· Tools: {reasoning.tools_used.join(", ")}</span>
        )}
        {reasoning.landform_class && (
          <span>
            · Result: {reasoning.landform_class} (
            {reasoning.confidence != null
              ? `${Math.round(reasoning.confidence * 100)}%`
              : "?"}
            )
          </span>
        )}
      </div>

      {/* Error */}
      {hasError && (
        <div className="rounded border border-red-500/20 bg-red-500/5 px-2 py-1 text-[9px] text-red-400">
          {reasoning.error}
        </div>
      )}

      {/* Expanded reasoning chain */}
      {expanded && hasChain && (
        <div className="flex flex-col gap-1 border-t border-amber-500/10 pt-1.5">
          {reasoning.reasoning_chain.map((step, idx) => (
            <div
              key={idx}
              className="flex flex-col gap-0.5 rounded bg-[#0d1520] px-2 py-1"
            >
              <div className="flex items-center gap-1.5">
                <span className="rounded bg-[#1a2744] px-1 py-0.5 font-mono text-[8px] text-[#92a4c9]">
                  #{step.step}
                </span>
                {step.action && (
                  <span className="rounded bg-violet-600/20 px-1 py-0.5 text-[8px] font-medium text-violet-300">
                    {step.action}
                  </span>
                )}
                {step.forced_final && (
                  <span className="rounded bg-orange-600/20 px-1 py-0.5 text-[8px] text-orange-400">
                    forced
                  </span>
                )}
                {step.error && (
                  <span className="rounded bg-red-600/20 px-1 py-0.5 text-[8px] text-red-400">
                    error
                  </span>
                )}
              </div>

              {/* Thought */}
              {step.thought && (
                <p className="text-[9px] italic text-slate-400">
                  💭 {step.thought}
                </p>
              )}

              {/* Action input */}
              {step.action_input &&
                Object.keys(step.action_input).length > 0 && (
                  <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-[#111b2a] px-1.5 py-0.5 font-mono text-[8px] text-slate-500">
                    {JSON.stringify(step.action_input, null, 1)}
                  </pre>
                )}

              {/* Observation (truncated) */}
              {step.observation && (
                <pre className="max-h-16 overflow-hidden whitespace-pre-wrap rounded bg-[#111b2a] px-1.5 py-0.5 font-mono text-[8px] text-slate-500">
                  {typeof step.observation === "string"
                    ? step.observation.slice(0, 300)
                    : JSON.stringify(step.observation, null, 1).slice(0, 300)}
                </pre>
              )}

              {/* Error */}
              {step.error && (
                <p className="text-[8px] text-red-400">{step.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* =========================================================
 * Component
 * =======================================================*/
export default function HiriseLandformPanel({
  productId,
  lat: productLat,
  lon: productLon,
  onClose,
}: HiriseLandformPanelProps) {
  const [model, setModel] = useState<ModelType>("v3");
  const [includeHeatmap, setIncludeHeatmap] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fusionRegistered, setFusionRegistered] = useState(false);
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

  /* ── Auto-register in fusion cache when classification completes ── */
  useEffect(() => {
    if (!result || fusionRegistered) return;
    if (productLat == null || productLon == null) return;
    if (result.dominant_class === "OTHER") return;

    registerLandform(
      result.product_id,
      productLat,
      productLon,
      result.dominant_class,
      result.dominant_confidence,
      result.model_used,
    ).then(() => {
      setFusionRegistered(true);
    }).catch(() => {
      // Non-critical — fusion cache registration failed silently
    });
  }, [result, fusionRegistered, productLat, productLon]);

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
              <option value="v3">V3 — DINOv2 + Tile Classifier</option>
              <option value="v2">V2 — DINOv2 + MIL + VLM</option>
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
              <span className="text-[10px] text-[#92a4c9]">Class Map</span>
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
                      ? "Running tile-level classification…"
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
                <span className="material-symbols-outlined text-[14px] text-green-400">check_circle</span>
                <div className="flex flex-col">
                  <span className="text-[11px] font-medium text-slate-200">{result.dominant_class}</span>
                  <span className="text-[9px] text-[#92a4c9]">
                    Confidence: {Math.round(result.dominant_confidence * 100)}%
                    {result.processing_time_s != null && ` · ${result.processing_time_s.toFixed(1)}s`}
                    {result.num_tiles != null && ` · ${result.num_tiles} tiles`}
                  </span>
                </div>
              </div>

              {/* Fusion registration badge */}
              {fusionRegistered && (
                <div className="flex items-center gap-1.5 rounded border border-violet-500/30 bg-violet-500/10 px-2 py-1">
                  <span className="material-symbols-outlined text-[12px] text-violet-400">hub</span>
                  <span className="text-[9px] text-violet-300">Added to ice prospecting fusion</span>
                </div>
              )}

              {/* Stacked distribution bar */}
              <div className="flex flex-col gap-1">
                <span className="text-[9px] uppercase tracking-wider text-[#92a4c9]">Tile Distribution</span>
                <div className="flex h-3 w-full overflow-hidden rounded-full bg-[#1a2744]">
                  {result.class_summary
                    ?.filter(s => s.tile_count > 0)
                    .map(s => {
                      const colors: Record<string, string> = {
                        LDA: "bg-blue-500", LVF: "bg-emerald-500", CCF: "bg-amber-500", OTHER: "bg-slate-600"
                      };
                      return (
                        <div
                          key={s.class_name}
                          className={`${colors[s.class_name] || "bg-slate-600"} transition-all`}
                          style={{ width: `${s.percentage}%` }}
                          title={`${s.class_name}: ${s.tile_count} tiles (${Math.round(s.percentage)}%)`}
                        />
                      );
                    })}
                </div>
                {/* Class breakdown list */}
                <div className="flex flex-col gap-0.5">
                  {result.class_summary
                    ?.filter(s => s.tile_count > 0)
                    .sort((a, b) => b.tile_count - a.tile_count)
                    .map(s => {
                      const dotColors: Record<string, string> = {
                        LDA: "bg-blue-500", LVF: "bg-emerald-500", CCF: "bg-amber-500", OTHER: "bg-slate-500"
                      };
                      return (
                        <div key={s.class_name} className="flex items-center gap-2 px-1">
                          <span className={`h-2 w-2 rounded-full ${dotColors[s.class_name] || "bg-slate-500"}`} />
                          <span className="text-[10px] font-medium text-slate-300 w-8">{s.class_name}</span>
                          <span className="text-[9px] text-[#92a4c9]">
                            {s.tile_count} tiles ({Math.round(s.percentage)}%) · {Math.round(s.mean_confidence * 100)}% conf
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* VLM Agent Reasoning */}
              {result.agent_reasoning && (
                <AgentReasoningPanel reasoning={result.agent_reasoning} />
              )}

              {/* Class Map */}
              {result.heatmap_url && (
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-[#92a4c9]">Class Map</span>
                  <div className="overflow-hidden rounded border border-[#232f48]">
                    <img src={result.heatmap_url} alt="Class map" className="h-auto w-full" loading="lazy" />
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
