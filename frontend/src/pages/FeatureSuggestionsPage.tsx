import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";

/* =========================================================
 * Types
 * =======================================================*/
type Status = "unread" | "in_progress" | "resolved";

interface Suggestion {
  id: string;
  title: string;
  description: string;
  status: Status;
  created_at: string;
}

/* =========================================================
 * Status helpers
 * =======================================================*/
const STATUS_META: Record<Status, { label: string; color: string; bg: string; border: string; icon: string }> = {
  unread:      { label: "Unread",      color: "text-slate-400", bg: "bg-slate-500/20", border: "border-slate-500/30", icon: "mark_email_unread" },
  in_progress: { label: "In Progress", color: "text-amber-400", bg: "bg-amber-500/20", border: "border-amber-500/30", icon: "pending" },
  resolved:    { label: "Resolved",    color: "text-green-400", bg: "bg-green-500/20", border: "border-green-500/30", icon: "check_circle" },
};

function StatusBadge({ status }: { status: Status }) {
  const m = STATUS_META[status];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-bold uppercase rounded-full border ${m.color} ${m.bg} ${m.border}`}>
      <span className="material-symbols-outlined text-xs">{m.icon}</span>
      {m.label}
    </span>
  );
}

/* =========================================================
 * API helpers
 * =======================================================*/
async function fetchSuggestions(): Promise<Suggestion[]> {
  const res = await fetch("/api/feature_suggestions");
  if (!res.ok) throw new Error("Failed to load suggestions");
  return res.json();
}

async function createSuggestion(title: string, description: string): Promise<Suggestion> {
  const res = await fetch("/api/feature_suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, description }),
  });
  if (!res.ok) throw new Error("Failed to create suggestion");
  return res.json();
}

async function updateStatus(id: string, status: Status): Promise<Suggestion> {
  const res = await fetch(`/api/feature_suggestions/${id}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error("Failed to update status");
  return res.json();
}

/* =========================================================
 * Page
 * =======================================================*/
export default function FeatureSuggestionsPage() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [showNewForm, setShowNewForm] = useState(false);
  const [filter, setFilter] = useState<Status | "all">("all");

  const load = useCallback(async () => {
    try {
      const data = await fetchSuggestions();
      setSuggestions(data);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleStatusChange = async (id: string, status: Status) => {
    const updated = await updateStatus(id, status);
    setSuggestions(prev => prev.map(s => (s.id === id ? updated : s)));
  };

  const handleCreated = (s: Suggestion) => {
    setSuggestions(prev => [s, ...prev]);
    setShowNewForm(false);
  };

  const detail = detailId ? suggestions.find(s => s.id === detailId) ?? null : null;

  const filtered = filter === "all" ? suggestions : suggestions.filter(s => s.status === filter);

  const counts = {
    all: suggestions.length,
    unread: suggestions.filter(s => s.status === "unread").length,
    in_progress: suggestions.filter(s => s.status === "in_progress").length,
    resolved: suggestions.filter(s => s.status === "resolved").length,
  };

  return (
    <div className="flex h-screen flex-col bg-bg-dark text-white">
      {/* Top bar */}
      <header className="flex h-14 items-center justify-between border-b border-border-dark px-6 shrink-0">
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-3">
            <div className="flex size-6 items-center justify-center text-primary">
              <span className="material-symbols-outlined text-2xl">rocket_launch</span>
            </div>
            <h1 className="text-xl font-bold tracking-tight">MarsLab</h1>
          </Link>
          <div className="h-6 w-px bg-border-dark" />
          <nav className="flex items-center gap-6">
            <Link to="/" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">
              Workbench
            </Link>
            <Link to="/download" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">
              Data Download
            </Link>
            <Link to="/upload" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">
              Data Upload
            </Link>
            <span className="text-sm font-medium text-white border-b-2 border-primary pb-1">
              Suggestions
            </span>
          </nav>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Page header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Feature Suggestions</h2>
              <p className="text-sm text-slate-500 mt-1">Ideas, feedback, and feature requests for MarsLab</p>
            </div>
            <button
              onClick={() => setShowNewForm(true)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-lg bg-primary/20 text-primary border border-primary/40 hover:bg-primary/30 transition-colors"
            >
              <span className="material-symbols-outlined text-lg">add</span>
              New Suggestion
            </button>
          </div>

          {/* Filter tabs */}
          <div className="flex items-center gap-1 mb-5 border-b border-border-dark pb-3">
            {(["all", "unread", "in_progress", "resolved"] as const).map(key => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className={`px-3 py-1.5 text-xs font-bold uppercase rounded-md transition-colors ${
                  filter === key
                    ? "bg-white/10 text-white"
                    : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                }`}
              >
                {key === "all" ? "All" : STATUS_META[key].label}
                <span className={`ml-1.5 text-[10px] ${filter === key ? "text-slate-300" : "text-slate-600"}`}>
                  {counts[key]}
                </span>
              </button>
            ))}
          </div>

          {/* List */}
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <span className="material-symbols-outlined animate-spin text-3xl text-primary">progress_activity</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-20">
              <span className="material-symbols-outlined text-4xl text-slate-600 mb-3 block">lightbulb</span>
              <p className="text-slate-500">
                {filter === "all" ? "No suggestions yet. Be the first!" : `No ${STATUS_META[filter as Status].label.toLowerCase()} suggestions.`}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {filtered.map(s => (
                <SuggestionRow
                  key={s.id}
                  suggestion={s}
                  onOpen={() => setDetailId(s.id)}
                  onStatusChange={handleStatusChange}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail modal */}
      {detail && (
        <DetailModal
          suggestion={detail}
          onClose={() => setDetailId(null)}
          onStatusChange={handleStatusChange}
        />
      )}

      {/* New suggestion modal */}
      {showNewForm && (
        <NewSuggestionModal
          onClose={() => setShowNewForm(false)}
          onCreated={handleCreated}
        />
      )}
    </div>
  );
}

/* =========================================================
 * SuggestionRow
 * =======================================================*/
function SuggestionRow({
  suggestion: s,
  onOpen,
  onStatusChange,
}: {
  suggestion: Suggestion;
  onOpen: () => void;
  onStatusChange: (id: string, status: Status) => void;
}) {
  const date = new Date(s.created_at);
  const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

  return (
    <div className="flex items-center gap-4 px-4 py-3 rounded-lg border border-border-dark bg-surface-dark/40 hover:bg-surface-dark/70 transition-colors group">
      {/* Status badge */}
      <StatusBadge status={s.status} />

      {/* Title (clickable) */}
      <button
        onClick={onOpen}
        className="flex-1 text-left text-sm font-medium text-slate-200 hover:text-white truncate transition-colors"
      >
        {s.title}
      </button>

      {/* Date */}
      <span className="text-[10px] text-slate-600 font-mono shrink-0">{dateStr}</span>

      {/* Status buttons */}
      <div className="flex gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <StatusButtons current={s.status} onChange={status => onStatusChange(s.id, status)} size="sm" />
      </div>
    </div>
  );
}

/* =========================================================
 * StatusButtons
 * =======================================================*/
function StatusButtons({
  current,
  onChange,
  size = "sm",
}: {
  current: Status;
  onChange: (status: Status) => void;
  size?: "sm" | "md";
}) {
  const statuses: Status[] = ["unread", "in_progress", "resolved"];
  const cls = size === "sm"
    ? "px-1.5 py-0.5 text-[9px]"
    : "px-2.5 py-1 text-[11px]";

  return (
    <>
      {statuses
        .filter(s => s !== current)
        .map(s => {
          const m = STATUS_META[s];
          return (
            <button
              key={s}
              onClick={() => onChange(s)}
              className={`${cls} font-bold uppercase rounded border transition-colors ${m.color} ${m.border} hover:${m.bg}`}
              title={`Mark as ${m.label}`}
            >
              {m.label}
            </button>
          );
        })}
    </>
  );
}

/* =========================================================
 * DetailModal
 * =======================================================*/
function DetailModal({
  suggestion: s,
  onClose,
  onStatusChange,
}: {
  suggestion: Suggestion;
  onClose: () => void;
  onStatusChange: (id: string, status: Status) => void;
}) {
  const date = new Date(s.created_at);
  const dateStr = date.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-[560px] max-w-[90vw] max-h-[80vh] flex flex-col rounded-xl border border-border-dark bg-[#101622] shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-border-dark">
          <div className="flex-1 min-w-0 pr-4">
            <h3 className="text-lg font-bold text-white break-words">{s.title}</h3>
            <div className="flex items-center gap-3 mt-2">
              <StatusBadge status={s.status} />
              <span className="text-[10px] text-slate-500 font-mono">{dateStr}</span>
            </div>
          </div>
          <button onClick={onClose} className="p-1 text-slate-500 hover:text-white transition-colors shrink-0">
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Description */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          <p className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
            {s.description || <span className="italic text-slate-600">No description provided.</span>}
          </p>
        </div>

        {/* Footer: status actions */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-border-dark">
          <span className="text-[10px] text-slate-600 font-mono">ID: {s.id}</span>
          <div className="flex gap-2">
            <StatusButtons
              current={s.status}
              onChange={status => onStatusChange(s.id, status)}
              size="md"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
 * NewSuggestionModal
 * =======================================================*/
function NewSuggestionModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (s: Suggestion) => void;
}) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleSubmit = async () => {
    if (!title.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const s = await createSuggestion(title.trim(), description.trim());
      onCreated(s);
    } catch {
      setError("Failed to submit suggestion. Please try again.");
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-[480px] max-w-[90vw] rounded-xl border border-border-dark bg-[#101622] shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border-dark">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">New Suggestion</h3>
          <button onClick={onClose} className="p-1 text-slate-500 hover:text-white transition-colors">
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Form */}
        <div className="px-5 py-4 space-y-3">
          <div>
            <label className="block text-[10px] text-slate-500 font-bold uppercase mb-1">Title</label>
            <input
              autoFocus
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="Short summary of the idea..."
              className="w-full rounded-md border border-border-dark bg-[#0a0f18] px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-primary/50"
            />
          </div>
          <div>
            <label className="block text-[10px] text-slate-500 font-bold uppercase mb-1">Description</label>
            <textarea
              rows={5}
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Describe the feature in detail..."
              className="w-full rounded-md border border-border-dark bg-[#0a0f18] px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-primary/50 resize-y"
            />
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}

          <div className="flex justify-end pt-1">
            <button
              disabled={!title.trim() || submitting}
              onClick={handleSubmit}
              className="flex items-center gap-2 px-4 py-2 text-xs font-bold uppercase rounded-lg bg-primary/20 text-primary border border-primary/40 hover:bg-primary/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <>
                  <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
                  Submitting...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-sm">send</span>
                  Submit
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
