import { useState, useEffect, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import DOMPurify from "dompurify";

/* =========================================================
 * Types
 * =======================================================*/

interface DiscussionSummary {
  date: string;
  topic: string;
  science_keywords: string;
  marslab_features: string;
  word_count: number;
  size_bytes: number;
}

interface DiscussionDetail {
  date: string;
  topic: string;
  science_keywords: string;
  marslab_features: string;
  content: string;
}

/* =========================================================
 * Markdown → HTML (inline, matches project convention)
 * =======================================================*/

function discussionMarkdownToHtml(md: string): string {
  let html = md
    // Remove front-matter block (everything before first ## Focus:)
    .replace(/^# MarsLab Daily Discussion[\s\S]*?---\n/m, "")
    // Product ID links — HiRISE
    .replace(/(ESP_\d{6}_\d{4})/g, '<a href="/download?product_id=$1&instrument=HIRISE" class="text-primary hover:text-primary/80 underline">$1</a>')
    .replace(/(PSP_\d{6}_\d{4})/g, '<a href="/download?product_id=$1&instrument=HIRISE" class="text-primary hover:text-primary/80 underline">$1</a>')
    // Product ID links — CRISM
    .replace(/(frt[0-9a-f]{8})/g, '<a href="/download?product_id=$1&instrument=CRISM" class="text-primary hover:text-primary/80 underline">$1</a>')
    .replace(/(hrs[0-9a-f]{8})/g, '<a href="/download?product_id=$1&instrument=CRISM" class="text-primary hover:text-primary/80 underline">$1</a>')
    // Product ID links — SHARAD
    .replace(/(R_\d{7}_\d{3}_SS\d+_\d+_A)/g, '<a href="/download?product_id=$1&instrument=SHARAD" class="text-primary hover:text-primary/80 underline">$1</a>')
    // Product ID links — DTM
    .replace(/(DTE[A-Z]{2}_\d{6}_\d{4}_\d{6}_\d{4}_[A-Z]\d{2})/g, '<a href="/download?product_id=$1&instrument=HIRISE_DTM" class="text-primary hover:text-primary/80 underline">$1</a>')
    // Headings
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-bold text-white mt-8 mb-3 flex items-center gap-2"><span class="w-1 h-5 rounded-full bg-primary inline-block"></span>$1</h3>')
    .replace(/^## Focus: (.+)$/gm, '<h2 class="text-xl font-bold text-white mb-4">$1</h2>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold text-white mt-8 mb-3">$1</h2>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em class="text-slate-300 italic">$1</em>')
    // Numbered list items
    .replace(/^(\d+)\.\s+(.+)$/gm, '<li class="ml-4 mb-2 text-slate-300 list-decimal list-inside"><span>$2</span></li>')
    // Wrap consecutive <li> in <ol>
    .replace(/((?:<li[^>]*>.*<\/li>\n?)+)/g, '<ol class="my-3 space-y-1">$1</ol>')
    // Paragraphs (lines not already tagged)
    .split("\n\n")
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (trimmed.startsWith("<")) return trimmed;
      return `<p class="text-slate-300 leading-relaxed mb-4">${trimmed.replace(/\n/g, " ")}</p>`;
    })
    .join("\n");

  return html;
}

/* =========================================================
 * API helpers
 * =======================================================*/

async function fetchDiscussionList(): Promise<DiscussionSummary[]> {
  const resp = await fetch("/api/discussions");
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.discussions ?? [];
}

async function fetchDiscussion(date: string): Promise<DiscussionDetail | null> {
  const resp = await fetch(`/api/discussions/${date}`);
  if (!resp.ok) return null;
  return resp.json();
}

/* =========================================================
 * Sub-components
 * =======================================================*/

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr + "T00:00:00");
    return d.toLocaleDateString("en-US", { weekday: "short", year: "numeric", month: "short", day: "numeric" });
  } catch {
    return dateStr;
  }
}

function DiscussionListItem({
  item,
  isSelected,
  onClick,
}: {
  item: DiscussionSummary;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg p-3 transition-all ${
        isSelected
          ? "bg-primary/15 border border-primary/40"
          : "bg-transparent border border-transparent hover:bg-white/5 hover:border-border-dark"
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className={`text-xs font-mono ${isSelected ? "text-primary" : "text-slate-500"}`}>
          {formatDate(item.date)}
        </span>
        <span className="text-[9px] text-slate-600 font-mono">{item.word_count}w</span>
      </div>
      <div className={`text-sm font-medium leading-snug ${isSelected ? "text-white" : "text-slate-300"}`}>
        {item.topic || "Untitled Discussion"}
      </div>
      {item.science_keywords && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {item.science_keywords
            .split(",")
            .slice(0, 3)
            .map((kw) => (
              <span
                key={kw.trim()}
                className="text-[9px] rounded-full px-1.5 py-0.5 bg-white/5 text-slate-500 border border-white/5"
              >
                {kw.trim()}
              </span>
            ))}
        </div>
      )}
    </button>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3 p-2">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="rounded-lg p-3 animate-pulse">
          <div className="h-3 w-24 bg-white/5 rounded mb-2" />
          <div className="h-4 w-48 bg-white/5 rounded mb-2" />
          <div className="flex gap-1 mt-2">
            <div className="h-3 w-16 bg-white/5 rounded-full" />
            <div className="h-3 w-12 bg-white/5 rounded-full" />
          </div>
        </div>
      ))}
    </div>
  );
}

function MetadataHeader({ discussion }: { discussion: DiscussionDetail }) {
  const keywords = discussion.science_keywords
    ? discussion.science_keywords.split(",").map((s) => s.trim())
    : [];
  const features = discussion.marslab_features
    ? discussion.marslab_features.split(",").map((s) => s.trim())
    : [];

  return (
    <div className="mb-6 rounded-xl border border-border-dark bg-white/[0.02] p-5">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/15 border border-primary/30 px-3 py-1 text-xs font-bold text-primary">
          <span className="material-symbols-outlined text-sm">topic</span>
          {discussion.topic}
        </span>
        <span className="text-xs text-slate-500 font-mono">{formatDate(discussion.date)}</span>
      </div>

      {keywords.length > 0 && (
        <div className="mb-2">
          <span className="text-[9px] uppercase text-slate-600 tracking-widest mr-2">Science</span>
          {keywords.map((kw) => (
            <span
              key={kw}
              className="inline-block text-[10px] rounded-full px-2 py-0.5 mr-1 mb-1 bg-amber-500/10 border border-amber-500/20 text-amber-400/80"
            >
              {kw}
            </span>
          ))}
        </div>
      )}

      {features.length > 0 && (
        <div>
          <span className="text-[9px] uppercase text-slate-600 tracking-widest mr-2">Features</span>
          {features.map((f) => (
            <span
              key={f}
              className="inline-block text-[10px] rounded-full px-2 py-0.5 mr-1 mb-1 bg-sky-500/10 border border-sky-500/20 text-sky-400/80"
            >
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <span className="material-symbols-outlined text-5xl text-slate-700 mb-4">forum</span>
      <h3 className="text-lg font-bold text-slate-400 mb-2">No discussions yet</h3>
      <p className="text-sm text-slate-600 max-w-md">
        Daily AI discussions are generated automatically each morning. They simulate
        team meetings between planetary scientists analyzing Arcadia Planitia data
        using MarsLab tools.
      </p>
      <p className="text-xs text-slate-700 mt-3 font-mono">
        Run: python backend/scripts/daily_discussion.py
      </p>
    </div>
  );
}

/* =========================================================
 * Main Page Component
 * =======================================================*/

export default function DailyDiscussionsPage() {
  const [discussions, setDiscussions] = useState<DiscussionSummary[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [detail, setDetail] = useState<DiscussionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [mobileListOpen, setMobileListOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [topicFilter, setTopicFilter] = useState<string | null>(null);

  // Derived: unique topics from all discussions
  const uniqueTopics = useMemo(
    () => Array.from(new Set(discussions.map((d) => d.topic).filter(Boolean))),
    [discussions]
  );

  // Derived: filtered discussion list
  const filteredDiscussions = useMemo(() => {
    let list = discussions;
    if (topicFilter) {
      list = list.filter((d) => d.topic === topicFilter);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      list = list.filter(
        (d) =>
          d.topic.toLowerCase().includes(q) ||
          d.science_keywords.toLowerCase().includes(q)
      );
    }
    return list;
  }, [discussions, topicFilter, searchQuery]);

  // Load discussion list
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchDiscussionList().then((list) => {
      if (cancelled) return;
      setDiscussions(list);
      if (list.length > 0 && !selectedDate) {
        setSelectedDate(list[0]?.date ?? null);
      }
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load selected discussion detail
  useEffect(() => {
    if (!selectedDate) return;
    let cancelled = false;
    setDetailLoading(true);
    fetchDiscussion(selectedDate).then((d) => {
      if (cancelled) return;
      setDetail(d);
      setDetailLoading(false);
    });
    return () => { cancelled = true; };
  }, [selectedDate]);

  const handleSelect = useCallback((date: string) => {
    setSelectedDate(date);
    setMobileListOpen(false);
  }, []);

  const renderedHtml = detail
    ? DOMPurify.sanitize(discussionMarkdownToHtml(detail.content))
    : "";

  return (
    <div className="h-screen flex flex-col bg-[#0a0f18] text-slate-200">
      {/* Top bar */}
      <header className="flex items-center justify-between h-14 px-6 border-b border-border-dark bg-bg-dark shrink-0">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined text-lg">arrow_back</span>
            <span className="text-sm font-medium">Workbench</span>
          </Link>
          <div className="h-5 w-px bg-border-dark" />
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-xl">forum</span>
            <h1 className="text-lg font-bold tracking-tight">Daily Discussions</h1>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="font-mono">{discussions.length} discussion{discussions.length !== 1 ? "s" : ""}</span>
        </div>
      </header>

      {/* Mobile list toggle */}
      <div className="md:hidden flex items-center border-b border-border-dark px-4 py-2 bg-bg-dark">
        <button
          onClick={() => setMobileListOpen((p) => !p)}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white"
        >
          <span className="material-symbols-outlined text-base">
            {mobileListOpen ? "expand_less" : "expand_more"}
          </span>
          {selectedDate ? `${formatDate(selectedDate)}` : "Select discussion"}
        </button>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — discussion list */}
        <aside
          className={`${
            mobileListOpen ? "block absolute inset-x-0 top-[7rem] z-40 bg-[#0a0f18] border-b border-border-dark max-h-[60vh]" : "hidden"
          } md:block md:relative md:max-h-none w-full md:w-80 shrink-0 border-r border-border-dark overflow-y-auto`}
        >
          {/* Search bar */}
          <div className="p-3 border-b border-border-dark space-y-2">
            <div className="relative">
              <span className="material-symbols-outlined absolute left-2.5 top-1/2 -translate-y-1/2 text-sm text-slate-500">search</span>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search discussions..."
                className="w-full rounded-lg bg-white/5 border border-border-dark pl-8 pr-8 py-1.5 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                >
                  <span className="material-symbols-outlined text-sm">close</span>
                </button>
              )}
            </div>

            {/* Topic filter badges */}
            {uniqueTopics.length > 0 && (
              <div className="flex flex-wrap gap-1">
                <button
                  onClick={() => setTopicFilter(null)}
                  className={`text-[9px] rounded-full px-2 py-0.5 border transition-colors ${
                    topicFilter === null
                      ? "bg-primary/20 border-primary/40 text-primary font-bold"
                      : "bg-white/5 border-white/5 text-slate-500 hover:border-slate-500"
                  }`}
                >
                  All
                </button>
                {uniqueTopics.map((topic) => (
                  <button
                    key={topic}
                    onClick={() => setTopicFilter(topicFilter === topic ? null : topic)}
                    className={`text-[9px] rounded-full px-2 py-0.5 border transition-colors truncate max-w-[140px] ${
                      topicFilter === topic
                        ? "bg-primary/20 border-primary/40 text-primary font-bold"
                        : "bg-white/5 border-white/5 text-slate-500 hover:border-slate-500"
                    }`}
                  >
                    {topic}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-slate-600 font-bold">
              <span className="material-symbols-outlined text-xs">calendar_month</span>
              Archive
            </div>
          </div>

          {loading ? (
            <LoadingSkeleton />
          ) : filteredDiscussions.length === 0 ? (
            <div className="p-4 text-sm text-slate-600">No discussions found.</div>
          ) : (
            <div className="p-2 space-y-1">
              {filteredDiscussions.map((d) => (
                <DiscussionListItem
                  key={d.date}
                  item={d}
                  isSelected={d.date === selectedDate}
                  onClick={() => handleSelect(d.date)}
                />
              ))}
            </div>
          )}
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto">
          {detailLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex items-center gap-3 text-slate-500">
                <span className="material-symbols-outlined animate-spin text-2xl">progress_activity</span>
                <span className="text-sm">Loading discussion…</span>
              </div>
            </div>
          ) : detail ? (
            <div className="max-w-3xl mx-auto px-6 py-8">
              <MetadataHeader discussion={detail} />
              <article
                className="prose-marslab"
                dangerouslySetInnerHTML={{ __html: renderedHtml }}
              />
            </div>
          ) : discussions.length === 0 && !loading ? (
            <EmptyState />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-600 text-sm">
              Select a discussion from the sidebar
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
