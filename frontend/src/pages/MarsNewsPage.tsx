import { useState, useEffect, useCallback, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import DOMPurify from "dompurify";
import MarsResearchPage from "./MarsResearchPage";

/* =========================================================
 * Types
 * =======================================================*/

interface NewsSummary {
  date: string;
  item_count: number;
  categories: string[];
  trend_summary: string;
}

interface NewsItem {
  title: string;
  source: string;
  date: string;
  summary: string;
  significance: string;
  category: string;
  url: string | null;
}

interface NewsDetail {
  date: string;
  items: NewsItem[];
  trend_summary: string;
  summary_md: string;
  categories: string[];
}

/* =========================================================
 * Markdown → HTML (inline, matches project convention)
 * =======================================================*/

function newsMarkdownToHtml(md: string): string {
  const html = md
    // Headings
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-bold text-white mt-8 mb-3 flex items-center gap-2"><span class="w-1 h-5 rounded-full bg-primary inline-block"></span>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold text-white mt-8 mb-3">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold text-white mt-8 mb-4">$1</h1>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em class="text-slate-300 italic">$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-white/10 text-primary text-xs font-mono">$1</code>')
    // Numbered list items
    .replace(/^(\d+)\.\s+(.+)$/gm, '<li class="ml-4 mb-2 text-slate-300 list-decimal list-inside"><span>$2</span></li>')
    // Wrap consecutive <li> in <ol>
    .replace(/((?:<li[^>]*>.*<\/li>\n?)+)/g, '<ol class="my-3 space-y-1">$1</ol>')
    // Unordered bullet lists
    .replace(/^- (.+)$/gm, '<li class="ml-4 mb-2 text-slate-300 list-disc list-inside"><span>$1</span></li>')
    // Wrap consecutive <li> with list-disc in <ul>
    .replace(/((?:<li[^>]*list-disc[^>]*>.*<\/li>\n?)+)/g, '<ul class="my-3 space-y-1">$1</ul>')
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
 * Helpers
 * =======================================================*/

function getCategoryStyle(category: string): string {
  const styles: Record<string, string> = {
    missions: "bg-sky-500/10 border-sky-500/20 text-sky-400/80",
    discoveries: "bg-amber-500/10 border-amber-500/20 text-amber-400/80",
    technology: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400/80",
    human_exploration: "bg-purple-500/10 border-purple-500/20 text-purple-400/80",
    sample_return: "bg-rose-500/10 border-rose-500/20 text-rose-400/80",
    international: "bg-teal-500/10 border-teal-500/20 text-teal-400/80",
    commercial: "bg-orange-500/10 border-orange-500/20 text-orange-400/80",
  };
  return styles[category] ?? "bg-white/5 border-white/5 text-slate-500";
}

function formatCategoryLabel(category: string): string {
  return category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/* =========================================================
 * API helpers
 * =======================================================*/

async function fetchNewsList(): Promise<NewsSummary[]> {
  const resp = await fetch("/api/mars-news");
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.news ?? [];
}

async function fetchNewsDetail(date: string): Promise<NewsDetail | null> {
  const resp = await fetch(`/api/mars-news/${date}`);
  if (!resp.ok) return null;
  const data = await resp.json();
  return {
    date: data.date ?? date,
    items: data.items ?? [],
    trend_summary: data.trend_summary ?? "",
    summary_md: data.summary_md ?? "",
    categories: data.categories ?? [],
  };
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

function NewsListItem({
  item,
  isSelected,
  onClick,
}: {
  item: NewsSummary;
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
        <span className="text-[9px] text-slate-600 font-mono">{item.item_count} items</span>
      </div>
      <div className={`text-sm font-medium leading-snug ${isSelected ? "text-white" : "text-slate-300"}`}>
        {item.trend_summary
          ? item.trend_summary.slice(0, 80) + (item.trend_summary.length > 80 ? "..." : "")
          : "Mars News Update"}
      </div>
      {item.categories && item.categories.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {item.categories.slice(0, 3).map((cat) => (
            <span
              key={cat}
              className={`text-[9px] rounded-full px-1.5 py-0.5 border ${getCategoryStyle(cat)}`}
            >
              {formatCategoryLabel(cat)}
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

function MetadataHeader({ news }: { news: NewsDetail }) {
  return (
    <div className="mb-6 rounded-xl border border-border-dark bg-white/[0.02] p-5">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/15 border border-primary/30 px-3 py-1 text-xs font-bold text-primary">
          <span className="material-symbols-outlined text-sm">newspaper</span>
          Mars News
        </span>
        <span className="text-xs text-slate-500 font-mono">{formatDate(news.date)}</span>
        <span className="text-xs text-slate-600 font-mono">
          {news.items.length} item{news.items.length !== 1 ? "s" : ""}
        </span>
      </div>

      {news.categories && news.categories.length > 0 && (
        <div className="mb-2">
          <span className="text-[9px] uppercase text-slate-600 tracking-widest mr-2">Categories</span>
          {news.categories.map((cat) => (
            <span
              key={cat}
              className={`inline-block text-[10px] rounded-full px-2 py-0.5 mr-1 mb-1 border ${getCategoryStyle(cat)}`}
            >
              {formatCategoryLabel(cat)}
            </span>
          ))}
        </div>
      )}

      {news.trend_summary && (
        <div>
          <span className="text-[9px] uppercase text-slate-600 tracking-widest mr-2">Trend</span>
          <span className="text-xs text-slate-400">{news.trend_summary}</span>
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <span className="material-symbols-outlined text-5xl text-slate-700 mb-4">newspaper</span>
      <h3 className="text-lg font-bold text-slate-400 mb-2">No Mars news yet</h3>
      <p className="text-sm text-slate-600 max-w-md">
        Mars news digests are generated automatically each day. They aggregate and
        summarize the latest developments in Mars exploration, missions, and research.
      </p>
      <p className="text-xs text-slate-700 mt-3 font-mono">
        Run: python backend/scripts/mars_news.py
      </p>
    </div>
  );
}

/* =========================================================
 * Main Page Component
 * =======================================================*/

export default function MarsNewsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const tab: "news" | "research" = searchParams.get("tab") === "research" ? "research" : "news";
  const setTab = (t: "news" | "research") => {
    const next = new URLSearchParams(searchParams);
    if (t === "research") next.set("tab", "research"); else next.delete("tab");
    setSearchParams(next, { replace: true });
  };

  const [news, setNews] = useState<NewsSummary[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [detail, setDetail] = useState<NewsDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [mobileListOpen, setMobileListOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);

  // Derived: unique categories from all news summaries
  const uniqueCategories = useMemo(
    () => Array.from(new Set(news.flatMap((n) => n.categories ?? []).filter(Boolean))),
    [news]
  );

  // Derived: filtered news list
  const filteredNews = useMemo(() => {
    let list = news;
    if (categoryFilter) {
      list = list.filter((n) => n.categories?.includes(categoryFilter));
    }
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      list = list.filter(
        (n) =>
          n.trend_summary?.toLowerCase().includes(q) ||
          n.categories?.some((c) => c.toLowerCase().includes(q))
      );
    }
    return list;
  }, [news, categoryFilter, searchQuery]);

  // Load news list
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchNewsList().then((list) => {
      if (cancelled) return;
      setNews(list);
      if (list.length > 0 && !selectedDate) {
        setSelectedDate(list[0]?.date ?? null);
      }
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load selected news detail
  useEffect(() => {
    if (!selectedDate) return;
    let cancelled = false;
    setDetailLoading(true);
    fetchNewsDetail(selectedDate).then((d) => {
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

  const renderedHtml = detail?.summary_md
    ? DOMPurify.sanitize(newsMarkdownToHtml(detail.summary_md))
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
            <span className="material-symbols-outlined text-primary text-xl">
              {tab === "research" ? "science" : "newspaper"}
            </span>
            <h1 className="text-lg font-bold tracking-tight">
              {tab === "research" ? "Mars Research" : "Mars News"}
            </h1>
          </div>
          {/* Tab switcher */}
          <div className="flex items-center gap-1 ml-4 bg-white/5 rounded-lg p-1 border border-border-dark">
            <button
              onClick={() => setTab("news")}
              className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                tab === "news" ? "bg-primary/20 text-primary" : "text-slate-400 hover:text-white"
              }`}
            >
              News
            </button>
            <button
              onClick={() => setTab("research")}
              className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                tab === "research" ? "bg-primary/20 text-primary" : "text-slate-400 hover:text-white"
              }`}
            >
              Research
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-500">
          {tab === "news" && (
            <span className="font-mono">{news.length} edition{news.length !== 1 ? "s" : ""}</span>
          )}
        </div>
      </header>

      {/* Research tab — render embedded MarsResearchPage and skip the rest */}
      {tab === "research" ? (
        <MarsResearchPage embedded />
      ) : (
        <NewsBody
          news={news}
          loading={loading}
          detail={detail}
          detailLoading={detailLoading}
          selectedDate={selectedDate}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          categoryFilter={categoryFilter}
          setCategoryFilter={setCategoryFilter}
          uniqueCategories={uniqueCategories}
          filteredNews={filteredNews}
          handleSelect={handleSelect}
          mobileListOpen={mobileListOpen}
          setMobileListOpen={setMobileListOpen}
          renderedHtml={renderedHtml}
        />
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// News body extracted as a sub-component for the tabbed layout
// ───────────────────────────────────────────────────────────
type NewsBodyProps = {
  news: NewsSummary[];
  loading: boolean;
  detail: NewsDetail | null;
  detailLoading: boolean;
  selectedDate: string | null;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  categoryFilter: string | null;
  setCategoryFilter: (c: string | null) => void;
  uniqueCategories: string[];
  filteredNews: NewsSummary[];
  handleSelect: (date: string) => void;
  mobileListOpen: boolean;
  setMobileListOpen: (v: boolean | ((p: boolean) => boolean)) => void;
  renderedHtml: string;
};

function NewsBody({
  news, loading, detail, detailLoading, selectedDate,
  searchQuery, setSearchQuery, categoryFilter, setCategoryFilter,
  uniqueCategories, filteredNews, handleSelect,
  mobileListOpen, setMobileListOpen, renderedHtml,
}: NewsBodyProps) {
  return (
    <>
      {/* Mobile list toggle */}
      <div className="md:hidden flex items-center border-b border-border-dark px-4 py-2 bg-bg-dark">
        <button
          onClick={() => setMobileListOpen((p) => !p)}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white"
        >
          <span className="material-symbols-outlined text-base">
            {mobileListOpen ? "expand_less" : "expand_more"}
          </span>
          {selectedDate ? `${formatDate(selectedDate)}` : "Select edition"}
        </button>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — news list */}
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
                placeholder="Search news..."
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

            {/* Category filter badges */}
            {uniqueCategories.length > 0 && (
              <div className="flex flex-wrap gap-1">
                <button
                  onClick={() => setCategoryFilter(null)}
                  className={`text-[9px] rounded-full px-2 py-0.5 border transition-colors ${
                    categoryFilter === null
                      ? "bg-primary/20 border-primary/40 text-primary font-bold"
                      : "bg-white/5 border-white/5 text-slate-500 hover:border-slate-500"
                  }`}
                >
                  All
                </button>
                {uniqueCategories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setCategoryFilter(categoryFilter === cat ? null : cat)}
                    className={`text-[9px] rounded-full px-2 py-0.5 border transition-colors truncate max-w-[140px] ${
                      categoryFilter === cat
                        ? "bg-primary/20 border-primary/40 text-primary font-bold"
                        : "bg-white/5 border-white/5 text-slate-500 hover:border-slate-500"
                    }`}
                  >
                    {formatCategoryLabel(cat)}
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
          ) : filteredNews.length === 0 ? (
            <div className="p-4 text-sm text-slate-600">No news found.</div>
          ) : (
            <div className="p-2 space-y-1">
              {filteredNews.map((n) => (
                <NewsListItem
                  key={n.date}
                  item={n}
                  isSelected={n.date === selectedDate}
                  onClick={() => handleSelect(n.date)}
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
                <span className="text-sm">Loading news...</span>
              </div>
            </div>
          ) : detail ? (
            <div className="max-w-3xl mx-auto px-6 py-8">
              <MetadataHeader news={detail} />

              {/* News item cards */}
              {detail.items && detail.items.length > 0 && (
                <div className="space-y-4 mb-8">
                  {detail.items.map((item, idx) => (
                    <div key={idx} className="rounded-xl border border-border-dark bg-white/[0.02] p-5">
                      <h3 className="text-base font-bold text-white mb-2">
                        {item.url ? (
                          <a href={item.url} target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">
                            {item.title}
                            <span className="material-symbols-outlined text-xs ml-1 align-middle text-slate-500">open_in_new</span>
                          </a>
                        ) : item.title}
                      </h3>
                      <div className="flex items-center gap-2 mb-3 text-xs text-slate-500">
                        <span>{item.source}</span>
                        <span className="w-1 h-1 rounded-full bg-slate-600" />
                        <span>{item.date}</span>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed mb-3">{item.summary}</p>
                      <div className="flex flex-wrap gap-2">
                        {item.significance && (
                          <span className="text-[10px] rounded-full px-2 py-0.5 border bg-primary/10 border-primary/20 text-primary/80">
                            {item.significance}
                          </span>
                        )}
                        {item.category && (
                          <span className={`text-[10px] rounded-full px-2 py-0.5 border ${getCategoryStyle(item.category)}`}>
                            {formatCategoryLabel(item.category)}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Summary markdown */}
              {renderedHtml && (
                <article
                  className="prose-marslab"
                  dangerouslySetInnerHTML={{ __html: renderedHtml }}
                />
              )}
            </div>
          ) : news.length === 0 && !loading ? (
            <EmptyState />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-600 text-sm">
              Select a news edition from the sidebar
            </div>
          )}
        </main>
      </div>
    </>
  );
}
