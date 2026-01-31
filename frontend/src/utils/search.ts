export interface SearchResult {
  item: string;
  matchStart: number;
  matchEnd: number;
  isPrefix: boolean;
}

/**
 * Search through an array of strings with ranking.
 * Prefix matches rank higher than substring matches.
 * Returns up to `limit` results, case-insensitive.
 */
export function search(
  items: string[],
  query: string,
  limit: number = 10
): SearchResult[] {
  if (!query.trim()) {
    return [];
  }

  const lowerQuery = query.toLowerCase();
  const results: SearchResult[] = [];

  for (const item of items) {
    const lowerItem = item.toLowerCase();
    const matchIndex = lowerItem.indexOf(lowerQuery);

    if (matchIndex !== -1) {
      results.push({
        item,
        matchStart: matchIndex,
        matchEnd: matchIndex + query.length,
        isPrefix: matchIndex === 0,
      });
    }
  }

  // Sort: prefix matches first, then by match position, then alphabetically
  results.sort((a, b) => {
    // Prefix matches rank highest
    if (a.isPrefix && !b.isPrefix) return -1;
    if (!a.isPrefix && b.isPrefix) return 1;

    // Within same category, sort by match position (earlier is better)
    if (a.matchStart !== b.matchStart) {
      return a.matchStart - b.matchStart;
    }

    // Finally, alphabetical order
    return a.item.localeCompare(b.item);
  });

  return results.slice(0, limit);
}

/**
 * Returns an array of segments for rendering highlighted text.
 * Each segment has `text` and `highlight` (boolean).
 */
export function highlightMatch(
  item: string,
  matchStart: number,
  matchEnd: number
): { text: string; highlight: boolean }[] {
  if (matchStart < 0 || matchEnd > item.length || matchStart >= matchEnd) {
    return [{ text: item, highlight: false }];
  }

  const segments: { text: string; highlight: boolean }[] = [];

  if (matchStart > 0) {
    segments.push({ text: item.slice(0, matchStart), highlight: false });
  }

  segments.push({ text: item.slice(matchStart, matchEnd), highlight: true });

  if (matchEnd < item.length) {
    segments.push({ text: item.slice(matchEnd), highlight: false });
  }

  return segments;
}

/**
 * Convenience function: search and return results with highlight segments.
 */
export function searchWithHighlight(
  items: string[],
  query: string,
  limit: number = 10
): { item: string; segments: { text: string; highlight: boolean }[]; isPrefix: boolean }[] {
  const results = search(items, query, limit);

  return results.map((result) => ({
    item: result.item,
    segments: highlightMatch(result.item, result.matchStart, result.matchEnd),
    isPrefix: result.isPrefix,
  }));
}
