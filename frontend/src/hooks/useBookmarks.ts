import { useState, useCallback, useEffect } from "react";

export interface MapBookmark {
  id: string;
  name: string;
  lat: number;
  lon: number;
  height: number; // camera height in meters
  createdAt: string; // ISO date string
}

const STORAGE_KEY = "marslab_bookmarks";
const MAX_BOOKMARKS = 50;

function loadBookmarks(): MapBookmark[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function saveBookmarks(bookmarks: MapBookmark[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
  } catch {
    // localStorage might be full or unavailable
  }
}

function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback
  return `bm_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export default function useBookmarks() {
  const [bookmarks, setBookmarks] = useState<MapBookmark[]>(loadBookmarks);

  // Sync to localStorage on every change
  useEffect(() => {
    saveBookmarks(bookmarks);
  }, [bookmarks]);

  const addBookmark = useCallback((name: string, lat: number, lon: number, height: number) => {
    setBookmarks((prev) => {
      const newBookmark: MapBookmark = {
        id: generateId(),
        name,
        lat,
        lon,
        height,
        createdAt: new Date().toISOString(),
      };
      const updated = [newBookmark, ...prev];
      // Enforce max cap — drop oldest
      if (updated.length > MAX_BOOKMARKS) {
        return updated.slice(0, MAX_BOOKMARKS);
      }
      return updated;
    });
  }, []);

  const removeBookmark = useCallback((id: string) => {
    setBookmarks((prev) => prev.filter((b) => b.id !== id));
  }, []);

  const renameBookmark = useCallback((id: string, name: string) => {
    setBookmarks((prev) =>
      prev.map((b) => (b.id === id ? { ...b, name } : b)),
    );
  }, []);

  return { bookmarks, addBookmark, removeBookmark, renameBookmark };
}
