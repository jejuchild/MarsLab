// src/api/hirise.ts

export type HiRISEPixelResponse = {
  dn: number[][];
  reflectance?: number[][];
  min_dn: number;
  max_dn: number;
  mean_dn: number;
};

// ✅ src/api/hirise.ts (정답)
export async function fetchHiRISEWindow(
  productId: string,
  x: number,
  y: number,
  halfSize: number
) {
  const params = new URLSearchParams({
    productId,
    x: String(x),
    y: String(y),
    halfSize: String(halfSize),
  });

  const url = `/hirise/window_xy?${params.toString()}`;

  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HiRISE window fetch failed: ${res.status}\n${text}`);
  }

  return res.json();
}
