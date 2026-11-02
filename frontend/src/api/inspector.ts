/**
 * Inspector at-point client.
 *
 * Calls the backend /api/inspector/at-point aggregator that returns
 * products from all 4 lanes (SHARAD, CRISM, HIRISE, CTX) within a small
 * radius of a coordinate.
 */

export type Lane = "SHARAD" | "CRISM" | "HIRISE" | "CTX";

export type LaneVariant =
  | "standard"
  | "highres"
  | "trr3"
  | "image"
  | "dtm"
  | "mosaic";

export interface LaneProduct {
  product_id: string;
  title: string | null;
  lat: number | null;
  lon: number | null;
  variant: LaneVariant;
  /** Distance from the point to the footprint (0 when inside). */
  distance_km: number | null;
  /** Footprint contains the point (SHARAD tracks: passes within ~3 km). */
  covers?: boolean;
}

export interface AtPointResponse {
  lat: number;
  lon: number;
  radius_km: number;
  lanes: Record<Lane, LaneProduct[]>;
  counts: Record<Lane, number>;
}

const DEFAULT_RADIUS_KM = 10;

/**
 * Fetch all 4-lane products near a coordinate.
 *
 * @throws on network error or non-2xx HTTP response
 */
export async function fetchAtPoint(
  lat: number,
  lon: number,
  radiusKm: number = DEFAULT_RADIUS_KM
): Promise<AtPointResponse> {
  const url = `/api/inspector/at-point?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`;
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`inspector/at-point ${res.status}: ${res.statusText}`);
  }
  return (await res.json()) as AtPointResponse;
}

/**
 * Returns the lane that has the most products. Used to pick a default
 * active lane when the user clicks an empty point.
 */
export function pickDefaultLane(response: AtPointResponse): Lane {
  let best: Lane = "HIRISE";
  let bestCount = -1;
  for (const lane of ["HIRISE", "CRISM", "SHARAD", "CTX"] as Lane[]) {
    const c = response.counts[lane] ?? 0;
    if (c > bestCount) {
      best = lane;
      bestCount = c;
    }
  }
  return best;
}

/** Nearest TRR3 sensor pixel for a lat/lon (DDR-based); null if outside or unavailable. */
export async function fetchTrr3Pixel(
  productId: string,
  lat: number,
  lon: number,
): Promise<{ line: number; sample: number } | null> {
  try {
    const res = await fetch(
      `/api/crism-trr3/${encodeURIComponent(productId)}/pixel?lat=${lat}&lon=${lon}`,
    );
    if (!res.ok) return null;
    const px = (await res.json()) as { line: number; sample: number; inside: boolean };
    return px.inside ? { line: px.line, sample: px.sample } : null;
  } catch {
    return null;
  }
}
