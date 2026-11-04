/**
 * Radargram ↔ map cursor link. The radargram emits the ground position under
 * the mouse many times per second; going through React state would re-render
 * the whole page each time, so the map subscribes directly.
 */
export type GroundPoint = { lat: number; lon: number } | null;

const listeners = new Set<(p: GroundPoint) => void>();

export function emitSharadHover(p: GroundPoint): void {
  for (const l of listeners) l(p);
}

export function onSharadHover(listener: (p: GroundPoint) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
