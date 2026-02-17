/**
 * API client for Aqueous Mineral Sequence Mapper.
 */

export interface TransectPoint {
  position_idx: number;
  row: number;
  col: number;
  mineral_id?: number | null;
  mineral_name?: string | null;
  geochem_group?: string | null;
  confidence?: number | null;
}

export interface MineralTransition {
  position_idx: number;
  from_group: string;
  to_group: string;
  from_mineral: string;
  to_mineral: string;
}

export interface SequenceMatch {
  environment: string;
  matched_groups: string[];
  confidence: number;
}

export interface MineralSequenceSummary {
  obs_id: string;
  total_transect_points: number;
  classified_points: number;
  classification_rate: number;
  n_transitions: number;
  dominant_group?: string | null;
  n_groups_present: number;
  matched_environments: string[];
  mean_confidence?: number | null;
}

export interface MineralSequenceParameters {
  obs_id: string;
  transect_direction: string;
  transect_offset: number;
}

export interface MineralSequenceResult {
  success: boolean;
  error?: string | null;
  summary?: MineralSequenceSummary | null;
  transect: TransectPoint[];
  transitions: MineralTransition[];
  sequence_matches: SequenceMatch[];
  group_histogram: Record<string, number>;
  parameters?: MineralSequenceParameters | null;
}

export async function fetchMineralSequence(
  obsId: string,
  transectDirection: string = "NS",
  transectOffset: number = 0.5,
): Promise<MineralSequenceResult> {
  const params = new URLSearchParams({
    obs_id: obsId,
    transect_direction: transectDirection,
    transect_offset: transectOffset.toString(),
  });

  const res = await fetch(`/api/mineral-sequence/analyze?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Mineral sequence analysis failed (${res.status})`);
  }
  return res.json();
}

export function getMineralSequenceCsvUrl(
  obsId: string,
  transectDirection: string = "NS",
  transectOffset: number = 0.5,
): string {
  const params = new URLSearchParams({
    obs_id: obsId,
    transect_direction: transectDirection,
    transect_offset: transectOffset.toString(),
  });
  return `/api/mineral-sequence/export_csv?${params}`;
}
