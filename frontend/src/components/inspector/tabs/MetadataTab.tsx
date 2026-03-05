import type { InspectorContext, ProductMetadata } from "../types";
import ProductIdBadge from "../widgets/ProductIdBadge";
import CoordinateDisplay from "../widgets/CoordinateDisplay";
import QuickviewImage from "../widgets/QuickviewImage";
import DownloadLinks from "../widgets/DownloadLinks";
import TRR3MineralSection from "./TRR3MineralSection";

type MetadataTabProps = {
  selected: InspectorContext;
  metadata: ProductMetadata | null;
  metadataLoading: boolean;
  hasHighResData?: boolean;
  onOpenMineralSequence?: (obsId: string) => void;
};

export default function MetadataTab({
  selected,
  metadata,
  metadataLoading,
  hasHighResData,
  onOpenMineralSequence,
}: MetadataTabProps) {
  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3";

  return (
    <div className="space-y-4">
      {/* Instrument + Status Badge */}
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">
          {isCRISM ? "spectrum" : "satellite_alt"}
        </span>
        <span className="text-sm font-bold">{selected.instrument}</span>
        {isHiRISE && (
          hasHighResData ? (
            <span className="ml-auto flex items-center gap-1 rounded-full bg-green-500/20 border border-green-500/40 px-2 py-0.5 text-[9px] font-bold uppercase text-green-400">
              <span className="material-symbols-outlined" style={{ fontSize: 10 }}>check_circle</span>
              Full Res
            </span>
          ) : (
            <span className="ml-auto flex items-center gap-1 rounded-full bg-amber-500/20 border border-amber-500/40 px-2 py-0.5 text-[9px] font-bold uppercase text-amber-400">
              <span className="material-symbols-outlined" style={{ fontSize: 10 }}>photo_camera</span>
              Quickview Only
            </span>
          )
        )}
      </div>

      {/* Title */}
      {(selected.title || metadata?.title) && (
        <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
          <span className="text-[10px] uppercase text-primary/70 block mb-1">Title</span>
          <span className="text-sm text-white font-medium">
            {selected.title || metadata?.title}
          </span>
        </div>
      )}

      {/* Product ID + Coordinates */}
      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <ProductIdBadge productId={selected.productId} instrument={selected.instrument} />
        <CoordinateDisplay lat={selected.lat} lon={selected.lon} />

        {/* Enriched metadata from GeoJSON index */}
        {metadataLoading && (
          <div className="flex items-center gap-2 text-slate-500 text-[11px]">
            <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
            Loading metadata…
          </div>
        )}

        {metadata && (
          <div className="space-y-2 pt-2 border-t border-border-dark/50">
            {metadata.observationDate && (
              <MetaRow label="Date" value={metadata.observationDate} />
            )}
            {metadata.resolution != null && (
              <MetaRow label="Resolution" value={`${metadata.resolution.toFixed(2)} m/px`} />
            )}
            {metadata.mapScale != null && (
              <MetaRow label="Map Scale" value={`${metadata.mapScale.toFixed(2)} m/px`} />
            )}
            {metadata.solarIncidence != null && (
              <MetaRow label="Solar Inc." value={`${metadata.solarIncidence.toFixed(1)}°`} />
            )}
            {metadata.emissionAngle != null && (
              <MetaRow label="Emission" value={`${metadata.emissionAngle.toFixed(1)}°`} />
            )}
            {metadata.phaseAngle != null && (
              <MetaRow label="Phase" value={`${metadata.phaseAngle.toFixed(1)}°`} />
            )}
            {(metadata.imageLines != null || metadata.imageSamples != null) && (
              <MetaRow
                label="Image Size"
                value={`${metadata.imageSamples ?? "?"} × ${metadata.imageLines ?? "?"} px`}
              />
            )}
            {metadata.orbitNumber != null && (
              <MetaRow label="Orbit" value={String(metadata.orbitNumber)} />
            )}
            {metadata.productType && (
              <MetaRow label="Type" value={metadata.productType} />
            )}
            {metadata.sensorId && (
              <MetaRow label="Sensor" value={metadata.sensorId} />
            )}
            {metadata.wavelengthRange && (
              <MetaRow label="λ Range" value={metadata.wavelengthRange} />
            )}
          </div>
        )}
      </div>

      {/* Quickview Image */}
      <div>
        <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-primary">
          Quickview
        </h4>
        <div className="overflow-hidden rounded-lg border border-border-dark">
          <QuickviewImage productId={selected.productId} instrument={selected.instrument} />
        </div>
      </div>

      {/* PDS Download Links */}
      {(isHiRISE || isCRISM) && (
        <div>
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-primary">
            PDS Downloads
          </h4>
          <DownloadLinks productId={selected.productId} instrument={selected.instrument} />
        </div>
      )}

      {/* TRR3 Mineral Classification */}
      {selected.instrument === "CRISM_TRR3" && (
        <TRR3MineralSection
          obsId={selected.productId.replace(/_\d{2}$/, "")}
          onOpenMineralSequence={onOpenMineralSequence}
        />
      )}
    </div>
  );
}

/* ── Metadata Row ── */
function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="text-[10px] uppercase text-slate-500">{label}</span>
      <span className="font-mono text-xs text-white">{value}</span>
    </div>
  );
}
