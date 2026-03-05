import type { CustomDataset } from "../../../pages/MainPage";

type CustomDataTabProps = {
  dataset: CustomDataset;
};

export default function CustomDataTab({ dataset }: CustomDataTabProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-fuchsia-400">upload_file</span>
        <span className="text-sm font-bold">Custom Dataset</span>
      </div>

      <div className="rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/10 p-3">
        <span className="text-[10px] uppercase text-fuchsia-400/70 block mb-1">Name</span>
        <span className="text-sm text-white font-medium">{dataset.name}</span>
      </div>

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <MetaRow label="Dataset ID" value={dataset.id} />
        <MetaRow label="CRS" value={dataset.crs} />
        {dataset.crs_warning && (
          <div className="text-[10px] text-amber-400 bg-amber-500/10 rounded p-2 border border-amber-500/30">
            {dataset.crs_warning}
          </div>
        )}
        <MetaRow label="Dimensions" value={`${dataset.width} x ${dataset.height}`} />
        <MetaRow label="Bands" value={String(dataset.bands)} />
        <MetaRow label="Data Type" value={dataset.dtype} />
        {dataset.nodata !== null && (
          <MetaRow label="NoData" value={String(dataset.nodata)} />
        )}
      </div>

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">
          Geographic Bounds
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <BoundsItem label="West" value={dataset.bounds.west} />
          <BoundsItem label="East" value={dataset.bounds.east} />
          <BoundsItem label="South" value={dataset.bounds.south} />
          <BoundsItem label="North" value={dataset.bounds.north} />
        </div>
      </div>

      <div className="text-[9px] text-slate-500">
        Uploaded: {new Date(dataset.created_at).toLocaleString()}
      </div>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[10px] uppercase text-slate-500">{label}</span>
      <span className="font-mono text-xs">{value}</span>
    </div>
  );
}

function BoundsItem({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <span className="text-[9px] text-slate-500 block">{label}</span>
      <span className="font-mono text-xs">{value.toFixed(4)}°</span>
    </div>
  );
}
