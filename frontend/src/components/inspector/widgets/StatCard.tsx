type StatCardProps = {
  label: string;
  value: string;
  highlight?: boolean;
  icon?: string;
};

export default function StatCard({ label, value, highlight = false, icon }: StatCardProps) {
  return (
    <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2.5">
      <div className="flex items-center gap-1">
        {icon && (
          <span className="material-symbols-outlined text-[12px] text-slate-500">
            {icon}
          </span>
        )}
        <span className="text-[9px] uppercase text-slate-500">{label}</span>
      </div>
      <div
        className={`font-mono text-sm font-bold ${
          highlight ? "text-blue-400" : "text-white"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
