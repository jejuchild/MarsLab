export type SHARADPopupData = {
  productId: string;
  quickviewUrl: string;
  startLat: number;
  startLon: number;
  stopLat: number;
  stopLon: number;
};

type SHARADPopupOverlayProps = {
  popup: SHARADPopupData | null;
  onClose: () => void;
};

export default function SHARADPopupOverlay({ popup, onClose }: SHARADPopupOverlayProps) {
  if (!popup) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="relative max-w-4xl max-h-[90vh] bg-[#101622] rounded-lg border border-[#232f48] shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48] bg-[#0a0f18]">
          <div>
            <h3 className="text-white font-bold text-sm">{popup.productId}</h3>
            <p className="text-[#92a4c9] text-[10px] mt-0.5">
              SHARAD Radargram: ({popup.startLat.toFixed(2)}°, {popup.startLon.toFixed(2)}°) → ({popup.stopLat.toFixed(2)}°, {popup.stopLon.toFixed(2)}°)
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-[#232f48] transition-colors text-[#92a4c9] hover:text-white"
            aria-label="Close SHARAD popup"
          >
            <span className="material-symbols-outlined text-xl">close</span>
          </button>
        </div>

        <div className="p-4 overflow-auto max-h-[calc(90vh-120px)]">
          <img
            src={popup.quickviewUrl}
            alt={`SHARAD ${popup.productId}`}
            className="max-w-full h-auto"
            style={{ imageRendering: "crisp-edges" }}
          />
        </div>

        <div className="px-4 py-3 border-t border-[#232f48] bg-[#0a0f18] flex justify-end gap-2">
          <button
            disabled
            className="px-3 py-1.5 text-[11px] font-medium bg-[#1a2333] border border-[#232f48] rounded text-[#6b7c9c] cursor-not-allowed"
            title="High-resolution data not available yet"
          >
            Activate High-Res Image (Coming Soon)
          </button>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-[11px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors"
            aria-label="Close SHARAD popup"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
