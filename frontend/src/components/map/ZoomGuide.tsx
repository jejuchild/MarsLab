interface ZoomGuideProps {
  visible: boolean;
}

export default function ZoomGuide({ visible }: ZoomGuideProps) {
  if (!visible) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
      <div className="flex flex-col items-center gap-1.5 rounded-full bg-black/40 px-5 py-2.5 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-white/50">
            zoom_in
          </span>
          <span className="text-sm text-white/60">
            Zoom in to see footprints
          </span>
        </div>
        <span className="text-[10px] text-white/30">
          Use scroll wheel or + key
        </span>
      </div>
    </div>
  );
}
