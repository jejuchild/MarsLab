import { useEffect, useState } from "react";

interface CesiumLoadingPlaceholderProps {
  visible: boolean;
}

export default function CesiumLoadingPlaceholder({
  visible,
}: CesiumLoadingPlaceholderProps) {
  const [show, setShow] = useState(visible);

  // Fade-out: keep mounted for 500ms after visible becomes false
  useEffect(() => {
    if (visible) {
      setShow(true);
      return;
    }
    const timer = setTimeout(() => setShow(false), 500);
    return () => clearTimeout(timer);
  }, [visible]);

  if (!show) return null;

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-[#0a0e17] transition-opacity duration-500"
      style={{ opacity: visible ? 1 : 0 }}
    >
      <div className="flex flex-col items-center gap-4">
        {/* Mars icon */}
        <span className="material-symbols-outlined text-[48px] text-primary">
          public
        </span>

        {/* Title */}
        <span className="text-xl font-bold text-white">MarsLab</span>

        {/* Progress bar */}
        <div className="h-0.5 w-48 overflow-hidden rounded-full bg-surface-dark">
          <div
            className="h-full w-1/3 animate-pulse rounded-full"
            style={{
              background:
                "linear-gradient(90deg, transparent, rgba(59,130,246,0.5), transparent)",
            }}
          />
        </div>

        {/* Status text */}
        <span className="text-xs text-slate-500">
          Initializing 3D Globe…
        </span>
      </div>
    </div>
  );
}
