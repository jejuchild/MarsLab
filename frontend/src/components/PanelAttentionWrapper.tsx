/**
 * PanelAttentionWrapper — Thin wrapper that applies attention-pulse CSS animation.
 *
 * Wraps right panel content. When `isActive` is true, the `.panel-attention-pulse`
 * class is applied and the CSS animation plays once (1.2 s teal glow).
 * The `pulseKey` prop forces React to remount the element so the animation restarts.
 */

import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  isActive: boolean;
  pulseKey: number;
  onPulseEnd: () => void;
}

export default function PanelAttentionWrapper({ children, isActive, pulseKey, onPulseEnd }: Props) {
  const divRef = useRef<HTMLDivElement>(null);

  // Restart animation by toggling the class (no key-based remount).
  useEffect(() => {
    if (!isActive || !divRef.current) return;
    const el = divRef.current;
    // Force animation restart: remove class → reflow → add class
    el.classList.remove("panel-attention-pulse");
    void el.offsetWidth; // trigger reflow
    el.classList.add("panel-attention-pulse");
    const timer = setTimeout(onPulseEnd, 1300);
    return () => clearTimeout(timer);
  }, [isActive, pulseKey, onPulseEnd]);

  return (
    <div
      ref={divRef}
      className="h-full"
      onAnimationEnd={onPulseEnd}
    >
      {children}
    </div>
  );
}
