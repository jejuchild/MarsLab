import { useEffect, useState, useRef } from "react";

/* ================================================================
 * Easter Egg Components for MarsLab
 * ================================================================
 * Lightweight fun surprises. All dismissible via click, Escape,
 * or auto-dismiss after a timeout.
 * ================================================================ */

/* ─────────────────────────────────────────────────────────────────
 * #2 - Terraform Mode Overlay
 * Trigger: Type "terraform" in search bar
 * ───────────────────────────────────────────────────────────────── */
interface TerraformOverlayProps {
  active: boolean;
}

export function TerraformOverlay({ active }: TerraformOverlayProps) {
  const [phase, setPhase] = useState<"idle" | "fadein" | "hold" | "fadeout">("idle");

  useEffect(() => {
    if (!active) {
      setPhase("idle");
      return;
    }

    // Kick off animation sequence
    setPhase("fadein");
    const t1 = setTimeout(() => setPhase("hold"), 500);
    const t2 = setTimeout(() => setPhase("fadeout"), 9500);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [active]);

  if (phase === "idle") return null;

  const opacity = phase === "fadeout" ? 0 : 1;

  return (
    <div
      className="fixed inset-0 z-[10] pointer-events-none flex flex-col items-center"
      style={{
        opacity,
        transition: phase === "fadein" ? "opacity 0.5s ease-in" : "opacity 0.5s ease-out",
      }}
    >
      {/* Color tint overlay */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 50%, rgba(30, 130, 200, 0.22) 0%, rgba(40, 160, 80, 0.18) 35%, rgba(30, 130, 200, 0.12) 65%, transparent 100%)",
        }}
      />

      {/* Label */}
      <div className="mt-20 px-4 py-2 rounded-full bg-emerald-500/20 border border-emerald-500/30 backdrop-blur-sm">
        <span
          className="text-emerald-400 text-xs font-bold tracking-[0.2em] uppercase"
          style={{ textShadow: "0 0 12px rgba(74, 222, 128, 0.5)" }}
        >
          TERRAFORM MODE ACTIVATED
        </span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
 * #3 - Curiosity Selfie Modal
 * Trigger: Click MarsLab logo 7 times rapidly
 * ───────────────────────────────────────────────────────────────── */
interface CuriositySelfieModalProps {
  onClose: () => void;
}

export function CuriositySelfieModal({ onClose }: CuriositySelfieModalProps) {
  const [imgLoaded, setImgLoaded] = useState(false);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Auto-dismiss after 15s
  useEffect(() => {
    const timer = setTimeout(onClose, 15000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative w-full max-w-md mx-4 bg-[#101622] rounded-lg border border-[#232f48] shadow-2xl overflow-hidden">
        {/* Rover Image */}
        <div className="relative h-64 bg-[#0a0f18] overflow-hidden">
          {!imgLoaded && (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="material-symbols-outlined text-4xl text-[#232f48] animate-pulse">
                smart_toy
              </span>
            </div>
          )}
          <img
            src="https://d2pn8kiwq2w21t.cloudfront.net/original_images/1-PIA24543.jpg"
            alt="Curiosity Rover Selfie"
            className={`w-full h-full object-cover transition-opacity duration-500 ${imgLoaded ? "opacity-100" : "opacity-0"}`}
            onLoad={() => setImgLoaded(true)}
            onError={() => setImgLoaded(true)}
          />
          <div className="absolute bottom-1 right-2 text-[8px] text-white/50">
            NASA/JPL-Caltech/MSSS
          </div>
        </div>

        {/* Speech Bubble */}
        <div className="relative px-6 py-5">
          <div
            className="absolute -top-2 left-8 w-0 h-0"
            style={{
              borderLeft: "8px solid transparent",
              borderRight: "8px solid transparent",
              borderBottom: "8px solid #1a2333",
            }}
          />
          <div className="bg-[#1a2333] rounded-lg px-4 py-3 border border-[#232f48]">
            <p className="text-[#92a4c9] italic text-sm leading-relaxed">
              "I'm still here. Sol 4,300+"
            </p>
            <p className="text-[#4a5a78] text-[10px] mt-2 font-mono">
              -- Curiosity, Gale Crater, Mars
            </p>
          </div>
        </div>

        {/* Close */}
        <div className="px-6 pb-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-[11px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors"
          >
            Close
          </button>
        </div>

        {/* X button */}
        <button
          onClick={onClose}
          className="absolute top-2 right-2 p-1 rounded hover:bg-white/10 text-white/50 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined text-lg">close</span>
        </button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
 * #7 - Olympus Mons Height Comparison Panel
 * Trigger: Click on Olympus Mons summit area 3 times
 * ───────────────────────────────────────────────────────────────── */
interface OlympusMonsPanelProps {
  onClose: () => void;
}

export function OlympusMonsPanel({ onClose }: OlympusMonsPanelProps) {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimate(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Auto-dismiss after 12s
  useEffect(() => {
    const timer = setTimeout(onClose, 12000);
    return () => clearTimeout(timer);
  }, [onClose]);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const OLYMPUS_KM = 21.9;
  const EVEREST_KM = 8.849;
  const EVEREST_PCT = (EVEREST_KM / OLYMPUS_KM) * 100;

  return (
    <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-[150] w-80 bg-[#101622] border border-[#232f48] rounded-lg shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#232f48] bg-[#0a0f18]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-base text-orange-400">landscape</span>
          <span className="text-xs font-bold text-white">Height Comparison</span>
        </div>
        <button
          onClick={onClose}
          className="p-0.5 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined text-base">close</span>
        </button>
      </div>

      {/* Bars */}
      <div className="px-4 py-4 space-y-4">
        {/* Olympus Mons */}
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="text-[11px] font-medium text-orange-400">Olympus Mons</span>
            <span className="text-[11px] font-mono text-orange-300">{OLYMPUS_KM} km</span>
          </div>
          <div className="h-6 bg-[#1a2333] rounded overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-orange-600 to-orange-400 rounded flex items-center justify-end pr-2"
              style={{
                width: animate ? "100%" : "0%",
                transition: "width 1.5s ease-out",
              }}
            >
              <span className="text-[8px] font-bold text-white/80">MARS</span>
            </div>
          </div>
        </div>

        {/* Mount Everest */}
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="text-[11px] font-medium text-emerald-400">Mount Everest</span>
            <span className="text-[11px] font-mono text-emerald-300">{EVEREST_KM} km</span>
          </div>
          <div className="h-6 bg-[#1a2333] rounded overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded flex items-center justify-end pr-2"
              style={{
                width: animate ? `${EVEREST_PCT.toFixed(1)}%` : "0%",
                transition: "width 1.5s ease-out 0.3s",
              }}
            >
              <span className="text-[8px] font-bold text-white/80">EARTH</span>
            </div>
          </div>
        </div>
      </div>

      {/* Fun Fact Footer */}
      <div className="px-4 pb-3">
        <p className="text-[10px] text-[#6b7c9c] italic text-center">
          Olympus Mons is 2.5x taller than Everest -- and would cover most of France.
        </p>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
 * #7b - Olympus Mons Climber Animation
 * Trigger: Click on Olympus Mons summit area 7 times
 * ───────────────────────────────────────────────────────────────── */
interface OlympusMonsClimberProps {
  onClose: () => void;
}

// Mountain profile path points (shield volcano shape)
const MOUNTAIN_POINTS = [
  [0, 260], [30, 258], [60, 250], [90, 238], [120, 220],
  [150, 198], [180, 172], [210, 145], [240, 118], [260, 100],
  [280, 82], [300, 66], [315, 55], [325, 48], [335, 42],
  // Summit caldera dip
  [342, 40], [350, 38], [358, 36], [365, 35], [372, 34],
  [380, 35], [388, 36], [395, 38], [402, 40],
  // Descent
  [415, 48], [430, 58], [450, 75], [470, 95], [490, 118],
  [510, 145], [530, 172], [550, 198], [570, 220], [590, 238],
  [620, 250], [650, 258], [680, 260],
] as const;

// Climber walks along points 0 → summit (index ~18) over the animation
const CLIMB_PATH = MOUNTAIN_POINTS.slice(0, 20);

export function OlympusMonsClimber({ onClose }: OlympusMonsClimberProps) {
  const [climberIdx, setClimberIdx] = useState(0);
  const [reachedSummit, setReachedSummit] = useState(false);
  const [showFlag, setShowFlag] = useState(false);
  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Animate the climber stepping along the path
  useEffect(() => {
    animRef.current = setInterval(() => {
      setClimberIdx((prev) => {
        if (prev >= CLIMB_PATH.length - 1) {
          if (animRef.current) clearInterval(animRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, 300);
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, []);

  // Summit reached
  useEffect(() => {
    if (climberIdx >= CLIMB_PATH.length - 1) {
      setReachedSummit(true);
      setTimeout(() => setShowFlag(true), 400);
    }
  }, [climberIdx]);

  // Auto-dismiss after 15s
  useEffect(() => {
    const timer = setTimeout(onClose, 15000);
    return () => clearTimeout(timer);
  }, [onClose]);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const pathD =
    "M " + MOUNTAIN_POINTS.map(([x, y]) => `${x},${y}`).join(" L ");
  const cx = CLIMB_PATH[climberIdx][0];
  const cy = CLIMB_PATH[climberIdx][1];

  // Altitude labels along the left side
  const altitudes = [
    { y: 35, label: "21.9 km" },
    { y: 105, label: "15 km" },
    { y: 175, label: "10 km" },
    { y: 240, label: "5 km" },
  ];

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative w-full max-w-2xl mx-4 bg-[#101622] rounded-lg border border-[#232f48] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#232f48] bg-[#0a0f18]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-base text-orange-400">hiking</span>
            <span className="text-xs font-bold text-white">Climbing Olympus Mons</span>
            {reachedSummit && (
              <span className="text-[10px] text-emerald-400 font-bold ml-2 animate-pulse">SUMMIT REACHED!</span>
            )}
          </div>
          <button onClick={onClose} className="p-0.5 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors">
            <span className="material-symbols-outlined text-base">close</span>
          </button>
        </div>

        {/* Mountain SVG */}
        <div className="px-4 py-4">
          <svg viewBox="0 0 700 280" className="w-full h-auto">
            {/* Sky gradient */}
            <defs>
              <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#1a0a2e" />
                <stop offset="100%" stopColor="#0f1923" />
              </linearGradient>
              <linearGradient id="mountainGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#c2410c" />
                <stop offset="40%" stopColor="#9a3412" />
                <stop offset="100%" stopColor="#431407" />
              </linearGradient>
            </defs>

            {/* Background */}
            <rect x="0" y="0" width="700" height="280" fill="url(#skyGrad)" />

            {/* Stars */}
            {[
              [50, 20], [120, 45], [200, 15], [310, 30], [420, 12],
              [500, 40], [580, 18], [650, 35], [80, 60], [450, 55],
              [250, 50], [370, 22], [550, 48], [160, 38], [620, 25],
            ].map(([sx, sy], i) => (
              <circle key={i} cx={sx} cy={sy} r={0.8} fill="white" opacity={0.4 + (i % 3) * 0.2} />
            ))}

            {/* Altitude gridlines */}
            {altitudes.map(({ y, label }) => (
              <g key={y}>
                <line x1="40" y1={y} x2="680" y2={y} stroke="#232f48" strokeWidth="0.5" strokeDasharray="4 4" />
                <text x="5" y={y + 3} fill="#4a5a78" fontSize="8" fontFamily="monospace">{label}</text>
              </g>
            ))}

            {/* Mountain fill */}
            <path
              d={pathD + " L 680,260 L 0,260 Z"}
              fill="url(#mountainGrad)"
              opacity="0.9"
            />

            {/* Mountain outline */}
            <path d={pathD} fill="none" stroke="#ea580c" strokeWidth="1.5" opacity="0.6" />

            {/* Summit caldera label */}
            <text x="365" y="26" fill="#fb923c" fontSize="8" textAnchor="middle" fontWeight="bold">
              Summit Caldera
            </text>

            {/* Climber trail (footprints) */}
            {CLIMB_PATH.slice(0, climberIdx).map(([px, py], i) => (
              <circle
                key={i}
                cx={px}
                cy={py - 4}
                r={1}
                fill="#fbbf24"
                opacity={0.3 + (i / CLIMB_PATH.length) * 0.4}
              />
            ))}

            {/* Climber figure */}
            <g transform={`translate(${cx}, ${cy - 16})`}>
              {/* Body */}
              <circle cx="0" cy="0" r="4" fill="#fbbf24" />
              {/* Helmet */}
              <circle cx="0" cy="-1" r="3" fill="#f59e0b" stroke="#d97706" strokeWidth="0.5" />
              {/* Visor */}
              <rect x="-2" y="-2" width="4" height="2" rx="1" fill="#78350f" opacity="0.7" />
              {/* Body */}
              <line x1="0" y1="4" x2="0" y2="10" stroke="#fbbf24" strokeWidth="1.5" />
              {/* Legs - animate walking */}
              <line
                x1="0" y1="10"
                x2={climberIdx % 2 === 0 ? -3 : 2}
                y2="14"
                stroke="#fbbf24" strokeWidth="1.2"
              />
              <line
                x1="0" y1="10"
                x2={climberIdx % 2 === 0 ? 2 : -3}
                y2="14"
                stroke="#fbbf24" strokeWidth="1.2"
              />
              {/* Arm with ice axe */}
              <line x1="0" y1="6" x2="5" y2="2" stroke="#fbbf24" strokeWidth="1" />
              <line x1="5" y1="2" x2="7" y2="-1" stroke="#94a3b8" strokeWidth="0.8" />
            </g>

            {/* Flag at summit */}
            {showFlag && (
              <g transform="translate(365, 16)">
                <line x1="0" y1="0" x2="0" y2="16" stroke="#94a3b8" strokeWidth="1" />
                <polygon points="0,0 15,4 0,8" fill="#ef4444" opacity="0.9">
                  <animateTransform
                    attributeName="transform"
                    type="scale"
                    values="1,1;1.05,1;1,1"
                    dur="1s"
                    repeatCount="indefinite"
                  />
                </polygon>
                <text x="18" y="6" fill="#fbbf24" fontSize="7" fontWeight="bold">
                  CLAIMED!
                </text>
              </g>
            )}

            {/* Altitude indicator for climber */}
            <g>
              <text
                x={cx + 12}
                y={cy - 18}
                fill="#fbbf24"
                fontSize="7"
                fontFamily="monospace"
              >
                {((1 - (cy - 34) / 226) * 21.9).toFixed(1)} km
              </text>
            </g>

            {/* Ground line */}
            <line x1="0" y1="260" x2="700" y2="260" stroke="#232f48" strokeWidth="1" />
          </svg>
        </div>

        {/* Footer */}
        <div className="px-4 pb-3 flex items-center justify-between">
          <p className="text-[10px] text-[#6b7c9c] italic">
            {reachedSummit
              ? "You just climbed the tallest volcano in the solar system. Time for potatoes."
              : `Ascending... ${((climberIdx / (CLIMB_PATH.length - 1)) * 100).toFixed(0)}% to summit`}
          </p>
          {reachedSummit && (
            <span className="text-lg" role="img" aria-label="flag">&#127937;</span>
          )}
        </div>
      </div>
    </div>
  );
}
