import { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";

/* =========================================================
 * Tour step definition
 * =======================================================*/
interface TourStep {
  title: string;
  description: string;
  icon: string;
  /** Where to position the spotlight. null = center of screen. */
  target: "left-panel" | "right-panel" | "center" | null;
}

const TOUR_STEPS: TourStep[] = [
  {
    title: "Welcome to MarsLab",
    description:
      "MarsLab is a Mars orbital data analysis workbench. Explore multi-instrument datasets from MRO \u2014 CRISM mineral spectra, HiRISE imagery, SHARAD radar, and CTX context images \u2014 all in one unified 3D globe interface.",
    icon: "rocket_launch",
    target: null,
  },
  {
    title: "Load Instrument Data",
    description:
      "Open the Footprints section in the left panel and click \u2018Load\u2019 next to any instrument (try CRISM or HiRISE). This fetches observation footprints for the current map view. Toggle visibility with the instrument switches.",
    icon: "layers",
    target: "left-panel",
  },
  {
    title: "Navigate Mars",
    description:
      "Drag to pan, scroll to zoom, and use \u2018Fly To\u2019 to jump to named locations like Jezero Crater or Valles Marineris. Switch between 2D Map and 3D Globe views at the top of the left panel.",
    icon: "explore",
    target: "center",
  },
  {
    title: "Inspect Observations",
    description:
      "Click any footprint on the map to open the Inspector panel. View metadata, activate quickview image overlays, browse spectral data, compare products side-by-side, and download raw files from PDS.",
    icon: "info",
    target: "right-panel",
  },
  {
    title: "Explore Ice Detection",
    description:
      "The Ice Detection section combines five independent techniques \u2014 neutron spectroscopy, thermal inertia, radar surface, radar dielectric, and geomorphology \u2014 to map subsurface water ice. Toggle methods and adjust depth ranges.",
    icon: "ac_unit",
    target: "left-panel",
  },
  {
    title: "Analysis & AI Tools",
    description:
      "Use Terrain Analysis tools for slope analysis, elevation profiles, and measurements. AI-powered tools include Agentic AI for autonomous investigation, Landing Site Reports for region comparison, and Guided Workflows for step-by-step research.",
    icon: "build",
    target: "left-panel",
  },
  {
    title: "Meet MARVIS",
    description:
      "Click the MARVIS button in the bottom-right to chat with your AI assistant. MARVIS can navigate the map, load instruments, search for products, and answer questions about Mars. You can also save Field Notes and share your current view via URL.",
    icon: "smart_toy",
    target: null,
  },
];

const LOCALSTORAGE_KEY = "marslab-tour-completed";

/* =========================================================
 * Spotlight positioning
 * =======================================================*/
function getSpotlightRect(target: TourStep["target"]): {
  top: number;
  left: number;
  width: number;
  height: number;
} | null {
  if (!target || target === "center") return null;

  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const headerH = 56; // h-14 = 56px

  if (target === "left-panel") {
    // Approximate left panel area
    return {
      top: headerH,
      left: 0,
      width: Math.min(320, vw * 0.3),
      height: vh - headerH,
    };
  }

  if (target === "right-panel") {
    // Approximate right panel area
    const panelW = Math.min(384, vw * 0.3);
    return {
      top: headerH,
      left: vw - panelW,
      width: panelW,
      height: vh - headerH,
    };
  }

  return null;
}

/* =========================================================
 * Props
 * =======================================================*/
interface OnboardingTourProps {
  /** Force-open the tour (overrides localStorage). Used for "re-trigger" */
  forceOpen?: boolean;
  onComplete?: () => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function OnboardingTour({ forceOpen, onComplete }: OnboardingTourProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Check localStorage on mount
  useEffect(() => {
    if (forceOpen) {
      setIsOpen(true);
      setCurrentStep(0);
      return;
    }
    try {
      const completed = localStorage.getItem(LOCALSTORAGE_KEY);
      if (completed !== "true") {
        setIsOpen(true);
        setCurrentStep(0);
      }
    } catch {
      // localStorage unavailable, show tour
      setIsOpen(true);
      setCurrentStep(0);
    }
  }, [forceOpen]);

  const handleComplete = useCallback(() => {
    try {
      localStorage.setItem(LOCALSTORAGE_KEY, "true");
    } catch {
      // Ignore
    }
    setIsOpen(false);
    onComplete?.();
  }, [onComplete]);

  const handleSkip = useCallback(() => {
    handleComplete();
  }, [handleComplete]);

  const handleNext = useCallback(() => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      handleComplete();
    }
  }, [currentStep, handleComplete]);

  const handlePrev = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  }, [currentStep]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        handleSkip();
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        handleNext();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handlePrev();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, handleSkip, handleNext, handlePrev]);

  if (!isOpen) return null;

  const step = TOUR_STEPS[currentStep]!;
  const spotlight = getSpotlightRect(step.target);
  const isLastStep = currentStep === TOUR_STEPS.length - 1;
  const isFirstStep = currentStep === 0;

  // Card position: if spotlight is on left, card on right; if on right, card on left; otherwise center
  let cardPositionClass = "items-center justify-center";
  if (step.target === "left-panel") {
    cardPositionClass = "items-center justify-center md:justify-end md:pr-[15%]";
  } else if (step.target === "right-panel") {
    cardPositionClass = "items-center justify-center md:justify-start md:pl-[15%]";
  }

  const overlay = (
    <div className="fixed inset-0 z-[10000]" role="dialog" aria-modal="true" aria-label="Onboarding tour">
      {/* Overlay background with spotlight cutout via SVG mask */}
      <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: "none" }}>
        <defs>
          <mask id="tour-spotlight-mask">
            {/* White = visible overlay (darkened) */}
            <rect x="0" y="0" width="100%" height="100%" fill="white" />
            {/* Black = cutout (transparent) */}
            {spotlight && (
              <rect
                x={spotlight.left}
                y={spotlight.top}
                width={spotlight.width}
                height={spotlight.height}
                rx="8"
                ry="8"
                fill="black"
              />
            )}
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill="rgba(0, 0, 0, 0.75)"
          mask="url(#tour-spotlight-mask)"
          style={{ pointerEvents: "auto" }}
          onClick={handleSkip}
        />
      </svg>

      {/* Spotlight border glow */}
      {spotlight && (
        <div
          className="absolute rounded-lg border-2 border-primary/50 pointer-events-none"
          style={{
            top: spotlight.top,
            left: spotlight.left,
            width: spotlight.width,
            height: spotlight.height,
            boxShadow: "0 0 30px rgba(59, 130, 246, 0.15), inset 0 0 30px rgba(59, 130, 246, 0.05)",
          }}
        />
      )}

      {/* Step card */}
      <div className={`absolute inset-0 flex ${cardPositionClass} pointer-events-none`}>
        <div
          className="pointer-events-auto w-full max-w-md mx-4 animate-slideDown"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="rounded-xl border border-[#232f48] bg-[#0d1520] shadow-2xl shadow-black/60 overflow-hidden">
            {/* Card header with icon */}
            <div className="flex items-center gap-4 px-6 pt-6 pb-3">
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-primary/15 border border-primary/30">
                <span className="material-symbols-outlined text-2xl text-primary">
                  {step.icon}
                </span>
              </div>
              <div className="flex-1">
                <h3 className="text-white text-base font-bold">
                  {step.title}
                </h3>
                <span className="text-[10px] text-[#6b7c9c] font-mono uppercase tracking-wider">
                  Step {currentStep + 1} of {TOUR_STEPS.length}
                </span>
              </div>
            </div>

            {/* Description */}
            <div className="px-6 pb-4">
              <p className="text-[#92a4c9] text-sm leading-relaxed">
                {step.description}
              </p>
            </div>

            {/* Step dots */}
            <div className="flex items-center justify-center gap-2 pb-4">
              {TOUR_STEPS.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentStep(i)}
                  className={`w-2 h-2 rounded-full transition-all ${
                    i === currentStep
                      ? "bg-primary w-6"
                      : i < currentStep
                        ? "bg-primary/40"
                        : "bg-[#3a4a68]"
                  }`}
                  aria-label={`Go to step ${i + 1}`}
                />
              ))}
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-[#232f48] bg-[#0a0f18]">
              <button
                onClick={handleSkip}
                className="text-xs text-[#6b7c9c] hover:text-[#92a4c9] transition-colors"
              >
                Skip tour
              </button>
              <div className="flex items-center gap-2">
                {!isFirstStep && (
                  <button
                    onClick={handlePrev}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-[#3a4a68] transition-colors"
                  >
                    <span className="material-symbols-outlined text-sm">arrow_back</span>
                    Back
                  </button>
                )}
                <button
                  onClick={handleNext}
                  className="flex items-center gap-1 px-4 py-1.5 rounded-lg text-xs font-medium bg-primary/20 border border-primary/50 text-primary hover:bg-primary/30 transition-colors"
                >
                  {isLastStep ? "Get Started" : "Next"}
                  {!isLastStep && (
                    <span className="material-symbols-outlined text-sm">arrow_forward</span>
                  )}
                  {isLastStep && (
                    <span className="material-symbols-outlined text-sm">check</span>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(overlay, document.body);
}
