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
      "MarsLab is a Mars orbital data analysis workbench. Explore multi-instrument datasets from MRO including CRISM mineral spectra, HiRISE high-resolution imagery, SHARAD subsurface radar, and CTX context images -- all in one unified interface.",
    icon: "rocket_launch",
    target: null,
  },
  {
    title: "Load Instruments",
    description:
      "Use the Layers panel on the left to load instrument footprints onto the map. Toggle CRISM, HiRISE, SHARAD, CTX, and more. Click the \"Load\" button next to each instrument to fetch footprints for the current map view.",
    icon: "layers",
    target: "left-panel",
  },
  {
    title: "Inspect Products",
    description:
      "Click any footprint on the map to open the Inspector panel on the right. View product metadata, activate quickview overlays, browse spectral data, and download raw files. Use the search bar to find specific product IDs.",
    icon: "info",
    target: "right-panel",
  },
  {
    title: "AI Analysis",
    description:
      "MarsLab includes AI-powered analysis tools. Open the Agentic AI panel for autonomous multi-instrument investigation, generate Landing Site Reports comparing regions, or use Guided Workflows for step-by-step research procedures.",
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
