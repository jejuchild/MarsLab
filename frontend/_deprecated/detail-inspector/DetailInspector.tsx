import { useMemo } from "react";
import type { DetailItem } from "./types";

import Header from "./ui/Header";
import TabsPanel from "./panels/TabsPanel";

import HiRISEDetailLayout from "./layouts/HiRISEDetailLayout";
import CRISMDetailLayout from "./layouts/CRISMDetailLayout";

import { useDetailState } from "./hooks/useDetailState";
import { useHiRISEAnalysis } from "./hooks/useHiRISEAnalysis";

/* =========================================================
 * Props
 * =======================================================*/
type Props = {
  items: DetailItem[];
  activeId: string | null;
  onSelect: (productId: string) => void;
  onRemove: (productId: string) => void;
};

export default function DetailInspector({
  items,
  activeId,
  onSelect,
  onRemove,
}: Props) {
  const { expanded, setExpanded, pin, setPin } = useDetailState(activeId);

  const hiRISEAnalysis = useHiRISEAnalysis();

  const activeItem = useMemo(() => {
    if (items.length === 0) return null;
    return items.find((v) => v.productId === activeId) ?? items[0];
  }, [items, activeId]);

  if (!activeItem) return null;

  const isHiRISE = activeItem.instrument === "HIRISE";
  const isCRISM = activeItem.instrument === "CRISM";

  return (
    <div
      className={`fixed inset-x-0 bottom-0 z-[9998] flex flex-col border-t border-border-dark bg-bg-dark text-white transition-all duration-300 ease-out ${
        expanded ? "h-[92vh]" : "h-[12vh]"
      }`}
    >
      {/* Header */}
      <Header expanded={expanded} onToggle={() => setExpanded((v) => !v)} />

      {/* Content */}
      <div className="flex flex-1 flex-col overflow-hidden p-3">
        {/* Tabs */}
        <TabsPanel
          items={items}
          activeProductId={activeItem.productId}
          onSelect={onSelect}
          onRemove={onRemove}
        />

        {/* Expanded Content */}
        {expanded && (
          <div className="mt-3 flex flex-1 overflow-hidden rounded-lg border border-border-dark bg-surface-dark/30">
            {isHiRISE && (
              <HiRISEDetailLayout
                productId={activeItem.productId}
                pin={pin}
                setPin={setPin}
                analysis={hiRISEAnalysis}
              />
            )}

            {isCRISM && <CRISMDetailLayout productId={activeItem.productId} />}
          </div>
        )}
      </div>
    </div>
  );
}
