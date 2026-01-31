import { useEffect, useState } from "react";

export function useDetailState() {
  const [expanded, setExpanded] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [pin, setPin] = useState<{ x: number; y: number } | null>(null);

  return {
    expanded,
    setExpanded,
    activeId,
    setActiveId,
    pin,
    setPin,
  };
}
