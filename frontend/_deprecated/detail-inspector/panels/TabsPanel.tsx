import type { DetailItem } from "../../types/detail";
import TabButton from "../ui/TabButton";

type Props = {
  items: DetailItem[];
  activeProductId: string;
  onSelect: (productId: string) => void;
  onRemove: (productId: string) => void;
};

export default function TabsPanel({
  items,
  activeProductId,
  onSelect,
  onRemove,
}: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((item) => {
        const active = item.productId === activeProductId;
        return (
          <TabButton
            key={item.productId}
            productId={item.productId}
            instrument={item.instrument}
            active={active}
            onSelect={() => onSelect(item.productId)}
            onRemove={() => onRemove(item.productId)}
          />
        );
      })}
    </div>
  );
}
