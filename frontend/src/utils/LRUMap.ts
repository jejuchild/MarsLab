export class LRUMap<K, V> extends Map<K, V> {
  private readonly maxSize: number;

  constructor(maxSize: number) {
    super();
    this.maxSize = maxSize;
  }

  override get(key: K): V | undefined {
    if (!super.has(key)) return undefined;
    const value = super.get(key);
    if (value === undefined) return undefined;
    super.delete(key);
    super.set(key, value);
    return value;
  }

  override set(key: K, value: V): this {
    if (super.has(key)) super.delete(key);
    super.set(key, value);
    if (super.size > this.maxSize) {
      const firstKey = super.keys().next().value as K | undefined;
      if (firstKey !== undefined) super.delete(firstKey);
    }
    return this;
  }
}
