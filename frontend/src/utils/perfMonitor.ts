/**
 * Performance monitoring utilities for MarsLab
 *
 * Usage:
 *   import { perf } from '../utils/perfMonitor';
 *
 *   perf.start('footprint-render');
 *   // ... code to measure
 *   perf.end('footprint-render');  // logs duration
 *
 * Enable/disable:
 *   perf.enable();
 *   perf.disable();
 */

const ENABLED_KEY = 'marslab-perf-monitor-enabled';

class PerfMonitor {
  private enabled: boolean;
  private timings: Map<string, number> = new Map();
  private history: Map<string, number[]> = new Map();

  constructor() {
    try {
      this.enabled = localStorage.getItem(ENABLED_KEY) === 'true';
    } catch {
      this.enabled = false;
    }
  }

  enable(): void {
    this.enabled = true;
    try { localStorage.setItem(ENABLED_KEY, 'true'); } catch {}

  }

  disable(): void {
    this.enabled = false;
    try { localStorage.setItem(ENABLED_KEY, 'false'); } catch {}

  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Start timing an operation
   */
  start(label: string): void {
    if (!this.enabled) return;
    this.timings.set(label, performance.now());
  }

  /**
   * End timing and log the result
   */
  end(label: string): number {
    if (!this.enabled) return 0;

    const startTime = this.timings.get(label);
    if (startTime === undefined) {

      return 0;
    }

    const duration = performance.now() - startTime;
    this.timings.delete(label);

    // Track history
    const hist = this.history.get(label) || [];
    hist.push(duration);
    if (hist.length > 100) hist.shift(); // Keep last 100
    this.history.set(label, hist);



    return duration;
  }

  /**
   * Log a one-time measurement
   */
  mark(_label: string, _duration: number): void {
    if (!this.enabled) return;
  }

  /**
   * Get average timing for a label
   */
  getAverage(label: string): number | null {
    const hist = this.history.get(label);
    if (!hist || hist.length === 0) return null;
    return hist.reduce((a, b) => a + b, 0) / hist.length;
  }

  /**
   * Print summary of all tracked operations
   */
  summary(): void {
    // No-op: console output removed for production
  }

  /**
   * Clear history
   */
  clear(): void {
    this.timings.clear();
    this.history.clear();

  }
}

// Singleton instance
export const perf = new PerfMonitor();

// Expose to window for console access
if (typeof window !== 'undefined') {
  (window as any).perf = perf;
}
