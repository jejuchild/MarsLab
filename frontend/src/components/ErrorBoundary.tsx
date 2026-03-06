import React from 'react';
import type { ErrorInfo, ReactNode } from 'react';

type ErrorBoundaryProps = {
  children: ReactNode;
  /** Optional fallback UI. If not provided, default recovery UI is shown. */
  fallback?: ReactNode;
  /** Scope label for error logging (e.g. "MapView", "Inspector") */
  scope?: string;
};

type ErrorBoundaryState = {
  hasError: boolean;
  error: Error | null;
  errorCount: number;
};

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorCount: 0 };
  }

  public static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    const scope = this.props.scope || 'unknown';
    console.error(`[ErrorBoundary:${scope}]`, error.message, errorInfo.componentStack);
  }

  private handleRetry = (): void => {
    this.setState((prev) => ({
      hasError: false,
      error: null,
      errorCount: prev.errorCount + 1,
    }));
  };

  private handleReload = (): void => {
    window.location.reload();
  };

  public override render(): ReactNode {
    if (this.state.hasError) {
      // If custom fallback provided, use it
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorCount } = this.state;
      const isChunkError = error?.message?.includes('Loading chunk') ||
        error?.message?.includes('Failed to fetch dynamically imported module') ||
        error?.message?.includes('Importing a module script failed');

      // After 3 retries, suggest hard reload
      if (errorCount >= 3) {
        return (
          <div className="flex flex-col items-center justify-center h-full min-h-[200px] gap-4 p-8 text-center">
            <div className="text-red-400 text-lg font-medium">
              Unable to recover
            </div>
            <p className="text-slate-400 text-sm max-w-md">
              This section encountered a persistent error. Try refreshing the page.
            </p>
            {error && (
              <pre className="text-xs text-slate-600 max-w-lg overflow-auto bg-slate-900/50 rounded p-2">
                {error.message}
              </pre>
            )}
            <button
              onClick={this.handleReload}
              className="px-4 py-2 text-sm rounded bg-red-500/20 text-red-300 border border-red-500/30 hover:bg-red-500/30 transition-colors"
            >
              Reload page
            </button>
          </div>
        );
      }

      return (
        <div className="flex flex-col items-center justify-center h-full min-h-[200px] gap-4 p-8 text-center">
          <div className="text-amber-400 text-lg font-medium">
            {isChunkError ? 'Failed to load component' : 'Something went wrong'}
          </div>
          <p className="text-slate-400 text-sm max-w-md">
            {isChunkError
              ? 'A component failed to load. This may be due to a network issue or a new deployment.'
              : 'An unexpected error occurred in this section.'}
          </p>
          {error && (
            <pre className="text-xs text-slate-600 max-w-lg overflow-auto bg-slate-900/50 rounded p-2">
              {error.message}
            </pre>
          )}
          <div className="flex gap-3">
            <button
              onClick={this.handleRetry}
              className="px-4 py-2 text-sm rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
            >
              Try again
            </button>
            <button
              onClick={this.handleReload}
              className="px-4 py-2 text-sm rounded bg-white/5 text-slate-400 border border-white/10 hover:bg-white/10 transition-colors"
            >
              Reload page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
