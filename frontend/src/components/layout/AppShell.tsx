import type { ReactNode } from "react";

interface AppShellProps {
  header: ReactNode;
  leftPanel: ReactNode;
  rightPanel?: ReactNode;
  children: ReactNode;
}

/**
 * AppShell provides the main application layout structure.
 *
 * Layout:
 * ┌─────────────────────────────────────────────────────┐
 * │                     Header                          │
 * ├──────────┬──────────────────────────┬───────────────┤
 * │          │                          │               │
 * │   Left   │         Main             │    Right      │
 * │  Panel   │        Content           │    Panel      │
 * │          │                          │               │
 * └──────────┴──────────────────────────┴───────────────┘
 */
export default function AppShell({
  header,
  leftPanel,
  rightPanel,
  children,
}: AppShellProps) {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-bg-dark">
      {/* Header */}
      <header className="z-50 shrink-0">{header}</header>

      {/* Main content area */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Left Panel */}
        <aside className="z-10 shrink-0">{leftPanel}</aside>

        {/* Central Content (Map Canvas) */}
        <main className="relative flex-1 overflow-hidden">{children}</main>

        {/* Right Panel (Inspector) */}
        {rightPanel && (
          <aside className="z-10 shrink-0">{rightPanel}</aside>
        )}
      </div>
    </div>
  );
}
