import type { ReactNode } from "react";

interface AppShellProps {
  header: ReactNode;
  leftPanel: ReactNode;
  rightPanel?: ReactNode;
  children: ReactNode;
  /** Mobile mode: hide side panels, show full-screen map */
  isMobile?: boolean;
  /** Bottom navigation bar for mobile */
  mobileNav?: ReactNode;
}

/**
 * AppShell provides the main application layout structure.
 *
 * Desktop:
 * ┌─────────────────────────────────────────────────────┐
 * │                     Header                          │
 * ├──────────┬──────────────────────────┬───────────────┤
 * │   Left   │         Main             │    Right      │
 * │  Panel   │        Content           │    Panel      │
 * └──────────┴──────────────────────────┴───────────────┘
 *
 * Mobile:
 * ┌─────────────────────┐
 * │    Header (compact)  │
 * ├─────────────────────┤
 * │   Full-screen Map   │
 * ├─────────────────────┤
 * │   Bottom Nav Bar    │
 * └─────────────────────┘
 */
export default function AppShell({
  header,
  leftPanel,
  rightPanel,
  children,
  isMobile = false,
  mobileNav,
}: AppShellProps) {
  if (isMobile) {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-bg-dark">
        <header className="z-50 shrink-0">{header}</header>
        <main className="relative flex-1 overflow-hidden">{children}</main>
        {mobileNav && (
          <nav className="z-50 shrink-0 mobile-bottom-nav">{mobileNav}</nav>
        )}
      </div>
    );
  }

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
