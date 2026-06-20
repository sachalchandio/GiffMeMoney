/**
 * AppLayout — the authenticated shell: a fixed/collapsible desktop Sidebar, a
 * mobile drawer (overlay), the glassy sticky TopBar, and a max-width content
 * container that renders nested routes via <Outlet />. Starts the market
 * WebSocket so live prices stream on every page. Collapse state is persisted.
 */

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Outlet } from 'react-router-dom';
import { X } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { Button } from '@/components/ui/Button';
import { useMarketSocket } from '@/hooks/useMarketSocket';
import { cn } from '@/lib/utils';

const COLLAPSE_KEY = 'giff_sidebar_collapsed';

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === '1';
  } catch {
    return false;
  }
}

export interface AppLayoutProps {
  /** Override the routed content (defaults to <Outlet />). */
  children?: ReactNode;
}

export function AppLayout({ children }: AppLayoutProps): JSX.Element {
  // Keep the live price store warm across the whole authenticated app.
  useMarketSocket();

  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed());
  const [drawerOpen, setDrawerOpen] = useState(false);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try {
        localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0');
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  // Close the drawer on Escape.
  useEffect(() => {
    if (!drawerOpen) return undefined;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setDrawerOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [drawerOpen]);

  return (
    <div className="min-h-screen bg-bg text-text">
      {/* Desktop sidebar (fixed) */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-20 hidden border-r border-border bg-surface transition-[width] duration-200 lg:block',
          collapsed ? 'w-16' : 'w-60',
        )}
      >
        <Sidebar collapsed={collapsed} onToggleCollapsed={toggleCollapsed} />
      </aside>

      {/* Mobile drawer */}
      {drawerOpen && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm animate-fade-in"
            onClick={() => setDrawerOpen(false)}
            aria-hidden
          />
          <div className="absolute inset-y-0 left-0 w-72 max-w-[80%] animate-slide-in-left border-r border-border bg-surface">
            <div className="absolute right-2 top-2.5 z-10">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setDrawerOpen(false)}
                aria-label="Close navigation"
                className="h-8 w-8"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <Sidebar drawer onNavigate={() => setDrawerOpen(false)} />
          </div>
        </div>
      )}

      {/* Main column */}
      <div className={cn('flex min-h-screen flex-col transition-[padding] duration-200', collapsed ? 'lg:pl-16' : 'lg:pl-60')}>
        <TopBar onOpenSidebar={() => setDrawerOpen(true)} />
        <main className="flex-1">
          <div className="mx-auto w-full max-w-content px-3 py-4 sm:px-4 lg:px-6 lg:py-6">
            {children ?? <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
}
