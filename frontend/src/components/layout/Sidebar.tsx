/**
 * Sidebar — primary navigation. On desktop it's a fixed rail that can collapse to
 * an icon-only strip; on mobile it slides in as a drawer (the AppLayout owns the
 * open/overlay state). Active route is highlighted via NavLink. Colors via tokens.
 */

import { NavLink } from 'react-router-dom';
import {
  ChevronLeft,
  LayoutDashboard,
  ListChecks,
  FlaskConical,
  PieChart,
  Wallet,
  Bot,
  Table2,
  Zap,
  Radio,
  type LucideIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  /** Only match the exact path (used for the index route). */
  end?: boolean;
}

/** The frozen navigation set (FRONTEND.md). */
export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/recommendations', label: 'Recommendations', icon: ListChecks },
  { to: '/strategies', label: 'Strategy Lab', icon: FlaskConical },
  { to: '/portfolio', label: 'Portfolio', icon: PieChart },
  { to: '/invest', label: 'Invest', icon: Wallet },
  { to: '/auto-trader', label: 'Auto-Trader', icon: Bot },
  { to: '/real-time', label: 'Real-Time', icon: Radio },
  { to: '/speed-lab', label: 'Speed Lab', icon: Zap },
  { to: '/screener', label: 'Screener', icon: Table2 },
];

export interface SidebarProps {
  /** Icon-only collapsed rail (desktop only). */
  collapsed?: boolean;
  /** Toggle collapse (desktop). */
  onToggleCollapsed?: () => void;
  /** Called when a link is chosen (mobile drawer closes itself). */
  onNavigate?: () => void;
  /** Render in drawer mode (always expanded, full labels). */
  drawer?: boolean;
}

export function Sidebar({
  collapsed = false,
  onToggleCollapsed,
  onNavigate,
  drawer = false,
}: SidebarProps): JSX.Element {
  const showLabels = drawer || !collapsed;

  return (
    <div className="flex h-full flex-col">
      {/* Brand */}
      <div
        className={cn(
          'flex h-14 items-center border-b border-border px-3',
          showLabels ? 'justify-between' : 'justify-center',
        )}
      >
        <NavLink
          to="/"
          end
          onClick={onNavigate}
          className="flex items-center gap-2 font-semibold tracking-tight text-text"
        >
          <span
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary text-base font-extrabold text-white shadow-soft"
            aria-hidden
          >
            $
          </span>
          {showLabels && <span className="text-[0.9375rem]">GiffMeMoney</span>}
        </NavLink>
        {!drawer && showLabels && onToggleCollapsed && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapsed}
            aria-label="Collapse sidebar"
            className="h-7 w-7"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 overflow-y-auto p-2" aria-label="Primary">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            title={!showLabels ? label : undefined}
            className={({ isActive }) =>
              cn(
                'group flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium tracking-tight transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
                !showLabels && 'justify-center px-0',
                isActive
                  ? 'bg-primary/12 text-primary'
                  : 'text-muted hover:bg-surface-2 hover:text-text',
              )
            }
          >
            <Icon className="h-[1.125rem] w-[1.125rem] shrink-0" aria-hidden />
            {showLabels && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Expand affordance when collapsed */}
      {!drawer && collapsed && onToggleCollapsed && (
        <div className="border-t border-border p-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapsed}
            aria-label="Expand sidebar"
            className="mx-auto rotate-180"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Footer disclaimer */}
      {showLabels && (
        <div className="border-t border-border p-3">
          <p className="text-[0.625rem] leading-relaxed text-muted">
            Educational simulation. Not financial advice.
          </p>
        </div>
      )}
    </div>
  );
}
