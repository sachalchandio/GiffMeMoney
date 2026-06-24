/**
 * TopBar — the glassy, sticky header. Holds the mobile menu trigger, a symbol
 * search (Enter → asset detail), the live connection dot, the theme toggle, and
 * the signed-in user with a logout menu. Colors via semantic tokens only.
 */

import { useEffect, useRef, useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, Menu, Search } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { BrokerStatusBadge } from '@/components/domain/BrokerStatusBadge';
import { ConnDot } from './ConnDot';
import { ThemeToggle } from './ThemeToggle';
import { useAuth } from '@/lib/auth';
import { cn, initials } from '@/lib/utils';

export interface TopBarProps {
  /** Open the mobile navigation drawer. */
  onOpenSidebar: () => void;
}

export function TopBar({ onOpenSidebar }: TopBarProps): JSX.Element {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [query, setQuery] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return undefined;
    const onClick = (e: MouseEvent): void => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [menuOpen]);

  const onSearch = (e: FormEvent): void => {
    e.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (symbol) {
      navigate(`/asset/${encodeURIComponent(symbol)}`);
      setQuery('');
    }
  };

  return (
    <header className="glass sticky top-0 z-30 border-b border-border">
      <div className="flex h-14 items-center gap-2 px-3 sm:px-4 lg:px-6">
        {/* Mobile menu */}
        <Button
          variant="ghost"
          size="icon"
          onClick={onOpenSidebar}
          aria-label="Open navigation"
          className="lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </Button>

        {/* Search */}
        <form onSubmit={onSearch} className="relative max-w-md flex-1" role="search">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
            aria-hidden
          />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search symbol…"
            aria-label="Search by symbol"
            className={cn(
              'h-9 w-full rounded-xl border border-border bg-surface-2/60 pl-9 pr-3 text-sm text-text',
              'placeholder:text-muted transition-colors hover:border-muted',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
            )}
          />
        </form>

        <div className="flex items-center gap-1 sm:gap-2">
          {/* Data: Simulated|Live + Broker: Paper|LIVE (display-only; hidden on
              very narrow widths to keep the header uncluttered). */}
          <div className="hidden md:flex">
            <BrokerStatusBadge />
          </div>
          <div className="hidden sm:flex">
            <ConnDot />
          </div>
          <div className="sm:hidden">
            <ConnDot showLabel={false} />
          </div>
          <ThemeToggle />

          {/* User menu */}
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              onClick={() => setMenuOpen((o) => !o)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              className={cn(
                'flex items-center gap-2 rounded-xl py-1 pl-1 pr-2 text-sm transition-colors hover:bg-surface-2',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
              )}
            >
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-accent/15 text-xs font-semibold text-accent">
                {initials(user?.name ?? 'Guest')}
              </span>
              <span className="hidden max-w-[8rem] truncate font-medium text-text md:block">
                {user?.name ?? 'Guest'}
              </span>
            </button>

            {menuOpen && (
              <div
                role="menu"
                className="absolute right-0 mt-1.5 w-52 origin-top-right animate-fade-in rounded-xl border border-border bg-surface p-1 shadow-pop"
              >
                <div className="px-3 py-2">
                  <p className="truncate text-sm font-medium text-text">{user?.name ?? 'Guest'}</p>
                  <p className="truncate text-xs text-muted">{user?.email ?? ''}</p>
                </div>
                <div className="my-1 border-t border-border" />
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    logout();
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-text transition-colors hover:bg-surface-2"
                >
                  <LogOut className="h-4 w-4 text-muted" aria-hidden />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
