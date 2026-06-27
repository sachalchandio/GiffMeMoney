/**
 * App — the route tree.
 *
 * The global providers (QueryClient → Theme → Router → Auth) are mounted once in
 * `main.tsx`; this file owns only the routing layer so there is a single Router /
 * AuthProvider in the tree. Public routes (`/login`, `/signup`) render the branded
 * auth screens; every product page lives behind `<ProtectedRoute>` inside the
 * `<AppLayout>` shell and is code-split via `lazy()`. Unknown paths redirect home.
 */

import { Suspense, lazy, type ComponentType } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { ProtectedRoute } from '@/lib/auth';

/* ---- Public (auth) screens — eager so the first paint is instant ---- */
import LoginPage from '@/pages/LoginPage';
import SignupPage from '@/pages/SignupPage';

/* ---- Product pages — code-split (written by parallel agents) ---- */
const DashboardPage = lazy(() => import('@/pages/DashboardPage'));
const RecommendationsPage = lazy(() => import('@/pages/RecommendationsPage'));
const AssetDetailPage = lazy(() => import('@/pages/AssetDetailPage'));
const StrategyLabPage = lazy(() => import('@/pages/StrategyLabPage'));
const PortfolioPage = lazy(() => import('@/pages/PortfolioPage'));
const InvestPage = lazy(() => import('@/pages/InvestPage'));
const ScreenerPage = lazy(() => import('@/pages/ScreenerPage'));
const AutoTraderPage = lazy(() => import('@/pages/AutoTraderPage'));
const SpeedLabPage = lazy(() => import('@/pages/SpeedLabPage'));
const RealTimePage = lazy(() => import('@/pages/RealTimePage'));
const GuidePage = lazy(() => import('@/pages/GuidePage'));

/** Centered spinner shown while a lazily-imported page chunk loads. */
function RouteFallback(): JSX.Element {
  return (
    <div className="flex min-h-[60vh] items-center justify-center text-muted" role="status" aria-live="polite">
      <span
        className="h-7 w-7 animate-spin rounded-full border-2 border-border border-t-primary"
        aria-hidden
      />
      <span className="sr-only">Loading…</span>
    </div>
  );
}

/** Wrap a lazily-loaded page element in the shared Suspense fallback. */
function lazyPage(Page: ComponentType): JSX.Element {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Page />
    </Suspense>
  );
}

export default function App(): JSX.Element {
  return (
    <Routes>
      {/* Public auth routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />

      {/* Authenticated app shell + nested pages */}
      <Route
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={lazyPage(DashboardPage)} />
        <Route path="guide" element={lazyPage(GuidePage)} />
        <Route path="recommendations" element={lazyPage(RecommendationsPage)} />
        <Route path="asset/:symbol" element={lazyPage(AssetDetailPage)} />
        <Route path="strategies" element={lazyPage(StrategyLabPage)} />
        <Route path="portfolio" element={lazyPage(PortfolioPage)} />
        <Route path="invest" element={lazyPage(InvestPage)} />
        <Route path="auto-trader" element={lazyPage(AutoTraderPage)} />
        <Route path="speed-lab" element={lazyPage(SpeedLabPage)} />
        <Route path="real-time" element={lazyPage(RealTimePage)} />
        <Route path="screener" element={lazyPage(ScreenerPage)} />
      </Route>

      {/* Unknown → home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
