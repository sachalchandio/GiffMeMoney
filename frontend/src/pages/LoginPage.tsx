/**
 * LoginPage — branded (light/dark) sign-in screen (AUTH.md).
 *
 * Two-pane on desktop (brand panel + form), single column on mobile. Validates
 * email + password client-side, surfaces API errors inline, and offers a
 * "Use demo account" shortcut that logs in with the seeded demo credentials and
 * navigates home. Redirects already-authenticated users away from the page.
 */

import { useState, type FormEvent, type ReactNode } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  ArrowRight,
  Eye,
  EyeOff,
  LineChart,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { ThemeToggle } from '@/components/layout/ThemeToggle';
import { DEMO_CREDENTIALS, useAuth } from '@/lib/auth';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';

/** Pull a redirect target out of router state (set by ProtectedRoute). */
interface FromState {
  from?: { pathname?: string };
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/* ------------------------------------------------------------------ */
/* Shared branded auth chrome (kept local to the auth pages)           */
/* ------------------------------------------------------------------ */

const HIGHLIGHTS: ReadonlyArray<{ icon: JSX.Element; title: string; body: string }> = [
  {
    icon: <LineChart className="h-4 w-4" aria-hidden />,
    title: '18+ quant models, blended',
    body: 'CAPM, Fama–French, Monte Carlo and more fused into one composite score.',
  },
  {
    icon: <Sparkles className="h-4 w-4" aria-hidden />,
    title: 'Where to invest, and why',
    body: 'Expected return across five horizons with bull / base / bear bands.',
  },
  {
    icon: <ShieldCheck className="h-4 w-4" aria-hidden />,
    title: 'Sandbox by design',
    body: 'A simulated wallet and market — explore freely, no real money moves.',
  },
];

/** The emerald brand mark used across the auth screens. */
export function BrandMark({ className }: { className?: string }): JSX.Element {
  return (
    <span
      className={cn(
        'flex items-center justify-center rounded-2xl bg-primary font-extrabold text-white shadow-soft',
        className,
      )}
      aria-hidden
    >
      $
    </span>
  );
}

/**
 * The full-screen branded scaffold: a marketing panel (desktop only) beside the
 * form column. Used by both Login and Signup so the two screens stay identical.
 */
export function AuthLayout({
  heading,
  subheading,
  footer,
  children,
}: {
  heading: string;
  subheading: string;
  footer: ReactNode;
  children: ReactNode;
}): JSX.Element {
  return (
    <div className="min-h-screen bg-bg text-text lg:grid lg:grid-cols-[1.05fr_minmax(0,1fr)]">
      {/* Brand / marketing panel */}
      <aside className="relative hidden overflow-hidden bg-surface lg:flex lg:flex-col lg:justify-between lg:p-10 xl:p-12">
        <div
          className="pointer-events-none absolute inset-0 opacity-90"
          style={{
            background:
              'radial-gradient(60% 50% at 15% 10%, color-mix(in srgb, var(--primary) 22%, transparent), transparent 70%),' +
              'radial-gradient(55% 45% at 95% 95%, color-mix(in srgb, var(--accent) 20%, transparent), transparent 70%)',
          }}
          aria-hidden
        />
        <div className="relative flex items-center gap-2.5">
          <BrandMark className="h-9 w-9 text-lg" />
          <span className="text-lg font-semibold tracking-tight">GiffMeMoney</span>
        </div>

        <div className="relative max-w-md">
          <h2 className="text-2xl font-semibold leading-tight tracking-tight xl:text-3xl">
            Intelligent investment advisory, backed by real quant finance.
          </h2>
          <ul className="mt-7 flex flex-col gap-4">
            {HIGHLIGHTS.map((h) => (
              <li key={h.title} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  {h.icon}
                </span>
                <span>
                  <span className="block text-sm font-medium text-text">{h.title}</span>
                  <span className="block text-sm text-muted">{h.body}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>

        <p className="relative text-xs text-muted">
          Educational sandbox · not investment advice · no real funds are moved.
        </p>
      </aside>

      {/* Form column */}
      <main className="flex min-h-screen flex-col px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 lg:hidden">
            <BrandMark className="h-8 w-8 text-base" />
            <span className="font-semibold tracking-tight">GiffMeMoney</span>
          </div>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>

        <div className="flex flex-1 items-center justify-center py-8">
          <div className="w-full max-w-sm animate-slide-up">
            <h1 className="text-2xl font-semibold tracking-tight">{heading}</h1>
            <p className="mt-1.5 text-sm text-muted">{subheading}</p>
            {children}
            <div className="mt-6 text-center text-sm text-muted">{footer}</div>
          </div>
        </div>
      </main>
    </div>
  );
}

/** A labelled text input with a leading icon, matching the control tokens. */
export function Field({
  id,
  label,
  icon,
  error,
  trailing,
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  icon: JSX.Element;
  error?: string;
  trailing?: JSX.Element;
}): JSX.Element {
  return (
    <label htmlFor={id} className="block">
      <span className="mb-1.5 block text-sm font-medium text-text">{label}</span>
      <span className="relative block">
        <span
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted"
          aria-hidden
        >
          {icon}
        </span>
        <input
          id={id}
          aria-invalid={error ? true : undefined}
          className={cn(
            'h-11 w-full rounded-xl border bg-surface-2/60 pl-10 pr-3 text-sm text-text',
            'placeholder:text-muted transition-colors hover:border-muted',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
            trailing && 'pr-11',
            error ? 'border-danger' : 'border-border',
          )}
          {...rest}
        />
        {trailing && <span className="absolute right-2 top-1/2 -translate-y-1/2">{trailing}</span>}
      </span>
      {error && (
        <span className="mt-1 block text-xs text-danger" role="alert">
          {error}
        </span>
      )}
    </label>
  );
}

/* ------------------------------------------------------------------ */
/* Login page                                                          */
/* ------------------------------------------------------------------ */

export default function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, loginDemo, isAuthenticated, isLoading } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<'form' | 'demo' | null>(null);

  const redirectTo = (location.state as FromState | null)?.from?.pathname ?? '/';

  // Bounce already-authenticated users straight to the app.
  if (!isLoading && isAuthenticated) {
    return <Navigate to={redirectTo} replace />;
  }

  function validate(): boolean {
    const next: { email?: string; password?: string } = {};
    if (!EMAIL_RE.test(email.trim())) next.email = 'Enter a valid email address.';
    if (password.length < 1) next.password = 'Enter your password.';
    setFieldErrors(next);
    return Object.keys(next).length === 0;
  }

  function describeError(err: unknown): string {
    if (err instanceof ApiError) {
      if (err.status === 0) return 'Cannot reach the server. Is the backend running?';
      return err.detail || 'Sign in failed. Please try again.';
    }
    return 'Sign in failed. Please try again.';
  }

  async function onSubmit(e: FormEvent): Promise<void> {
    e.preventDefault();
    setFormError(null);
    if (!validate()) return;
    setSubmitting('form');
    try {
      await login({ email: email.trim(), password });
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setFormError(describeError(err));
    } finally {
      setSubmitting(null);
    }
  }

  async function onDemo(): Promise<void> {
    setFormError(null);
    setFieldErrors({});
    setEmail(DEMO_CREDENTIALS.email);
    setPassword(DEMO_CREDENTIALS.password);
    setSubmitting('demo');
    try {
      await loginDemo();
      navigate('/', { replace: true });
    } catch (err) {
      setFormError(describeError(err));
    } finally {
      setSubmitting(null);
    }
  }

  const busy = submitting !== null;

  return (
    <AuthLayout
      heading="Welcome back"
      subheading="Sign in to see where to invest and why."
      footer={
        <>
          New here?{' '}
          <Link
            to="/signup"
            className="font-medium text-primary underline-offset-4 hover:underline"
          >
            Create an account
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="mt-6 flex flex-col gap-4" noValidate>
        {formError && (
          <div
            className="flex items-start gap-2 rounded-xl border border-danger/40 bg-danger/10 px-3 py-2.5 text-sm text-danger"
            role="alert"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>{formError}</span>
          </div>
        )}

        <Field
          id="login-email"
          label="Email"
          type="email"
          autoComplete="email"
          inputMode="email"
          placeholder="you@example.com"
          icon={<Mail className="h-4 w-4" />}
          value={email}
          error={fieldErrors.email}
          disabled={busy}
          onChange={(e) => setEmail(e.target.value)}
        />

        <Field
          id="login-password"
          label="Password"
          type={showPassword ? 'text' : 'password'}
          autoComplete="current-password"
          placeholder="••••••••"
          icon={<Lock className="h-4 w-4" />}
          value={password}
          error={fieldErrors.password}
          disabled={busy}
          onChange={(e) => setPassword(e.target.value)}
          trailing={
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowPassword((s) => !s)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
              tabIndex={-1}
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          }
        />

        <Button
          type="submit"
          size="lg"
          fullWidth
          loading={submitting === 'form'}
          disabled={busy}
          rightIcon={<ArrowRight className="h-4 w-4" />}
        >
          Sign in
        </Button>
      </form>

      <div className="my-5 flex items-center gap-3 text-xs text-muted">
        <span className="h-px flex-1 bg-border" />
        or
        <span className="h-px flex-1 bg-border" />
      </div>

      <Button
        variant="outline"
        size="lg"
        fullWidth
        loading={submitting === 'demo'}
        disabled={busy}
        onClick={onDemo}
        leftIcon={<Sparkles className="h-4 w-4" />}
      >
        Use demo account
      </Button>
      <p className="mt-2 text-center text-xs text-muted">
        Instant access — no signup. Explore the full sandbox.
      </p>
    </AuthLayout>
  );
}
