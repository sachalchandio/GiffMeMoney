/**
 * SignupPage — branded (light/dark) account-creation screen (AUTH.md).
 *
 * Mirrors the LoginPage chrome (reusing its AuthLayout / Field). Validates name,
 * email, and password (≥ 6 chars per the backend contract) with a live strength
 * hint, confirms the password, surfaces API errors inline, and offers the same
 * "Use demo account" shortcut. Redirects already-authenticated users away.
 */

import { useState, type FormEvent } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { AlertCircle, ArrowRight, Eye, EyeOff, Lock, Mail, Sparkles, User } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { DEMO_CREDENTIALS, useAuth } from '@/lib/auth';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';
import { AuthLayout, Field } from './LoginPage';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD = 6;

interface FieldErrors {
  name?: string;
  email?: string;
  password?: string;
  confirm?: string;
}

/** Coarse 0–3 password strength + label for the inline meter. */
function passwordStrength(pw: string): { level: 0 | 1 | 2 | 3; label: string } {
  if (pw.length === 0) return { level: 0, label: '' };
  let score = 0;
  if (pw.length >= MIN_PASSWORD) score += 1;
  if (pw.length >= 10) score += 1;
  if (/[^A-Za-z0-9]/.test(pw) || (/[A-Za-z]/.test(pw) && /[0-9]/.test(pw))) score += 1;
  const level = Math.min(3, score) as 0 | 1 | 2 | 3;
  const label = level <= 1 ? 'Weak' : level === 2 ? 'Okay' : 'Strong';
  return { level, label };
}

export default function SignupPage(): JSX.Element {
  const navigate = useNavigate();
  const { signup, loginDemo, isAuthenticated, isLoading } = useAuth();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<'form' | 'demo' | null>(null);

  if (!isLoading && isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  const strength = passwordStrength(password);

  function validate(): boolean {
    const next: FieldErrors = {};
    if (name.trim().length < 2) next.name = 'Please enter your name.';
    if (!EMAIL_RE.test(email.trim())) next.email = 'Enter a valid email address.';
    if (password.length < MIN_PASSWORD) next.password = `Use at least ${MIN_PASSWORD} characters.`;
    if (confirm !== password) next.confirm = 'Passwords do not match.';
    setFieldErrors(next);
    return Object.keys(next).length === 0;
  }

  function describeError(err: unknown): string {
    if (err instanceof ApiError) {
      if (err.status === 0) return 'Cannot reach the server. Is the backend running?';
      return err.detail || 'Could not create your account. Please try again.';
    }
    return 'Could not create your account. Please try again.';
  }

  async function onSubmit(e: FormEvent): Promise<void> {
    e.preventDefault();
    setFormError(null);
    if (!validate()) return;
    setSubmitting('form');
    try {
      await signup({ name: name.trim(), email: email.trim(), password });
      navigate('/', { replace: true });
    } catch (err) {
      setFormError(describeError(err));
    } finally {
      setSubmitting(null);
    }
  }

  async function onDemo(): Promise<void> {
    setFormError(null);
    setFieldErrors({});
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
  const strengthColor =
    strength.level <= 1 ? 'bg-danger' : strength.level === 2 ? 'bg-warning' : 'bg-success';

  return (
    <AuthLayout
      heading="Create your account"
      subheading="Start exploring where to invest — free, in a safe sandbox."
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-primary underline-offset-4 hover:underline">
            Sign in
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
          id="signup-name"
          label="Name"
          type="text"
          autoComplete="name"
          placeholder="Ada Lovelace"
          icon={<User className="h-4 w-4" />}
          value={name}
          error={fieldErrors.name}
          disabled={busy}
          onChange={(e) => setName(e.target.value)}
        />

        <Field
          id="signup-email"
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

        <div>
          <Field
            id="signup-password"
            label="Password"
            type={showPassword ? 'text' : 'password'}
            autoComplete="new-password"
            placeholder="At least 6 characters"
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
          {password.length > 0 && !fieldErrors.password && (
            <div className="mt-2 flex items-center gap-2">
              <span className="flex flex-1 gap-1" aria-hidden>
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className={cn(
                      'h-1 flex-1 rounded-full transition-colors',
                      i < strength.level ? strengthColor : 'bg-border',
                    )}
                  />
                ))}
              </span>
              <span className="w-12 shrink-0 text-right text-xs text-muted">{strength.label}</span>
            </div>
          )}
        </div>

        <Field
          id="signup-confirm"
          label="Confirm password"
          type={showPassword ? 'text' : 'password'}
          autoComplete="new-password"
          placeholder="Re-enter your password"
          icon={<Lock className="h-4 w-4" />}
          value={confirm}
          error={fieldErrors.confirm}
          disabled={busy}
          onChange={(e) => setConfirm(e.target.value)}
        />

        <Button
          type="submit"
          size="lg"
          fullWidth
          loading={submitting === 'form'}
          disabled={busy}
          rightIcon={<ArrowRight className="h-4 w-4" />}
        >
          Create account
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
        Prefer to look around first? Jump in as{' '}
        <span className="font-medium text-text">{DEMO_CREDENTIALS.email}</span>.
      </p>
    </AuthLayout>
  );
}
