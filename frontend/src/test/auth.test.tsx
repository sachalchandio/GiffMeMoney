/**
 * Auth-context test (FRONTEND.md test matrix).
 *
 * Proves the AuthProvider contract end-to-end with a mocked api client:
 *  - starts unauthenticated when no token is persisted,
 *  - `login` applies the session, persists the token, and mirrors it into the
 *    api client's bearer header,
 *  - `logout` clears state + storage,
 *  - on mount with a persisted token it hydrates the user via `me()`,
 *  - a 401 from `me()` drops the stale session.
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';

// Mock the api client used by the AuthProvider.
vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api');
  return {
    ...actual,
    api: {
      login: vi.fn(),
      signup: vi.fn(),
      me: vi.fn(),
    },
    setAuthToken: vi.fn(),
  };
});

import { api, ApiError, setAuthToken } from '@/lib/api';
import { AuthProvider, DEMO_CREDENTIALS, useAuth } from '@/lib/auth';
import type { AuthResponse, UserDTO } from '@/lib/types';

const DEMO_USER: UserDTO = {
  id: 'u_demo',
  email: DEMO_CREDENTIALS.email,
  name: 'Demo Investor',
  createdAt: 1_700_000_000_000,
};

const DEMO_AUTH: AuthResponse = { token: 'tok_demo', user: DEMO_USER };

const mockedApi = vi.mocked(api);
const mockedSetAuthToken = vi.mocked(setAuthToken);

function wrapper({ children }: { children: ReactNode }): JSX.Element {
  return <AuthProvider>{children}</AuthProvider>;
}

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});

describe('AuthProvider', () => {
  it('starts unauthenticated with no persisted token', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
  });

  it('login applies the session, persists the token, and sets the bearer header', async () => {
    mockedApi.login.mockResolvedValue(DEMO_AUTH);
    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login(DEMO_CREDENTIALS);
    });

    expect(mockedApi.login).toHaveBeenCalledWith(DEMO_CREDENTIALS);
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(DEMO_USER);
    expect(result.current.token).toBe('tok_demo');
    expect(localStorage.getItem('giff_token')).toBe('tok_demo');
    expect(mockedSetAuthToken).toHaveBeenCalledWith('tok_demo');
  });

  it('loginDemo authenticates with the seeded demo credentials', async () => {
    mockedApi.login.mockResolvedValue(DEMO_AUTH);
    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.loginDemo();
    });

    expect(mockedApi.login).toHaveBeenCalledWith(DEMO_CREDENTIALS);
    expect(result.current.user).toEqual(DEMO_USER);
  });

  it('logout clears state and storage', async () => {
    mockedApi.login.mockResolvedValue(DEMO_AUTH);
    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login(DEMO_CREDENTIALS);
    });
    expect(result.current.isAuthenticated).toBe(true);

    act(() => {
      result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
    expect(localStorage.getItem('giff_token')).toBeNull();
    expect(mockedSetAuthToken).toHaveBeenLastCalledWith(null);
  });

  it('hydrates the user from a persisted token on mount', async () => {
    localStorage.setItem('giff_token', 'tok_persisted');
    mockedApi.me.mockResolvedValue(DEMO_USER);

    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(mockedApi.me).toHaveBeenCalledTimes(1);
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(DEMO_USER);
    expect(result.current.token).toBe('tok_persisted');
  });

  it('drops the session when the persisted token is rejected (401)', async () => {
    localStorage.setItem('giff_token', 'tok_stale');
    mockedApi.me.mockRejectedValue(new ApiError(401, 'Invalid token'));

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
    expect(localStorage.getItem('giff_token')).toBeNull();
  });
});
