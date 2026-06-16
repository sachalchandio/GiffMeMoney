# GiffMeMoney — Invest / Wallet Extension (Contract Addendum)

> Addendum to `docs/CONTRACT.md`. Same conventions (camelCase wire, Pydantic v2, strict TS,
> no placeholders). This adds the **brokerage / invest experience**: fund a wallet, split
> money across many investments, withdraw, and watch real-time per-position & total P&L.

## ⚠️ Money-handling stance (read first)

This is a **simulated / sandbox wallet** for a paper-trading demo. **No real money moves.**
- Deposits/withdrawals are simulated ledger entries — no card is ever charged.
- Card input is validated (Luhn + brand detection) and, if "remember" is chosen, stored
  **masked only** (brand + last4 + a fake token). Raw PAN/CVC is never persisted or logged.
- A `PaymentProvider` interface isolates this so a **real PSP (Stripe/Plaid) drops in later**
  behind the same contract (same pattern as the market-data provider; keys added later).
- The UI must label funding flows as **"Demo / sandbox — no real charge."**

Account model: single demo account, id from optional `X-Account-Id` header, default `"demo"`.
No auth (paper sandbox). State is in-memory + deterministic; resets on server restart.

---

## Backend additions — `backend/app/invest/`

```
app/invest/
├─ __init__.py
├─ store.py             # thread-safe in-memory AccountStore (wallets, positions, txns, cards)
├─ payments.py          # PaymentProvider ABC + SimulatedPaymentProvider (Luhn, brand, mask, token)
├─ wallet.py            # WalletService: deposit / withdraw / cards / transactions
├─ portfolio_service.py # PortfolioService: invest(split), sell, positions, mark-to-market P&L
├─ history.py           # value/P&L time series (reconstructed from entry time + price history)
└─ advisor.py           # AllocationAdvisor: "where to invest" using AnalysisEngine + Markowitz
```
New router: `backend/app/api/invest.py` (exposes `router`), mounted under `/api` in `main.py`.
> NOTE: `main.py` is owned by the core-backend build. The invest build adds a SEPARATE include
> by editing only the router-include block, or `main.py` is (re)written to include all 6 routers.
> Coordinate so the invest router is mounted. Tests must confirm it.

### Internal service API
```
payments.py:
  brand_for(number)->str ('visa'|'mastercard'|'amex'|'discover'|'unknown')
  luhn_valid(number)->bool ; mask(number)->str (•••• last4)
  class PaymentProvider(ABC): charge(card:CardIn, amount:float, account:str)->Transaction;
     payout(amount:float, account:str, destination:str|None)->Transaction
  class SimulatedPaymentProvider(PaymentProvider): always succeeds for valid card/amount>0,
     rejects invalid Luhn / amount<=0 / amount>limit(10_000) with a clear error.
  get_payment_provider()->PaymentProvider (singleton; env-selectable later).

wallet.py — WalletService(store, payments):
  get_wallet(account)->Wallet (cashBalance + investedValue(from positions) + totalValue)
  deposit(account, amount, card, save_card)->(Wallet, Transaction)  # validates, credits cash,
     optionally tokenizes+stores SavedCard
  withdraw(account, amount, destination)->(Wallet, Transaction)  # rejects amount>cash
  list_cards(account)->list[SavedCard] ; delete_card(account, card_id)->None
  list_transactions(account)->list[Transaction] (newest first)

portfolio_service.py — PortfolioService(store, provider):
  get_state(account)->PortfolioState (positions marked to live/last price; totals; pnl)
  invest(account, allocations:list[AllocationItem])->PortfolioState  # spends cash across symbols;
     rejects if sum(amount) > cash; updates/creates positions w/ avg cost basis; records buy txns
  sell(account, symbol, amount|all)->PortfolioState  # realizes P&L, credits cash, records sell txn
  Each Position: units, costBasis, avgPrice, currentPrice, marketValue, unrealizedPnl/%,
     allocationPct, realizedPnl, openedAt.

history.py — PortfolioHistory(store, provider):
  portfolio_history(account, points=120)->dict{ total: PortfolioHistoryPoint[],
     positions: [{symbol, points:[{t,value,pnl,pnlPct}]}] }  # reconstruct each position's value
     using that asset's historical closes from openedAt to now; total = cash + Σ position values.

advisor.py — AllocationAdvisor(engine, provider):
  advise(amount, riskTolerance, asset_classes)->AllocationAdvice
  Pipeline: rank universe by composite score (filter by asset_classes) -> take top N (N by risk:
     conservative 4 / balanced 6 / aggressive 8) -> Markowitz optimize (objective by risk:
     conservative=min_volatility, balanced=max_sharpe, aggressive=max_sharpe w/ higher cap) ->
     weights -> per-item amount = weight*amount, compositeScore, expectedReturn1YPct, rationale ->
     blended expected return/vol/sharpe + 5-horizon ExpectedReturn[] for the whole basket.
```

### New DTOs (add to `app/schemas.py` via the invest build; camelCase)
```ts
export interface CardIn { number: string; expMonth: number; expYear: number; cvc: string; holder: string; }
export interface SavedCard { id: string; brand: string; last4: string; expMonth: number; expYear: number; holder: string; }
export type TxnType = 'deposit' | 'withdrawal' | 'buy' | 'sell';
export interface Transaction { id: string; type: TxnType; amount: number; symbol: string | null;
  status: 'completed' | 'failed'; createdAt: number; ref: string; note: string; }
export interface Wallet { accountId: string; cashBalance: number; investedValue: number;
  totalValue: number; currency: string; savedCards: SavedCard[]; }
export interface DepositRequest { amount: number; card: CardIn; saveCard: boolean; savedCardId?: string | null; }
export interface WithdrawRequest { amount: number; destination?: string | null; }
export interface AllocationItem { symbol: string; amount: number; }   // dollars
export interface InvestRequest { allocations: AllocationItem[]; }
export interface SellRequest { symbol: string; amount: number | null; all: boolean; }
export interface Position { symbol: string; asset: Asset; units: number; costBasis: number;
  avgPrice: number; currentPrice: number; marketValue: number; unrealizedPnl: number;
  unrealizedPnlPct: number; allocationPct: number; realizedPnl: number; openedAt: number; }
export interface PortfolioState { wallet: Wallet; positions: Position[]; totalCost: number;
  totalValue: number; totalPnl: number; totalPnlPct: number; }
export interface PortfolioHistoryPoint { t: number; totalValue: number; invested: number; cash: number; }
export interface PositionHistory { symbol: string; points: { t: number; value: number; pnl: number; pnlPct: number }[]; }
export interface PortfolioHistory { total: PortfolioHistoryPoint[]; positions: PositionHistory[]; }
export type RiskTolerance = 'conservative' | 'balanced' | 'aggressive';
export interface AdviceRequest { amount: number; riskTolerance: RiskTolerance; assetClasses?: AssetClass[] | null; }
export interface AdviceItem { asset: Asset; weight: number; amount: number; compositeScore: number;
  expectedReturn1YPct: number; rationale: string; }
export interface AllocationAdvice { items: AdviceItem[]; expectedReturn: number; expectedVol: number;
  sharpe: number; horizons: ExpectedReturn[]; riskTolerance: RiskTolerance; amount: number; }
```

### New REST endpoints (prefix `/api`)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/wallet` | — | `Wallet` |
| POST | `/api/wallet/deposit` | `DepositRequest` | `{ wallet: Wallet, transaction: Transaction }` |
| POST | `/api/wallet/withdraw` | `WithdrawRequest` | `{ wallet: Wallet, transaction: Transaction }` |
| GET | `/api/wallet/cards` | — | `SavedCard[]` |
| DELETE | `/api/wallet/cards/{id}` | — | `{ ok: true }` |
| GET | `/api/wallet/transactions` | — | `Transaction[]` |
| GET | `/api/portfolio` | — | `PortfolioState` |
| POST | `/api/portfolio/invest` | `InvestRequest` | `PortfolioState` |
| POST | `/api/portfolio/sell` | `SellRequest` | `PortfolioState` |
| GET | `/api/portfolio/history` | `points?` | `PortfolioHistory` |
| POST | `/api/advisor/allocate` | `AdviceRequest` | `AllocationAdvice` |

Errors: 400 for insufficient funds / invalid card / amount<=0; 404 unknown symbol.
The existing `POST /api/portfolio/optimize` (efficient frontier) stays — it is a different,
analytical endpoint; do not remove it.

### Backend tests (`tests/test_invest.py`)
- Luhn: a valid test card passes, a bad one is rejected; brand detection.
- Deposit credits cash; save-card stores a masked SavedCard (no raw PAN anywhere); withdraw
  rejects > balance; balances reconcile (cash + invested == total).
- Invest splits cash across multiple symbols; rejects over-spend; avg cost basis correct;
  sell realizes P&L and returns cash.
- History endpoint returns total + per-position series of correct length & shape.
- Advisor returns weights summing ~1, amounts summing ~= request amount, 5 horizons.
- API smoke for every new route; invest router is mounted under /api.

---

## Frontend additions

### New page `src/pages/InvestPage.tsx` (route `/invest`) — the flagship experience
Layout = dense, responsive grid, light/dark. Sections:
1. **Wallet header** — cash, invested, total value, total P&L (live, color-coded), with
   **Add Funds** + **Withdraw** buttons. A small "Demo / sandbox — no real charge" tag.
2. **Add Funds modal** (`components/domain/AddFundsModal.tsx`) — amount with quick chips
   (incl. **$20**), debit-card form (number w/ live brand icon + Luhn check, exp, cvc, holder),
   **"Remember this card"** toggle, saved-card picker. Submits to `/api/wallet/deposit`.
3. **Withdraw modal** — amount (max = cash), confirm → `/api/wallet/withdraw`.
4. **Allocation builder** (`components/domain/AllocationBuilder.tsx`) — pick assets by class
   tabs (Stocks / Crypto / ETFs / All), add rows, set $ per asset (with % readout), live
   "allocated / remaining of cash" bar. **"Suggest for me"** with a risk segmented control
   (conservative/balanced/aggressive) calls `/api/advisor/allocate` and fills the rows.
   **Invest** posts `/api/portfolio/invest`.
5. **Positions** (`components/domain/PositionCard.tsx` / table) — per asset: units, cost,
   value, P&L $/% (live), allocation %, sparkline, **Sell** (amount or all).
6. **Real-time P&L charts** (`components/charts/PnlChart.tsx`, `AllocationDonut.tsx`) —
   total portfolio value area chart over time + per-position P&L multi-line, both live via
   the market socket; allocation donut; a 1Y expected-outcome band from the advisor.
7. **"Where to invest now"** panel — advisor top picks with per-horizon expected returns.
8. **Transactions** list.

### Hooks / store
- `hooks/useWallet.ts`, `hooks/usePortfolio.ts` (state + invest/sell mutations),
  `hooks/usePortfolioHistory.ts`, `hooks/useAdvisor.ts` (react-query + mutations,
  invalidate wallet/portfolio on success).
- Live P&L: derive from `marketStore` live prices × position units on the client (no extra
  backend calls) so P&L updates every tick; fall back to `currentPrice` from the server.
- `lib/api.ts` gains the wallet/portfolio/advisor calls; `lib/types.ts` gains the DTOs above;
  `lib/payment.ts` (Luhn, brand detect, format card number) for the form.

### Nav / polish
- Sidebar: add **Invest** prominently (directly under Dashboard). Update PortfolioPage to
  focus on the analytical optimizer (efficient frontier) and link to Invest for real money.
- Reinforce dense layout everywhere: gutters `px-3 sm:px-4 lg:px-6`, card padding `p-4`,
  grid gaps `gap-3 lg:gap-4`, content max-width ~1440px. No large empty margins.

### Frontend tests
- `InvestPage` renders wallet + sections (with mocked api).
- `AllocationBuilder` math: amounts sum, remaining clamps, % readouts.
- `lib/payment.ts`: Luhn + brand detection unit tests.
- `useWallet`/`usePortfolio` hook smoke with a mocked client.
