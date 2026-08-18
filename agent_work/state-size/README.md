# State Size

A Wordle-style **daily** game. Each UTC day the game picks one country, and you
have **6 guesses** to name the US state whose **land area** is closest to that
country's. No build step, no dependencies — open `index.html` and play.

## Run it

The game loads its data with `fetch`, so it needs to be served over HTTP
(opening the file directly can be blocked by the browser):

```bash
# from this folder
npx serve .
# or
python3 -m http.server 8000
```

Then open http://localhost:3000 (or :8000).

## How it works

- **Daily pick** — the country is chosen deterministically from the UTC date
  (FNV-1a hash + mulberry32 PRNG). Everyone in the world sees the same country
  on the same UTC day.
- **Feedback** — after each guess you get:
  - **L / S** — the answer state is *larger* or *smaller* than your pick.
  - **HOT / WARM / COOL / COLD** — how close, by % difference from the answer
    state's area (≤10% hot, ≤30% warm, ≤75% cool, else cold).
- **Stats** — played / wins / win% / streak / best / average, plus a guess
  distribution, stored in `localStorage`. A given day is only ever counted
  once, so replaying can't inflate your numbers.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Page + markup (game card, modals, result panel). |
| `style.css` | Styling — mobile-first, dark-mode aware, no external assets. |
| `logic.js` | Pure, testable game logic (hash, PRNG, nearest-state, feedback, daily pick). Works in Node and the browser. |
| `game.js` | UI: rendering, input, persistence, stats, share. |
| `data/areas.json` | 50 US states, ~148 countries, and the precomputed answer key. |
| `test.js` | Unit tests for data integrity + game logic. |
| `e2e.js` | Headless end-to-end test (runs `game.js` against a DOM stub, no browser). |
| `verify-daily.js` | Confirms the daily pick is reproducible, valid, and well-spread. |
| `check.js` | Dev data validator (run from any cwd). |

## Test

```bash
node test.js            # unit tests
node e2e.js             # headless end-to-end
node verify-daily.js    # daily-pick checks
node check.js           # data validation
```

## Data

Areas are **land area in square kilometers** (US states from the Census Bureau;
countries from standard land-area figures). The answer key in `areas.json`
is the precomputed nearest state for each country; `test.js` recomputes it to
confirm there are no mismatches or ties.
