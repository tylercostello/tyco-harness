// Verify the daily pick is reproducible, valid, and well-spread.
const L = require('./logic.js');
const fs = require('fs');
const path = require('path');
const d = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'areas.json'), 'utf8'));
const N = Object.keys(d.countries).length;

function pickFor(dayStr) {
  return L.pickDailyCountry(d.countries, d.answers, 'state-size:' + dayStr);
}

// 1. Reproducibility: same day twice -> identical pick.
const a = pickFor('2026-08-16');
const b = pickFor('2026-08-16');
console.log('reproducibility (same day twice):', a.country === b.country && a.state === b.state ? 'PASS' : 'FAIL');

// 2. Validity: pick is a real country with a valid answer state.
let valid = true;
for (let i = 0; i < 365; i++) {
  const dt = new Date(Date.UTC(2026, 0, 1 + i));
  const ds = L.utcDay(dt);
  const p = pickFor(ds);
  if (!d.countries[p.country] || !d.states[p.state] || d.answers[p.country] !== p.state) { valid = false; break; }
}
console.log('validity (365 days, all picks valid):', valid ? 'PASS' : 'FAIL');

// 3. Spread: how many distinct countries over 365 days (expect a large number).
const seen = new Set();
for (let i = 0; i < 365; i++) {
  const dt = new Date(Date.UTC(2026, 0, 1 + i));
  seen.add(pickFor(L.utcDay(dt)).country);
}
console.log('spread (distinct countries in a 365-day year):', seen.size, 'of', N, '->', seen.size > 50 ? 'PASS (well-spread)' : 'FAIL (too few)');

// 4. Range safety: index can never be out of [0, N-1].
let safe = true;
for (let day = 0; day < 400; day++) {
  const rand = L.mulberry32(L.fnv1a('state-size:day' + day));
  const idx = Math.floor(rand() * N);
  if (idx < 0 || idx >= N) { safe = false; break; }
}
console.log('index range safety (0 <= idx < N):', safe ? 'PASS' : 'FAIL');

// 5. Show a sample of the next 7 days from today.
console.log('\nSample (next 7 days from today):');
const today = new Date();
for (let i = 0; i < 7; i++) {
  const dt = new Date(today.getTime() + i * 86400000);
  const p = pickFor(L.utcDay(dt));
  console.log('  ' + L.utcDay(dt) + '  ' + p.country + '  ->  ' + p.state + '  (' + d.states[p.state] + ' sq km)');
}
