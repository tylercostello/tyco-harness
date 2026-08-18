// Dev utility: validate the data file and dump game.js line by line.
// Not part of the deployed site.
const fs = require('fs');
const mode = process.argv[2] || 'dump';
const raw = fs.readFileSync('state-size/data/areas.json', 'utf8');
const non = raw.match(/[\u0080-\uffff]/g);
console.log('areas.json non-ascii chars:', non ? non.length : 0);
const d = JSON.parse(raw);
console.log('top-level keys:', Object.keys(d).join(', '));
console.log('states:', Object.keys(d.states).length, 'countries:', Object.keys(d.countries).length, 'answers:', Object.keys(d.answers).length);
// Re-verify answer keys against closest-state logic (strict; report ties)
let bad = 0, ties = 0;
for (const [c, a] of Object.entries(d.answers)) {
  const area = d.countries[c];
  let best = null, bestDiff = null;
  const entries = Object.entries(d.states).map(([n, v]) => [n, Math.abs(v - area)]).sort((x, y) => x[1] - y[1]);
  if (entries[0][1] === entries[1][1]) ties++;
  if (best === null) { best = entries[0][0]; bestDiff = entries[0][1]; }
  if (best !== a) { bad++; if (bad < 5) console.log('mismatch:', c, 'expected', a, 'got', best); }
}
console.log('answer-key mismatches:', bad, 'ties at best:', ties);
// State area duplicates (would make direction feedback ambiguous)
const seen = {};
for (const [n, v] of Object.entries(d.states)) {
  if (seen[v]) console.log('duplicate state area:', seen[v], 'and', n, '=', v);
  seen[v] = n;
}
console.log('min country:', Math.min(...Object.values(d.countries)), 'max country:', Math.max(...Object.values(d.countries)));
console.log('min state:', Math.min(...Object.values(d.states)), 'max state:', Math.max(...Object.values(d.states)));
if (mode === 'dumpjs') {
  const l = fs.readFileSync('state-size/game.js', 'utf8').split('\n');
  console.log('game.js has ' + l.length + ' lines');
  const from = parseInt(process.argv[3] || '1', 10);
  const to = parseInt(process.argv[4] || String(l.length), 10);
  for (let i = from; i <= to && i <= l.length; i++) console.log(i + ': ' + l[i - 1]);
}
