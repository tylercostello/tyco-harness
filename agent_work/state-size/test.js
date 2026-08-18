// State Size -- unit tests (run with: node test.js)
var fs = require('fs');
var path = require('path');
var L = require('./logic.js');

var passed = 0, failed = 0;
function assert(cond, msg) {
  if (cond) { passed++; }
  else { failed++; console.log('  FAIL: ' + msg); }
}

// ---- Load data ----
var data = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'areas.json'), 'utf8'));
var states = data.states, countries = data.countries, answers = data.answers;

// ---- Test 1: data integrity ----
console.log('data integrity:');
assert(typeof states === 'object' && states !== null, 'states is an object');
assert(typeof countries === 'object' && countries !== null, 'countries is an object');
assert(typeof answers === 'object' && answers !== null, 'answers is an object');
assert(Object.keys(states).length === 50, 'exactly 50 states (got ' + Object.keys(states).length + ')');
assert(Object.keys(countries).length >= 100, 'sane number of countries (got ' + Object.keys(countries).length + ')');
assert(Object.keys(answers).length === Object.keys(countries).length,
  'every country has an answer');

var dupState = {};
Object.keys(states).forEach(function (n) {
  if (states[n] <= 0) assert(false, 'state area > 0: ' + n);
  dupState[n] = true;
});
Object.keys(countries).forEach(function (n) {
  if (countries[n] <= 0) assert(false, 'country area > 0: ' + n);
});

// ---- Test 2: answer-key correctness (recompute nearest state) ----
console.log('answer-key correctness:');
var mismatches = 0, ties = 0;
Object.keys(countries).forEach(function (c) {
  var nearest = L.closestState(states, countries[c]);
  if (answers[c] !== nearest) { mismatches++; }
});
assert(mismatches === 0, 'answer key matches recomputed nearest state (' + mismatches + ' mismatches)');

// Count exact ties (two states equidistant from a country) to confirm none exist.
Object.keys(countries).forEach(function (c) {
  var a = countries[c];
  var diffs = Object.keys(states).map(function (n) { return Math.abs(states[n] - a); }).sort(function (x, y) { return x - y; });
  if (diffs[0] === diffs[1]) ties++;
});
assert(ties === 0, 'no country sits exactly between two states (' + ties + ' ties)');

// ---- Test 3: closestState tie-break is deterministic (name ascending) ----
console.log('closestState tie-break:');
var tieStates = { 'Alpha': 100, 'Beta': 200, 'Gamma': 300 };
// area 150 is exactly between Alpha(100) and Beta(200): expect 'Alpha' (earlier name).
assert(L.closestState(tieStates, 150) === 'Alpha', 'tie broken by earlier name (got ' + L.closestState(tieStates, 150) + ')');
// area 250 is exactly between Beta and Gamma: expect 'Beta'.
assert(L.closestState(tieStates, 250) === 'Beta', 'tie 250 -> Beta (got ' + L.closestState(tieStates, 250) + ')');

// ---- Test 4: evaluateGuess ----
console.log('evaluateGuess:');
var ev = L.evaluateGuess(states, 'California', 'California');
assert(ev.win === true, 'correct guess wins');
assert(ev.pct === 0, 'correct guess pct is 0');
var ev2 = L.evaluateGuess(states, 'California', 'Texas');
assert(ev2.win === false, 'wrong guess does not win');
assert(typeof ev2.pct === 'number' && ev2.pct > 0, 'wrong guess pct > 0');
assert(ev2.direction === 'L' || ev2.direction === 'S', 'direction is L or S');
assert(['hot', 'warm', 'cool', 'cold'].indexOf(ev2.band.cls) !== -1, 'band has a valid class');

// direction sanity: a smaller state than the answer -> 'L' (answer is larger)
var ev3 = L.evaluateGuess(states, 'Alaska', 'Rhode Island');
assert(ev3.direction === 'L', 'Rhode Island vs Alaska -> answer is LARGER (L)');
var ev4 = L.evaluateGuess(states, 'Rhode Island', 'Alaska');
assert(ev4.direction === 'S', 'Alaska vs Rhode Island -> answer is SMALLER (S)');

// ---- Test 5: band boundaries ----
console.log('band boundaries:');
assert(L.bandFor(5).cls === 'hot', '5% -> hot');
assert(L.bandFor(10).cls === 'hot', '10% -> hot (inclusive)');
assert(L.bandFor(11).cls === 'warm', '11% -> warm');
assert(L.bandFor(30).cls === 'warm', '30% -> warm (inclusive)');
assert(L.bandFor(31).cls === 'cool', '31% -> cool');
assert(L.bandFor(75).cls === 'cool', '75% -> cool (inclusive)');
assert(L.bandFor(76).cls === 'cold', '76% -> cold');
assert(L.bandFor(1000).cls === 'cold', '1000% -> cold');

// ---- Test 6: deterministic daily pick ----
console.log('daily pick determinism:');
var seed = 'state-size:2026-08-16';
var p1 = L.pickDailyCountry(countries, answers, seed);
var p2 = L.pickDailyCountry(countries, answers, seed);
assert(p1.country === p2.country && p1.state === p2.state, 'same seed -> same pick');
assert(answers[p1.country] === p1.state, 'picked country maps to its answer state');
assert(states[p1.state] !== undefined, 'picked state exists');
// different seed should (almost surely) give a different country over many days
var seen = {};
var allSame = true;
for (var d = 0; d < 30; d++) {
  var pick = L.pickDailyCountry(countries, answers, seed + ':' + d);
  seen[pick.country] = true;
}
var distinct = Object.keys(seen).length;
assert(distinct > 1, '30 different days produce >1 distinct country (got ' + distinct + ')');

// ---- Test 7: utcDay formatting ----
console.log('utcDay:');
assert(L.utcDay(new Date(Date.UTC(2026, 0, 5))) === '2026-01-05', 'utcDay zero-pads (got ' + L.utcDay(new Date(Date.UTC(2026, 0, 5))) + ')');
assert(L.utcDay(new Date(Date.UTC(2026, 11, 31))) === '2026-12-31', 'utcDay end of year');

// ---- Test 8: PRNG determinism + range ----
console.log('PRNG:');
var r1 = L.mulberry32(12345), r2 = L.mulberry32(12345);
var same = true;
for (var i = 0; i < 50; i++) { if (r1() !== r2()) { same = false; break; } }
assert(same, 'mulberry32 same seed -> same sequence');
var r3 = L.mulberry32(999);
var inRange = true;
for (var j = 0; j < 200; j++) { var v = r3(); if (v < 0 || v >= 1) { inRange = false; break; } }
assert(inRange, 'mulberry32 values in [0,1)');

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed === 0 ? 0 : 1);
