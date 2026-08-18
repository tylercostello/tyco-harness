// State Size -- pure game logic (no DOM). Isomorphic: usable in the
// browser (attaches to window.StateSize) and in Node (module.exports),
// so the core rules can be unit-tested without a browser.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.StateSize = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var NS = 'state-size';

  // Temperature bands: a guess is at least this "hot" when its land
  // area is within `max` percent of the answer state's land area.
  var TEMP_BANDS = [
    { max: 10, cls: 'hot',  label: 'HOT',  code: 'H' },
    { max: 30, cls: 'warm', label: 'WARM', code: 'W' },
    { max: 75, cls: 'cool', label: 'COOL', code: 'C' }
  ];
  var COLD_BAND = { cls: 'cold', label: 'COLD', code: 'D' };

  // FNV-1a 32-bit string hash; deterministic across environments.
  function fnv1a(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  // mulberry32: tiny deterministic PRNG seeded from the UTC date.
  function mulberry32(seed) {
    var t = seed >>> 0;
    return function () {
      t = (t + 0x6D2B79F5) >>> 0;
      var r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // The US state whose land area is closest to `area`.
  // Ties are broken deterministically by name (ascending) so the result
  // is stable regardless of object key order.
  function closestState(states, area) {
    var best = null;
    var bestDiff = null;
    var names = Object.keys(states).sort();
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      var diff = Math.abs(states[name] - area);
      if (best === null || diff < bestDiff) {
        best = name;
        bestDiff = diff;
      }
    }
    return best;
  }

  function bandFor(pct) {
    for (var i = 0; i < TEMP_BANDS.length; i++) {
      if (pct <= TEMP_BANDS[i].max) return TEMP_BANDS[i];
    }
    return COLD_BAND;
  }

  // Direction of the answer relative to the guess:
  // 'L' means the answer state is LARGER, 'S' means SMALLER.
  function directionFor(states, answerState, guessState) {
    return states[guessState] < states[answerState] ? 'L' : 'S';
  }

  function evaluateGuess(states, answerState, guessState) {
    var answerArea = states[answerState];
    var diff = Math.abs(states[guessState] - answerArea);
    var pct = Math.round((diff / answerArea) * 100);
    return {
      pct: pct,
      win: guessState === answerState,
      direction: directionFor(states, answerState, guessState),
      band: bandFor(pct)
    };
  }

  // Deterministic daily pick. `seedString` is typically NS + ':' + day.
  // Returns { country, state }.
  function pickDailyCountry(countries, answers, seedString) {
    var rand = mulberry32(fnv1a(seedString));
    var names = Object.keys(countries).sort();
    var idx = Math.floor(rand() * names.length);
    var country = names[idx];
    return { country: country, state: answers[country] };
  }

  // Format a UTC Date as YYYY-MM-DD.
  function utcDay(d) {
    return d.getUTCFullYear() + '-' +
      String(d.getUTCMonth() + 1).padStart(2, '0') + '-' +
      String(d.getUTCDate()).padStart(2, '0');
  }

  return {
    NS: NS,
    TEMP_BANDS: TEMP_BANDS,
    COLD_BAND: COLD_BAND,
    fnv1a: fnv1a,
    mulberry32: mulberry32,
    closestState: closestState,
    bandFor: bandFor,
    directionFor: directionFor,
    evaluateGuess: evaluateGuess,
    pickDailyCountry: pickDailyCountry,
    utcDay: utcDay
  };
});
