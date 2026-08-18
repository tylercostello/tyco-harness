// State Size
// A Wordle-style daily game: guess the US state closest in land area
// to the day's country. Six guesses. Plain vanilla JS, no dependencies.
//
// Layout of this file:
//   1. Constants and module state
//   2. Small utilities (hashing, RNG, storage)
//   3. Daily pick
//   4. Guessing and feedback
//   5. Rendering
//   6. Stats
//   7. Result and share
//   8. Modals
//   9. Event wiring and init

(function () {
  'use strict';

  // ------------------------------------------------------------------
  // 1. Constants and module state
  // ------------------------------------------------------------------

  var MAX_GUESSES = 6;
  var NS = 'state-size';
  var L = window.StateSize; // pure game logic (see logic.js)

  var states = {};      // state name -> land area in sq km
  var countries = {};   // country name -> land area in sq km
  var answers = {};     // country name -> closest state name
  var stateNames = [];  // sorted list of state names
  var day = '';         // UTC date string, e.g. "2026-08-16"
  var answerCountry = '';
  var answerState = '';
  var session = null;   // { guesses: [state name...], result: 'win'|'lose'|null }
  var lastFocus = null; // element to refocus when a modal closes

  // ------------------------------------------------------------------
  // 2. Small utilities
  // ------------------------------------------------------------------

  function $(id) { return document.getElementById(id); }


  function safeGet(key) {
    try { return JSON.parse(localStorage.getItem(key)); }
    catch (e) { return null; }
  }

  function safeSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); }
    catch (e) { /* storage unavailable; play on without persistence */ }
  }

  function safeRemove(key) {
    try { localStorage.removeItem(key); }
    catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------
  // 3. Daily pick and session
  // ------------------------------------------------------------------

  function pickDailyCountry() {
    // Everyone in the world sees the same country on the same UTC day.
    var pick = L.pickDailyCountry(countries, answers, NS + ':' + day);
    answerCountry = pick.country;
    answerState = pick.state;
  }

  function sessionKey() { return NS + ':session:' + day; }

  function loadSession() {
    var s = safeGet(sessionKey());
    if (s && Array.isArray(s.guesses)) {
      s.result = (s.result === 'win' || s.result === 'lose') ? s.result : null;
      return s;
    }
    return { guesses: [], result: null };
  }

  function saveSession() {
    safeSet(sessionKey(), session);
  }

  // ------------------------------------------------------------------
  // 4. Guessing and feedback
  // ------------------------------------------------------------------

  // Delegate to the pure logic module (logic.js).
  function evaluateGuess(state) {
    return L.evaluateGuess(states, answerState, state);
  }

  function submitGuess(rawState) {
    var state = rawState.trim();
    if (!state) return showError('Pick a state first.');
    if (!Object.prototype.hasOwnProperty.call(states, state)) {
      return showError('Not a US state. Pick one from the list.');
    }
    if (session.guesses.indexOf(state) !== -1) {
      return showError('You already guessed ' + state + '.');
    }
    if (session.result) return;

    var evalr = evaluateGuess(state);
    session.guesses.push(state);
    if (evalr.win) session.result = 'win';
    else if (session.guesses.length >= MAX_GUESSES) session.result = 'lose';
    saveSession();

    $('input-error').hidden = true;
    $('state-input').value = '';
    $('state-input').focus();

    if (session.result) {
      finishGame(session.result);
    } else {
      var left = MAX_GUESSES - session.guesses.length;
      setStatus(
        state + ' is ' + (evalr.direction === 'L' ? 'smaller' : 'larger') +
        ' than the answer state. ' + evalr.band.label +
        ' (' + evalr.pct + '% off). ' + left + ' guess' +
        (left === 1 ? '' : 'es') + ' left.'
      );
    }
    renderGuesses();
  }

  function recordedKey() { return NS + ':recorded:' + day; }
  function isRecorded() { return safeGet(recordedKey()) === true; }
  function markRecorded() { safeSet(recordedKey(), true); }

  function finishGame(result) {
    if (!isRecorded()) {
      recordStats(result);
      markRecorded();
    }
    renderGuesses();
    renderResult(result);
    setStatus(result === 'win'
      ? 'Solved in ' + session.guesses.length + '. ' + answerCountry +
        ' is closest to ' + answerState + '.'
      : 'Not solved. ' + answerCountry + ' is closest to ' + answerState + '.');
    $('state-input').disabled = true;
    $('submit-btn').disabled = true;
  }

  // Let the player try the same day again (e.g. after losing). The day's
  // stats are only ever recorded once, so replaying can't be farmed.
  function replayToday() {
    session = { guesses: [], result: null };
    saveSession();
    $('result').hidden = true;
    $('result').className = 'card result';
    $('state-input').disabled = false;
    $('submit-btn').disabled = false;
    $('input-error').hidden = true;
    renderGuesses();
    setStatus('Which US state is closest in area to ' + answerCountry + '?');
    $('state-input').focus();
  }

  function showError(msg) {
    var el = $('input-error');
    el.textContent = msg;
    el.hidden = false;
  }

  function setStatus(msg) {
    $('status').textContent = msg;
  }

  // ------------------------------------------------------------------
  // 5. Rendering
  // ------------------------------------------------------------------

  function buildDatalist() {
    var list = $('state-list');
    list.innerHTML = '';
    stateNames.forEach(function (name) {
      var opt = document.createElement('option');
      opt.value = name;
      list.appendChild(opt);
    });
  }

  function tempMeter(band) {
    var filled = { hot: 5, warm: 4, cool: 3, cold: 2 }[band.cls];
    var out = '<span class="temp ' + band.cls + '" aria-hidden="true">';
    for (var i = 0; i < 5; i++) out += '<i></i>';
    out += '</span><span class="sr-temp">' + band.label +
         ', ' + filled + ' of 5</span>';
    return out;
  }

  function renderGuesses() {
    var list = $('guess-list');
    list.innerHTML = '';
    session.guesses.forEach(function (state, i) {
      var ev = evaluateGuess(state);
      var li = document.createElement('li');
      var cls = 'guess' +
        (session.result === 'win' && ev.win ? ' win' : '') +
        (session.result === 'lose' ? ' lose' : '');
      li.className = cls;
      var dirWord = ev.direction === 'L' ? 'larger' : 'smaller';
      li.innerHTML =
        '<span class="guess-num" aria-hidden="true">' + (i + 1) + '</span>' +
        '<div class="guess-main">' +
          '<div class="guess-line">' +
            '<span class="guess-name">' + state + '</span>' +
            '<span class="badge ' + (ev.direction === 'L' ? 'larger' : 'smaller') +
              '" title="The answer state is ' + dirWord + '">' +
              ev.direction + '</span>' +
            '<span aria-label="Temperature ' + ev.band.label.toLowerCase() +
              ', ' + ev.pct + ' percent difference">' +
              tempMeter(ev.band) + '</span>' +
            '<span class="diff-pct">' + ev.pct + '% off</span>' +
          '</div>' +
        '</div>';
      list.appendChild(li);
    });
  }

  function renderAll() {
    $('day-tag').textContent = 'Day ' + day + ' (UTC)';
    $('country-name').textContent = answerCountry;
    buildDatalist();
    renderGuesses();
    renderStatsModal();
    if (session.result) {
      renderResult(session.result, true);
      $('state-input').disabled = true;
      $('submit-btn').disabled = true;
    } else {
      $('result').hidden = true;
      $('result').className = 'card result';
      setStatus('Which US state is closest in area to today\'s country?');
    }
  }

  // ------------------------------------------------------------------
  // 6. Stats
  // ------------------------------------------------------------------

  var STATS_KEY = NS + ':stats';

  function defaultStats() {
    return { played: 0, wins: 0, streak: 0, best: 0, dist: [0, 0, 0, 0, 0, 0] };
  }

  function loadStats() {
    var s = safeGet(STATS_KEY);
    if (!s || !Array.isArray(s.dist)) return defaultStats();
    return s;
  }

  function saveStats(s) { safeSet(STATS_KEY, s); }

  function recordStats(result) {
    var s = loadStats();
    s.played += 1;
    if (result === 'win') {
      s.wins += 1;
      s.streak += 1;
      if (s.streak > s.best) s.best = s.streak;
      s.dist[session.guesses.length - 1] += 1;
    } else {
      s.streak = 0;
    }
    saveStats(s);
    renderStatsModal();
  }

  function resetStats() {
    safeRemove(STATS_KEY);
    safeRemove(sessionKey());
    safeRemove(recordedKey());
    renderStatsModal();
  }

  function renderStatsModal() {
    var s = loadStats();
    $('stat-played').textContent = s.played;
    $('stat-wins').textContent = s.wins;
    $('stat-pct').textContent = s.played
      ? Math.round((s.wins / s.played) * 100) + '%' : '0%';
    $('stat-streak').textContent = s.streak;
    $('stat-best').textContent = s.best;
    var total = 0;
    for (var i = 0; i < 6; i++) {
      $('dist-' + (i + 1)).textContent = s.dist[i];
      total += s.dist[i] * (i + 1);
    }
    $('stat-avg').textContent = s.wins ? (total / s.wins).toFixed(1) : '-';
  }

  // ------------------------------------------------------------------
  // 7. Result and share
  // ------------------------------------------------------------------

  function fmtArea(n) {
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function buildShareText() {
    var lines = [];
    lines.push('State Size ' + day + ' - ' +
      session.guesses.length + '/' + MAX_GUESSES);
    session.guesses.forEach(function (state) {
      var ev = evaluateGuess(state);
      lines.push(ev.direction + '  ' + ev.band.code + '  ' +
        (ev.win ? 'X' : 'O'));
    });
    lines.push(answerCountry + ' -> ' + answerState);
    lines.push('L larger, S smaller | H hot W warm C cool D cold | X win');
    return lines.join('\n');
  }

  function renderResult(result, restored) {
    var panel = $('result');
    panel.hidden = false;
    panel.className = 'card result ' + (result === 'win' ? 'win' : 'lose');
    $('result-title').textContent = result === 'win'
      ? 'Solved in ' + session.guesses.length
      : 'Not solved';
    $('result-detail').textContent =
      answerCountry + ' (' + fmtArea(countries[answerCountry]) +
      ' sq km) is closest to ' + answerState + ' (' +
      fmtArea(states[answerState]) + ' sq km).';
    $('share-text').textContent = buildShareText();
    $('close-result').focus();
    if (restored) {
      // Restored from storage: keep focus where the user had it.
      $('close-result').blur();
    }
  }

  function copyShare() {
    var text = $('share-text').textContent;
    var btn = $('copy-btn');

    function done() {
      var old = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(function () { btn.textContent = old; }, 1500);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }

    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', 'readonly');
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (e) { /* ignore */ }
      ta.remove();
      done();
    }
  }

  // ------------------------------------------------------------------
  // 8. Modals
  // ------------------------------------------------------------------

  function openModal(id) {
    lastFocus = document.activeElement;
    var modal = $(id);
    modal.hidden = false;
    var focusTarget = modal.querySelector('button');
    if (focusTarget) focusTarget.focus();
  }

  function closeModal(id) {
    $(id).hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function trapTab(id, event) {
    if (event.key !== 'Tab') return;
    var modal = $(id);
    var focusables = modal.querySelectorAll('button, [href], [tabindex]');
    if (!focusables.length) return;
    var first = focusables[0];
    var last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  // ------------------------------------------------------------------
  // 9. Event wiring and init
  // ------------------------------------------------------------------

  function wireEvents() {
    $('guess-form').addEventListener('submit', function (e) {
      e.preventDefault();
      submitGuess($('state-input').value);
    });

    $('help-btn').addEventListener('click', function () {
      openModal('help-modal');
    });
    $('help-close').addEventListener('click', function () {
      closeModal('help-modal');
      $('help-btn').focus();
    });

    $('stats-btn').addEventListener('click', function () {
      openModal('stats-modal');
    });
    $('stats-close').addEventListener('click', function () {
      closeModal('stats-modal');
      $('stats-btn').focus();
    });
    $('reset-stats').addEventListener('click', function () {
      resetStats();
    });

    $('copy-btn').addEventListener('click', copyShare);
    $('replay-btn').addEventListener('click', replayToday);
    $('close-result').addEventListener('click', function () {
      $('result').hidden = true;
      setStatus('Which US state is closest in area to ' + answerCountry + '?');
    });

    // Escape closes any open modal; Tab is trapped inside it.
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      if (!$('help-modal').hidden) closeModal('help-modal');
      else if (!$('stats-modal').hidden) closeModal('stats-modal');
    });
    $('help-modal').addEventListener('keydown', function (e) { trapTab('help-modal', e); });
    $('stats-modal').addEventListener('keydown', function (e) { trapTab('stats-modal', e); });
  }

  function validateData(data) {
    if (!data || typeof data !== 'object') return 'Game data is missing.';
    if (!data.states || !data.countries || !data.answers) return 'Game data is incomplete.';
    if (!Object.keys(data.states).length) return 'No US state data.';
    if (!Object.keys(data.countries).length) return 'No country data.';
    for (var c in data.countries) {
      if (!Object.prototype.hasOwnProperty.call(data.answers, c)) return 'No answer for ' + c + '.';
      if (!Object.prototype.hasOwnProperty.call(data.states, data.answers[c])) return 'Bad answer for ' + c + '.';
    }
    return null;
  }

  function init(data) {
    var problem = validateData(data);
    if (problem) {
      document.querySelector('main').innerHTML =
        '<p class="error" style="padding:1rem">Game data problem: ' + problem + '</p>';
      return;
    }
    states = data.states;
    countries = data.countries;
    answers = data.answers;
    stateNames = Object.keys(states).sort();

    day = L.utcDay(new Date());
    pickDailyCountry();
    session = loadSession();

    buildDatalist();
    wireEvents();

    if (session.result) {
      finishGame(session.result, true);
    } else {
      setStatus('Which US state is closest in area to today\'s country?');
    }
    renderAll();
  }

  // Fetch the data file; fall back to same-origin if the fetch API is
  // unavailable (very old browsers) so the site still works offline.
  function loadData(cb) {
    var url = 'data/areas.json';
    if (window.fetch) {
      fetch(url)
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(cb, function (err) {
          document.body.innerHTML =
            '<p class="error" style="padding:2rem">Could not load game data. ' +
            'Run the site over HTTP (for example: npx serve .).</p>';
        });
    } else {
      // Very old browsers: fall back to XHR.
      var xhr = new XMLHttpRequest();
      xhr.open('GET', url);
      xhr.onload = function () {
        if (xhr.status >= 200 && xhr.status < 300) {
          cb(JSON.parse(xhr.responseText));
        }
      };
      xhr.send();
    }
  }

  loadData(init);
})();
