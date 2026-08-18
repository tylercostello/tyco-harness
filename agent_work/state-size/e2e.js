// Headless end-to-end test: runs logic.js + game.js against a minimal DOM
// stub (no browser needed) and drives a full play session.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'areas.json'), 'utf8'));
const LOGIC_SRC = fs.readFileSync(path.join(__dirname, 'logic.js'), 'utf8');
const GAME_SRC = fs.readFileSync(path.join(__dirname, 'game.js'), 'utf8');
const doc = { __els: {} };
function makeEl(id) {
  const el = {
    id: id || '', textContent: '', value: '', hidden: false, disabled: false,
    className: '', _html: '', style: {}, dataset: {}, children: [], listeners: {},
    set innerHTML(v) { this._html = v; if (v === '') this.children = []; },
    get innerHTML() { return this._html; },
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    removeChild(c) { this.children = this.children.filter(x => x !== c); },
    remove() { if (this.parentNode) this.parentNode.removeChild(this); },
    focus() { this.focused = true; doc.activeElement = this; },
    blur() { this.focused = false; },
    select() {},
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k]; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    removeEventListener(t, fn) { this.listeners[t] = (this.listeners[t] || []).filter(f => f !== fn); },
    querySelector() { return makeEl('child-button'); },
    querySelectorAll() { return []; },
    dispatch(t, ev) { (this.listeners[t] || []).forEach(fn => fn(ev || { preventDefault() {} })); }
  };
  return el;
}
const IDS = ['challenge-heading','close-result','copy-btn','country-name','day-tag',
  'dist-1','dist-2','dist-3','dist-4','dist-5','dist-6','dist-table','guess-form',
  'guess-list','help-btn','help-close','help-modal','help-title','input-error',
  'replay-btn','reset-stats','result','result-detail','result-title','share-text',
  'stat-avg','stat-best','stat-pct','stat-played','stat-streak','stat-wins',
  'state-input','state-list','stats-btn','stats-close','stats-modal','stats-title',
  'status','submit-btn','main'];
function makeDoc() {
  const els = {};
  IDS.forEach(id => { els[id] = makeEl(id); });
  const d = {
    activeElement: makeEl('body-focus'), body: makeEl('body'), els,
    getElementById(id) { return els[id] || null; },
    querySelector(sel) { if (sel === 'main') return els['main']; return makeEl('q'); },
    createElement(tag) { return makeEl(tag); },
    addEventListener() {}, removeEventListener() {}, execCommand() { return true; }
  };
  return d;
}
function makeStorage() {
  const m = {};
  return {
    getItem(k) { return Object.prototype.hasOwnProperty.call(m, k) ? m[k] : null; },
    setItem(k, v) { m[k] = String(v); },
    removeItem(k) { delete m[k]; },
    clear() { for (const k in m) delete m[k]; }
  };
}
function createApp(seed) {
  const d = makeDoc();
  const storage = makeStorage();
  if (seed) { for (const k in seed) storage.setItem(k, seed[k]); }
  const ctx = {
    console, setTimeout, clearTimeout, Promise, localStorage: storage,
    navigator: {},
    document: d,
    fetch() { return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(DATA) }); }
  };
  ctx.window = ctx; ctx.self = ctx; ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(LOGIC_SRC, ctx);
  vm.runInContext(GAME_SRC, ctx);
  return new Promise(resolve => {
    setTimeout(() => resolve({ ctx, doc: d, storage, els: d.els, L: ctx.StateSize }), 20);
  });
}
let passed = 0, failed = 0;
function assert(cond, msg) { if (cond) passed++; else { failed++; console.log('  FAIL: ' + msg); } }
function fire(els, id, t, ev) { (els[id].listeners[t] || []).forEach(fn => fn(ev || { preventDefault() {} })); }
function submitState(els, name) { els['state-input'].value = name; fire(els, 'guess-form', 'submit'); }
const L = require('./logic.js');
const expectedDay = new Date().toISOString().slice(0, 10);
const pick = L.pickDailyCountry(DATA.countries, DATA.answers, 'state-size:' + expectedDay);
(async function main() {
  console.log('scenario 1: fresh game, solve immediately');
  let app = await createApp();
  let { els, storage } = app;
  assert(els['country-name'].textContent === pick.country, 'country banner shows today\'s country (got "' + els['country-name'].textContent + '")');
  assert(els['day-tag'].textContent.indexOf(expectedDay) !== -1, 'day tag shows UTC date (got "' + els['day-tag'].textContent + '")');
  assert(els['state-list'].children.length === 50, 'datalist has 50 options (got ' + els['state-list'].children.length + ')');

  submitState(els, '');
  assert(els['input-error'].hidden === false && /Pick a state/.test(els['input-error'].textContent), 'empty input shows error');
  submitState(els, 'Atlantis');
  assert(els['input-error'].hidden === false && /Not a US state/.test(els['input-error'].textContent), 'invalid state shows error');

  submitState(els, pick.state);
  assert(els['result'].hidden === false, 'result panel shown after win');
  assert(/Solved/.test(els['result-title'].textContent), 'result title says solved (got "' + els['result-title'].textContent + '")');
  assert(els['state-input'].disabled === true && els['submit-btn'].disabled === true, 'input disabled after win');
  assert(els['guess-list'].children.length === 1, 'one guess recorded (got ' + els['guess-list'].children.length + ')');
  let stats = JSON.parse(storage.getItem('state-size:stats'));
  assert(stats.played === 1 && stats.wins === 1 && stats.streak === 1, 'stats recorded once (played=' + stats.played + ' wins=' + stats.wins + ')');
  assert(storage.getItem('state-size:recorded:' + expectedDay) === 'true', 'recorded flag set');
  assert(/^State Size/.test(els['share-text'].textContent), 'share text starts with title');
  assert(els['share-text'].textContent.indexOf(pick.state) !== -1, 'share text includes answer state');

  console.log('scenario 2: replay today');
  els['replay-btn'].dispatch('click');
  assert(els['state-input'].disabled === false && els['submit-btn'].disabled === false, 'input re-enabled on replay');
  assert(els['guess-list'].children.length === 0, 'guesses cleared on replay');
  submitState(els, pick.state);
  stats = JSON.parse(storage.getItem('state-size:stats'));
  assert(stats.played === 1 && stats.wins === 1, 'replay win NOT double-counted (played=' + stats.played + ' wins=' + stats.wins + ')');

  console.log('scenario 3: lose after 6 wrong guesses');
  app = await createApp();
  els = app.els; storage = app.storage;
  const wrong = Object.keys(DATA.states).filter(s => s !== pick.state).slice(0, 6);
  for (const s of wrong) submitState(els, s);
  assert(els['result'].hidden === false, 'result shown after 6 guesses');
  assert(/Not solved/.test(els['result-title'].textContent), 'result title says not solved (got "' + els['result-title'].textContent + '")');
  stats = JSON.parse(storage.getItem('state-size:stats'));
  assert(stats.played === 1 && stats.wins === 0 && stats.streak === 0, 'loss recorded (played=' + stats.played + ' wins=' + stats.wins + ')');
  assert(els['guess-list'].children.length === 6, 'six guesses recorded (got ' + els['guess-list'].children.length + ')');

  console.log('scenario 4: duplicate guess rejected');
  app = await createApp();
  els = app.els; storage = app.storage;
  submitState(els, wrong[0]);
  submitState(els, wrong[0]);
  assert(/already guessed/i.test(els['input-error'].textContent), 'duplicate guess shows error (got "' + els['input-error'].textContent + '")');
  assert(els['guess-list'].children.length === 1, 'duplicate not added (got ' + els['guess-list'].children.length + ')');

  console.log('scenario 5: same day -> same country across instances');
  const a = await createApp(); const b = await createApp();
  assert(a.els['country-name'].textContent === b.els['country-name'].textContent, 'two instances pick same country on same day');
  assert(a.els['country-name'].textContent === pick.country, 'matches logic.pickDailyCountry');

  console.log('scenario 6: finished session restored on reload');
  let played = await createApp();
  submitState(played.els, pick.state);
  const seed = {};
  seed['state-size:session:' + expectedDay] = played.storage.getItem('state-size:session:' + expectedDay);
  seed['state-size:recorded:' + expectedDay] = played.storage.getItem('state-size:recorded:' + expectedDay);
  const restored = await createApp(seed);
  assert(restored.els['result'].hidden === false, 'finished session restored shows result');
  assert(restored.els['state-input'].disabled === true, 'input disabled on restored finished game');
  assert(restored.els['country-name'].textContent === pick.country, 'restored banner still correct');
  assert(restored.els['guess-list'].children.length === 1, 'restored one guess shown');

  console.log('\n' + passed + ' passed, ' + failed + ' failed');
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('E2E ERROR:', e); process.exit(2); });
