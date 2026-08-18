function has(name) {
  try { require.resolve(name); return name + ' OK'; }
  catch (e) { return 'no ' + name; }
}
console.log(has('playwright'));
console.log(has('puppeteer'));
