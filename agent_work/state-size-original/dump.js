// Dev utility: print a file with line numbers. Not part of the deployed site.
var fs = require('fs');
var file = process.argv[2] || 'state-size/game.js';
var from = parseInt(process.argv[3] || '1', 10);
var to = parseInt(process.argv[4] || '0', 10);
var l = fs.readFileSync(file, 'utf8').split('\n');
if (!to) to = l.length;
for (var i = from; i <= to && i <= l.length; i++) {
  console.log(i + ': ' + l[i - 1]);
}
