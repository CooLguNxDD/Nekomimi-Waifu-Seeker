"""Static HTML for the Nekomimi guessing UI.

The page talks to /api/nekomimi/* with fetch and renders through textContent /
DOM nodes only -- candidate names and blurbs come from scraped web pages and
must never be interpolated as HTML.
"""

from __future__ import annotations

NEKOMIMI_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Nekomimi</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { max-width: 720px; margin: 2rem auto; padding: 0 1rem; background: #0f1115; color: #e8eaed; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    .sub { color: #9aa0a6; margin-bottom: 1.25rem; }
    .card { margin-top: 1rem; padding: 1rem; border-radius: 12px; background: #171a21; border: 1px solid #2a2f3a; }
    .q { font-size: 1.3rem; font-weight: 700; margin: 0.25rem 0 1rem; }
    button { padding: 0.6rem 1.1rem; border: 0; border-radius: 10px; background: #7c5cff; color: #fff; font-weight: 600; cursor: pointer; }
    button.ghost { background: #232833; color: #c5c8ce; }
    button:disabled { opacity: 0.5; cursor: progress; }
    .row { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    [hidden] { display: none !important; }
    .hint { margin: 0.4rem 0 0.8rem; padding: 0.55rem 0.75rem; border-radius: 10px; background: #2a2412; color: #f1d58a; font-size: 0.9rem; }
    input[type=text] { flex: 1 1 14rem; padding: 0.55rem 0.7rem; border-radius: 10px; border: 1px solid #333; background: #1a1d24; color: inherit; }
    .meta { font-size: 0.85rem; color: #9aa0a6; margin-top: 0.75rem; }
    .bar { height: 6px; border-radius: 999px; background: #232833; overflow: hidden; margin: 0.5rem 0 0; }
    .bar > i { display: block; height: 100%; background: #7c5cff; }
    .top { margin-top: 0.75rem; font-size: 0.85rem; color: #b9bdc6; }
    .top div { display: flex; justify-content: space-between; gap: 1rem; padding: 0.15rem 0; }
    .guess img { max-width: 100%; max-height: 320px; border-radius: 10px; display: block; margin: 0.75rem 0; background: #0b0d11; }
    a { color: #9db7ff; }
    .err { color: #ff8f8f; }
  </style>
</head>
<body>
  <h1>Nekomimi</h1>
  <p class="sub">Think of a character from an <b>anime</b>, <b>manga</b>, <b>comic</b>, <b>game</b>, <b>movie</b> or <b>TV series</b>. Answer each question &mdash; yes/no or pick an option &mdash; or type a detail to help. Laya decides what to ask next.</p>

  <div id="intro" class="card">
    <div class="row">
      <input type="text" id="seed" placeholder="Optional hint (e.g. sci-fi game, silver hair)"/>
      <button id="startBtn">Start</button>
    </div>
    <div class="meta">Leave the hint blank for a cold start.</div>
  </div>

  <div id="play" class="card" hidden>
    <div class="meta" id="progress"></div>
    <div class="bar"><i id="progressBar" style="width:0%"></i></div>
    <div class="q" id="question"></div>
    <div class="hint" id="emptyHint" hidden>No candidates yet &mdash; search needs something specific. Type a series, franchise or name-like detail below (e.g. &ldquo;Vocaloid&rdquo;, &ldquo;Final Fantasy&rdquo;), or keep answering.</div>
    <div class="row" id="yesno">
      <button data-answer="yes">Yes</button>
      <button data-answer="no">No</button>
    </div>
    <div class="row" id="choices" hidden></div>
    <div class="row" style="margin-top:0.6rem">
      <input type="text" id="detail" placeholder="Add a detail instead (e.g. she pilots a mech)"/>
      <button class="ghost" id="detailBtn">Send detail</button>
    </div>
    <div class="top" id="top"></div>
  </div>

  <div id="guess" class="card guess" hidden></div>
  <div id="error" class="card err" hidden></div>

  <script>
  (function () {
    var sessionId = null;
    var busy = false;
    var el = function (id) { return document.getElementById(id); };

    function show(node, on) { node.hidden = !on; }

    function setBusy(on) {
      busy = on;
      document.querySelectorAll('button').forEach(function (b) { b.disabled = on; });
    }

    function fail(msg) {
      el('error').textContent = msg;
      show(el('error'), true);
    }

    function post(path, body) {
      setBusy(true);
      show(el('error'), false);
      return fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      }).then(function (r) { return r.json(); }).then(function (data) {
        setBusy(false);
        if (data.error) { fail(data.error); return null; }
        return data;
      }).catch(function (e) { setBusy(false); fail(String(e)); return null; });
    }

    function renderTop(list) {
      var box = el('top');
      box.textContent = '';
      (list || []).forEach(function (c) {
        var row = document.createElement('div');
        var left = document.createElement('span');
        left.textContent = c.name + (c.series ? ' \\u2014 ' + c.series : '');
        var right = document.createElement('span');
        right.textContent = Math.round((c.probability || 0) * 100) + '%';
        row.appendChild(left); row.appendChild(right);
        box.appendChild(row);
      });
    }

    function renderGuess(data) {
      show(el('play'), false);
      var box = el('guess');
      box.textContent = '';
      show(box, true);

      if (!data.guess) {
        box.textContent = data.message || 'No candidates left.';
        return;
      }
      var h = document.createElement('div');
      h.className = 'q';
      h.textContent = 'Is it ' + data.guess.name + '?';
      box.appendChild(h);

      var sub = document.createElement('div');
      sub.className = 'meta';
      sub.textContent = [data.guess.series, data.guess.medium].filter(Boolean).join(' \\u00b7 ')
        + '  \\u2014  ' + Math.round((data.guess.probability || 0) * 100) + '% confident'
        + '  (guess ' + data.guess_number + ')';
      box.appendChild(sub);

      if (data.guess.image_url) {
        var img = document.createElement('img');
        img.src = data.guess.image_url;
        img.alt = data.guess.name;
        img.referrerPolicy = 'no-referrer';
        img.loading = 'lazy';
        box.appendChild(img);
      }
      if (data.guess.blurb) {
        var p = document.createElement('p');
        p.textContent = data.guess.blurb;
        box.appendChild(p);
      }
      if (data.guess.source_url) {
        var a = document.createElement('a');
        a.href = data.guess.source_url; a.target = '_blank'; a.rel = 'noopener';
        a.textContent = 'source';
        box.appendChild(a);
      }

      var row = document.createElement('div');
      row.className = 'row';
      row.style.marginTop = '0.9rem';
      [['Yes, that\\'s it!', true], ['No, keep going', false]].forEach(function (pair) {
        var b = document.createElement('button');
        b.textContent = pair[0];
        if (!pair[1]) b.className = 'ghost';
        b.onclick = function () {
          post('/api/nekomimi/guess', { session_id: sessionId, correct: pair[1] }).then(render);
        };
        row.appendChild(b);
      });
      box.appendChild(row);
    }

    function renderDone(data) {
      show(el('play'), false);
      var box = el('guess');
      box.textContent = '';
      show(box, true);
      var h = document.createElement('div');
      h.className = 'q';
      if (data.correct && data.winner) {
        h.textContent = 'Got it: ' + data.winner.name;
      } else {
        h.textContent = data.message || 'Round over.';
      }
      box.appendChild(h);
      if (data.winner && data.winner.image_url) {
        var img = document.createElement('img');
        img.src = data.winner.image_url; img.alt = data.winner.name;
        img.referrerPolicy = 'no-referrer';
        box.appendChild(img);
      }
      var meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = 'Questions asked: ' + (data.turns || 0);
      box.appendChild(meta);
      var again = document.createElement('button');
      again.textContent = 'Play again';
      again.style.marginTop = '0.9rem';
      again.onclick = function () { location.reload(); };
      box.appendChild(again);
    }

    function renderOptions(question) {
      // Option labels come from the server's fixed question bank, but build
      // them with textContent like everything else on this page.
      var box = el('choices');
      box.textContent = '';
      var isChoice = question.kind === 'choice' && question.options;
      show(el('yesno'), !isChoice);
      show(box, !!isChoice);
      if (!isChoice) return;
      question.options.forEach(function (opt) {
        var b = document.createElement('button');
        b.textContent = opt.label;
        b.onclick = function () {
          post('/api/nekomimi/answer', { session_id: sessionId, answer: opt.key }).then(render);
        };
        box.appendChild(b);
      });
    }

    function render(data) {
      if (!data) return;
      sessionId = data.session_id || sessionId;
      show(el('intro'), false);
      if (data.stage === 'asking') {
        show(el('guess'), false);
        show(el('play'), true);
        el('question').textContent = data.question.text;
        renderOptions(data.question);
        show(el('emptyHint'), !data.candidates_alive);
        var t = data.question.turn, max = data.question.max_turns;
        el('progress').textContent = 'Question ' + t + ' of up to ' + max
          + '  \\u00b7  ' + (data.candidates_alive || 0) + ' candidates in play'
          + (data.laya ? '  \\u00b7  Laya' : '  \\u00b7  heuristics');
        el('progressBar').style.width = Math.min(100, (t / max) * 100) + '%';
        el('detail').value = '';
        renderTop(data.top);
      } else if (data.stage === 'guessing') {
        renderGuess(data);
      } else {
        renderDone(data);
      }
    }

    el('startBtn').onclick = function () {
      post('/api/nekomimi/start', { seed: el('seed').value }).then(render);
    };
    document.querySelectorAll('[data-answer]').forEach(function (b) {
      b.onclick = function () {
        post('/api/nekomimi/answer', { session_id: sessionId, answer: b.dataset.answer }).then(render);
      };
    });
    el('detailBtn').onclick = function () {
      var d = el('detail').value.trim();
      if (!d) return;
      post('/api/nekomimi/answer', { session_id: sessionId, answer: 'detail', detail: d }).then(render);
    };
    el('detail').addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !busy) el('detailBtn').click();
    });
  })();
  </script>
</body>
</html>"""
