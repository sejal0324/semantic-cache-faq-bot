"""A single-page demo UI for the live demo / video. No build step, no external assets.

Every value from the server (LLM answers included) is written with textContent, never as
HTML, so a poisoned or hallucinated answer cannot run script in the viewer's browser.
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic cache FAQ bot</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}
h1{font-size:1.4rem}form{display:flex;gap:.5rem;margin:1rem 0}
input{flex:1;padding:.6rem;border:1px solid #bbb;border-radius:8px;font:inherit}
button{padding:.6rem 1rem;border:0;border-radius:8px;background:#1d4ed8;color:#fff;font:inherit}
.msg{border:1px solid #e5e5e5;border-radius:10px;padding:.75rem 1rem;margin:.6rem 0}
.q{font-weight:600}.meta{font-size:.85rem;margin-top:.35rem}
.hit{color:#047857}.miss{color:#b45309}
</style></head><body>
<h1>BazaarNow support: semantic cache demo</h1>
<p>Ask the same question in different words (English or Hinglish). The first goes to the LLM;
paraphrases come back from the cache in milliseconds.</p>
<form id="f"><input id="q" placeholder="e.g. refund kab milega?" maxlength="500" required>
<button>Ask</button></form>
<div id="log"></div>
<script>
const f = document.getElementById("f"), q = document.getElementById("q"),
      log = document.getElementById("log");
f.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = q.value.trim(); if (!question) return;
  const box = document.createElement("div"); box.className = "msg";
  const qq = document.createElement("div"); qq.className = "q"; qq.textContent = question;
  const a = document.createElement("div"); a.textContent = "...";
  const meta = document.createElement("div"); meta.className = "meta";
  box.append(qq, a, meta); log.prepend(box);
  try {
    const r = await fetch("/ask", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({tenant_id: "bazaarnow", question})});
    const d = await r.json();
    if (!r.ok) { a.textContent = "Error: " + (d.detail || r.status); return; }
    a.textContent = d.answer;
    meta.className = "meta " + (d.cached ? "hit" : "miss");
    meta.textContent = (d.cached ? "CACHE HIT" : "LLM") + " · " + d.latency_ms + " ms · similarity "
      + d.similarity + (d.source ? " · FAQ " + d.source : "");
  } catch (err) { a.textContent = "Network error"; }
});
</script></body></html>
"""

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
