const BACKEND_URL = "http://127.0.0.1:8000";
const $ = id => document.getElementById(id);
const S = { files: [], ready: false, history: [], busy: false };

const toast = (msg, err) => {
  const t = $("toast"); t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._t); t._t = setTimeout(() => t.className = "toast", 3500);
};
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// tiny markdown: code blocks, inline code, bold, bullets, paragraphs
function md(src) {
  let t = esc(src);
  t = t.replace(/```(?:\w+)?\n?([\s\S]*?)```/g, (_, c) => `<pre>${c}</pre>`);
  return t.split(/\n{2,}/).map(b => {
    if (b.startsWith("<pre>")) return b;
    if (/^\s*[-*] /.test(b)) return "<ul>" + b.split("\n").map(l => `<li>${l.replace(/^\s*[-*] /, "")}</li>`).join("") + "</ul>";
    return "<p>" + b.replace(/\n/g, "<br>") + "</p>";
  }).join("").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

const settings = () => ({
  k: +$("k").value, lambda_mult: +$("lambda_mult").value, temperature: +$("temperature").value,
  memory: +$("memory").value, model: $("model").value,
});
[["chunk_size", "o-chunk"], ["chunk_overlap", "o-overlap"], ["k", "o-k"], ["lambda_mult", "o-lambda"],
 ["temperature", "o-temp"], ["memory", "o-mem"]].forEach(([i, o]) =>
  $(i).addEventListener("input", () => $(o).textContent = $(i).value));

function render() {
  $("status").className = "status" + (S.ready ? " ready" : "");
  $("status").lastElementChild.textContent = S.ready ? "Knowledge base ready" : "No knowledge base yet";
  $("chips").innerHTML = S.ready ? S.serverFiles.map(f => `<span class="chip" title="${esc(f)}">${esc(f)}</span>`).join("") : "";
  $("queue").innerHTML = S.files.map((f, i) => `<li><span>${esc(f.name)}</span><button data-i="${i}" aria-label="Remove">✕</button></li>`).join("");
  $("build").disabled = !S.files.length;
  $("input").disabled = $("send").disabled = !S.ready || S.busy;
  $("reset").disabled = !S.ready;
  $("clear").disabled = !S.history.length;
  document.querySelectorAll("#prompts button").forEach(b => b.disabled = !S.ready);
  $("empty-sub").textContent = S.ready ? "Ask anything. Answers come only from your uploaded files." : "Add a PDF in the sidebar to begin.";
}

async function api(url, opts) {
  let r;
  try { r = await fetch(url, opts); }
  catch { throw new Error("Can't reach the server. Is uvicorn still running?"); }
  const text = await r.text();
  let d = {};
  try { d = text ? JSON.parse(text) : {}; } catch {}
  if (!r.ok || !text)
    throw new Error(d.error || `Empty response from server (HTTP ${r.status}). Check the uvicorn terminal for the error.`);
  return d;
}

// uploads
const addFiles = list => {
  const pdfs = [...list].filter(f => f.name.toLowerCase().endsWith(".pdf"));
  if (pdfs.length < list.length) toast("Only PDF files are supported.", true);
  S.files.push(...pdfs); render();
};
$("file").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
$("drop").onkeydown = e => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), $("file").click());
["dragover", "dragenter"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => addFiles(e.dataTransfer.files));
$("queue").onclick = e => { const i = e.target.dataset.i; if (i != null) { S.files.splice(i, 1); render(); } };

$("build").onclick = async () => {
  const fd = new FormData();
  S.files.forEach(f => fd.append("files", f));
  fd.append("chunk_size", $("chunk_size").value); fd.append("chunk_overlap", $("chunk_overlap").value);
  $("build").classList.add("loading"); $("build").disabled = true; $("hint").hidden = false;
  try {
    const d = await api("/api/upload", { method: "POST", body: fd });
    S.ready = true; S.serverFiles = d.files; S.files = []; S.history = []; clearFeed();
    toast(`Indexed ${d.pages} pages into ${d.chunks} chunks.`);
  } catch (e) { toast(e.message, true); }
  $("build").classList.remove("loading"); $("hint").hidden = true; render();
};

// chat
function clearFeed() { $("feed").querySelectorAll(".msg").forEach(m => m.remove()); $("empty").hidden = false; }
function addMsg(role, html, extra = "") {
  $("empty").hidden = true;
  const el = document.createElement("div"); el.className = "msg " + role;
  el.innerHTML = role === "user" ? `<div class="bubble">${html}</div>`
    : `<div class="avatar">D</div><div class="bubble"><div class="text">${html}</div>${extra}</div>`;
  $("feed").appendChild(el); $("feed").scrollTop = $("feed").scrollHeight; return el;
}

async function ask(q) {
  if (!q.trim() || S.busy || !S.ready) return;
  S.busy = true; render();
  addMsg("user", esc(q));
  const wait = addMsg("ai", '<span class="typing"><i></i><i></i><i></i></span>');
  try {
    const d = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, history: S.history, settings: settings() }) });
    const note = d.standalone.trim() !== q.trim() ? `<p class="understood">Understood as: ${esc(d.standalone)}</p>` : "";
    const src = d.sources.length ? `<details class="sources"><summary>${d.sources.length} sources retrieved</summary>` +
      d.sources.map(s => `<div class="slip"><b>${esc(s.source)} <span class="pg">p. ${s.page}</span></b><p>${esc(s.text)}…</p></div>`).join("") + "</details>" : "";
    wait.remove(); addMsg("ai", md(d.answer), note + src);
    S.history.push({ role: "user", content: q }, { role: "assistant", content: d.answer });
  } catch (e) {
    wait.remove(); addMsg("ai error", esc(e.message));
  }
  S.busy = false; render(); $("input").focus();
}

$("form").onsubmit = e => { e.preventDefault(); const q = $("input").value; $("input").value = ""; $("input").style.height = "auto"; ask(q); };
$("input").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } };
$("input").oninput = e => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; };
$("prompts").onclick = e => e.target.tagName === "BUTTON" && ask(e.target.textContent);

$("clear").onclick = () => { S.history = []; clearFeed(); render(); };
$("reset").onclick = async () => {
  if (!confirm("Delete the knowledge base and all uploaded files?")) return;
  try { await api("/api/reset", { method: "POST" }); S.ready = false; S.history = []; clearFeed(); render(); toast("Knowledge base deleted."); }
  catch (e) { toast(e.message, true); }
};
$("menu").onclick = () => $("side").classList.toggle("open");

api("/api/status").then(d => { S.ready = d.ready; S.serverFiles = d.files; render(); }).catch(() => {
  S.serverFiles = []; render();
  const b = document.createElement("div");
  b.className = "toast show err"; b.style.cssText = "top:16px;bottom:auto;max-width:90%;text-align:center;transform:translateX(-50%)";
  b.textContent = "Not connected to the DocuMind server. Run `python api.py` and open http://127.0.0.1:8000 (not Live Server or a file).";
  document.body.appendChild(b);
});