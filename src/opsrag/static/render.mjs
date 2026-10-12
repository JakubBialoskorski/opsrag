// Answer rendering, kept out of index.html so it can be tested with `node --test`

export function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// [n] opens the same docs section as source n in the list; numbers without a source stay plain text
function citation(n, sources) {
  const source = sources.find((s) => s.n === Number(n));
  if (!source) return `[${n}]`;
  return `<a class="cite" href="${escapeHtml(source.url)}" target="_blank" rel="noopener" ` +
    `title="${escapeHtml(source.title)}">[${n}]</a>`;
}

function inline(text, sources) {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, (_, n) => citation(n, sources));
}

// Small markdown subset: fenced code blocks (with a copy button), inline code, bold, [n] citations
export function render(text, sources = []) {
  const parts = text.split(/```[\w-]*\n?/);
  return parts.map((part, i) => {
    if (i % 2 === 1) {
      return `<div class="code"><button class="copy" type="button">Copy</button><pre><code>${escapeHtml(part.replace(/\n$/, ""))}</code></pre></div>`;
    }
    return part.trim() ? `<p>${inline(part.trim(), sources)}</p>` : "";
  }).join("");
}
