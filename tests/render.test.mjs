import assert from "node:assert/strict";
import { test } from "node:test";

import { render } from "../src/opsrag/static/render.mjs";

const sources = [
  { n: 1, url: "https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/#daemonset", title: "ReplicaSet > DaemonSet" },
  { n: 2, url: "https://helm.sh/docs/topics/charts_hooks/", title: "Chart Hooks" },
];

test("citations open the docs section of their source, not the app page", () => {
  const html = render("Use a DaemonSet [1][2].", sources);
  assert.match(html, /<a class="cite" href="https:\/\/kubernetes\.io\/docs\/concepts\/workloads\/controllers\/replicaset\/#daemonset" target="_blank"[^>]*>\[1\]<\/a>/);
  assert.match(html, /href="https:\/\/helm\.sh\/docs\/topics\/charts_hooks\/"[^>]*>\[2\]<\/a>/);
  assert.doesNotMatch(html, /href="#/);
});

test("citations without a matching source stay plain text", () => {
  assert.equal(render("See [7].", sources), "<p>See [7].</p>");
  assert.equal(render("See [1].", []), "<p>See [1].</p>");
});

test("code blocks and html are escaped", () => {
  const html = render("Run:\n```bash\nkubectl get pods <name> [1]\n```", sources);
  assert.match(html, /<code>kubectl get pods &lt;name&gt; \[1\]<\/code>/);
  assert.match(render("a <b>", sources), /a &lt;b&gt;/);
});
