import assert from "node:assert/strict";
import { test } from "node:test";
import { displayProse, speechProse } from "../src/data/prose.ts";

test("displayProse strips markdown markers while streaming", () => {
  const raw = "### Fundamentação\n**Resposta:** o *ātman* — o Si — é sutil.\n- item um\n---\n1. dois";
  const out = displayProse(raw);
  assert.ok(!out.includes("*"));
  assert.ok(!out.includes("#"));
  assert.ok(!out.includes("—"));
  assert.ok(out.includes("o ātman, o Si, é sutil."));
  assert.ok(out.includes("item um"));
  assert.ok(out.includes("dois"));
});

test("displayProse keeps verse ranges and citations", () => {
  assert.equal(displayProse("Ver RV 1.1.1–2 [3]."), "Ver RV 1.1.1–2 [3].");
});

test("speechProse drops locator lines and citations", () => {
  const out = speechProse("**RV 1.1.1**\nLouvo Agni [1], o purohita.");
  assert.equal(out, "Louvo Agni, o purohita.");
});
