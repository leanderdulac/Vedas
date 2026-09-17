import assert from "node:assert/strict";
import { test } from "node:test";
import { RECITATIONS, echoGapMs } from "../src/data/recitation.ts";

test("curated recitations exist in canonical verse_id form", () => {
  assert.ok(RECITATIONS.length >= 5);
  for (const r of RECITATIONS) {
    assert.ok(/^[A-Z]{2,4}(\.\d+)+$/.test(r.verseId), `verse_id inválido: ${r.verseId}`);
    assert.ok(r.label && r.work && r.note, `entrada incompleta: ${r.verseId}`);
  }
});

test("echo gap grows with the pada and stays bounded", () => {
  assert.equal(echoGapMs(null), 3200);
  const short = echoGapMs({ sa: "प प" });
  const long = echoGapMs({ sa: "प".repeat(40) });
  assert.ok(short >= 3200);
  assert.ok(long > short);
  assert.ok(long <= 9000);
  // iast como fallback: mesma fórmula, medindo o texto disponível
  assert.equal(echoGapMs({ iast: "agním" }), 2400 + "agním".length * 240);
});

test("echo gap uses sa preferentially over iast", () => {
  const onlyIast = echoGapMs({ iast: "a b c d e f g" });
  const withSa = echoGapMs({ sa: "प", iast: "a b c d e f g" });
  assert.ok(withSa < onlyIast);
});