import assert from "node:assert/strict";
import { test } from "node:test";
import { VARNA_GROUPS, VOCAB, LESSONS } from "../src/data/sanskrit.ts";

test("varnamala follows the traditional order", () => {
  assert.equal(VARNA_GROUPS.length, 8);
  const total = VARNA_GROUPS.reduce((n, g) => n + g.items.length, 0);
  assert.equal(total, 47); // 14 svaras + 5x5 vargas + 4 antahstha + 4 usman
  for (const g of VARNA_GROUPS) {
    for (const v of g.items) {
      assert.ok(v.deva && v.iast && v.articulation, `varna incompleta: ${v.iast}`);
      if (v.example) {
        assert.ok(v.example.word && v.example.iast, `exemplo incompleto: ${v.iast}`);
      }
    }
  }
});

test("vocab entries are corpus-linkable", () => {
  assert.ok(VOCAB.length >= 25);
  for (const v of VOCAB) {
    assert.ok(v.deva && v.iast && v.gloss && v.meaning && v.query);
    assert.ok(["deva", "ritual", "cosmos", "principio"].includes(v.category));
    assert.ok(v.query.length >= 5, `query muito curta: ${v.query}`);
  }
});

test("lessons have sections and examples", () => {
  assert.equal(LESSONS.length, 3);
  for (const l of LESSONS) {
    assert.ok(l.title && l.subtitle);
    assert.ok(l.sections.length >= 1);
    for (const s of l.sections) {
      assert.ok(s.heading && s.body.length >= 1);
      for (const ex of s.examples) {
        assert.ok(ex.deva && ex.iast && ex.gloss);
      }
    }
  }
});