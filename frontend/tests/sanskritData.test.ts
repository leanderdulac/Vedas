import assert from "node:assert/strict";
import { test } from "node:test";
import {
  VARNA_GROUPS,
  VOCAB,
  LESSONS,
  devaToPhonetic,
  iastToSpeakable,
  pickSanskritVoice,
  pickPortugueseVoice,
  pickLatinVoice,
  planSanskritSpeech,
} from "../src/data/sanskrit.ts";

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

test("pickSanskritVoice selects hindi or indic fallback", () => {
  // Sem window no Node: retorna null seguramente
  assert.equal(pickSanskritVoice(), null);

  // Simula speechSynthesis com lista de vozes
  const fakeVoices = [
    { lang: "en-US", name: "Samantha" },
    { lang: "pt-BR", name: "Luciana" },
    { lang: "hi-IN", name: "Lekha" },
  ];
  const globalAny = globalThis as unknown as {
    window: { speechSynthesis: { getVoices: () => typeof fakeVoices } };
  };
  globalAny.window = {
    speechSynthesis: {
      getVoices: () => fakeVoices,
    },
  };
  try {
    const picked = pickSanskritVoice();
    assert.ok(picked);
    assert.equal(picked.lang, "hi-IN");
  } finally {
    delete (globalThis as Record<string, unknown>).window;
  }
});

test("pickPortugueseVoice selects pt voice", () => {
  const fakeVoices = [
    { lang: "en-US", name: "Alex" },
    { lang: "pt-BR", name: "Felipe" },
  ];
  const globalAny = globalThis as unknown as {
    window: { speechSynthesis: { getVoices: () => typeof fakeVoices } };
  };
  globalAny.window = {
    speechSynthesis: {
      getVoices: () => fakeVoices,
    },
  };
  try {
    const picked = pickPortugueseVoice();
    assert.ok(picked);
    assert.equal(picked.lang, "pt-BR");
  } finally {
    delete (globalThis as Record<string, unknown>).window;
  }
});

test("devaToPhonetic converts devanagari into pronounceable latin syllables", () => {
  assert.equal(devaToPhonetic("अग्नि"), "agni");
  assert.equal(devaToPhonetic("पुरुष"), "purusha");
  assert.equal(devaToPhonetic("गायत्री"), "gaayatree");
  assert.equal(devaToPhonetic("English text"), "English text");
});

test("iastToSpeakable spells syllables a latin voice can say", () => {
  assert.equal(iastToSpeakable("a"), "uh");
  assert.equal(iastToSpeakable("ā"), "aah");
  assert.equal(iastToSpeakable("i"), "ih");
  assert.equal(iastToSpeakable("ī"), "ee");
  assert.equal(iastToSpeakable("u"), "oo");
  assert.equal(iastToSpeakable("e"), "eh");
  assert.equal(iastToSpeakable("ai"), "eye");
  assert.equal(iastToSpeakable("o"), "oh");
  assert.equal(iastToSpeakable("au"), "ow");
  assert.equal(iastToSpeakable("ka"), "kuh");
  assert.equal(iastToSpeakable("kha"), "kuh-huh");
  assert.equal(iastToSpeakable("ga"), "guh");
  assert.equal(iastToSpeakable("gha"), "guh-huh");
  assert.equal(iastToSpeakable("ca"), "chuh");
  assert.equal(iastToSpeakable("cha"), "chuh-huh");
  assert.equal(iastToSpeakable("tha"), "tuh-huh");
  assert.equal(iastToSpeakable("pha"), "puh-huh");
  assert.equal(iastToSpeakable("śa"), "shuh");
  assert.equal(iastToSpeakable("ṣa"), "shruh");
  assert.equal(iastToSpeakable("ṛ"), "rih");
  assert.equal(iastToSpeakable("oṃ"), "ohm");
  assert.equal(iastToSpeakable("agni"), "uh gnih");
  assert.equal(iastToSpeakable("īḷe"), "ee leh");
  assert.equal(iastToSpeakable("dharma"), "duh-huh rmuh");
});

test("planSanskritSpeech reads devanagari only with an indic voice", () => {
  const hindi = { lang: "hi-IN", name: "Lekha" } as SpeechSynthesisVoice;
  const indic = planSanskritSpeech("अग्नि", "agni", hindi, null);
  assert.equal(indic.text, "अग्नि");
  assert.equal(indic.lang, "hi-IN");

  const english = { lang: "en-US", name: "Samantha" } as SpeechSynthesisVoice;
  const latin = planSanskritSpeech("अग्नि", "agni", null, english);
  assert.equal(latin.text, "uh gnih");
  assert.equal(latin.lang, "en-US");
  assert.equal(latin.voice, english);

  const phonetic = planSanskritSpeech("क", "ka");
  assert.equal(phonetic.text, "kuh");
  assert.equal(phonetic.lang, "en-US");
});

test("pickLatinVoice prefers English over Portuguese", () => {
  const fakeVoices = [
    { lang: "pt-BR", name: "Luciana" },
    { lang: "en-US", name: "Samantha" },
  ];
  const globalAny = globalThis as unknown as {
    window: { speechSynthesis: { getVoices: () => typeof fakeVoices } };
  };
  globalAny.window = {
    speechSynthesis: { getVoices: () => fakeVoices },
  };
  try {
    const picked = pickLatinVoice();
    assert.ok(picked);
    assert.equal(picked.lang, "en-US");
    assert.equal(pickSanskritVoice(), null);
  } finally {
    delete (globalThis as Record<string, unknown>).window;
  }
});