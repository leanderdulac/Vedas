import assert from "node:assert/strict";
import { test } from "node:test";
import {
  getStoredToken,
  PIPELINE_TOKEN_KEY,
  storeToken,
} from "../src/api/pipelineToken.ts";

function fakeStorage(initial: Record<string, string> = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    map,
  };
}

test("storeToken trims and persists", () => {
  const s = fakeStorage();
  storeToken("  abc123  ", s);
  assert.equal(s.map.get(PIPELINE_TOKEN_KEY), "abc123");
  assert.equal(getStoredToken(s), "abc123");
});

test("empty token removes the key", () => {
  const s = fakeStorage({ [PIPELINE_TOKEN_KEY]: "old" });
  storeToken("", s);
  assert.equal(s.map.has(PIPELINE_TOKEN_KEY), false);
  assert.equal(getStoredToken(s), "");
});

test("missing storage is a safe no-op", () => {
  assert.equal(getStoredToken(null), "");
  assert.equal(getStoredToken(undefined), "");
  storeToken("x", null);
  storeToken("x", undefined);
});

test("stores last token across instances", () => {
  const s = fakeStorage();
  storeToken("a", s);
  storeToken("b", s);
  assert.equal(getStoredToken(s), "b");
});
