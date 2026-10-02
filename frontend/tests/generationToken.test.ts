import assert from "node:assert/strict";
import { test } from "node:test";
import { generationAuthHeaders } from "../src/api/generationToken.ts";

test("empty token yields no Authorization header", () => {
  assert.deepEqual(generationAuthHeaders(""), {});
  assert.deepEqual(generationAuthHeaders("   "), {});
});

test("token becomes Bearer header", () => {
  assert.deepEqual(generationAuthHeaders("secret"), {
    Authorization: "Bearer secret",
  });
  assert.deepEqual(generationAuthHeaders("  secret  "), {
    Authorization: "Bearer secret",
  });
});
