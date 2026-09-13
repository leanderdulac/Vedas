import { test } from 'node:test';
import assert from 'node:assert/strict';
import { citationLabel, copyCitationText, displayHitText, documentHref, hitHeading } from '../src/api/citation.ts';
import type { SearchHit } from '../src/api/client.ts';

const nasadiya: SearchHit = {
  chunk_id: 'c1',
  doc_id: 'doc-rv',
  text: 'Then was not non-existent nor existent.',
  title: 'Rig Veda selected hymns (English)',
  tradition: 'vedic',
  license: 'public-domain',
  source_url: 'fixtures/sample_texts/rigveda_selection_en.txt',
  verse_id: 'RV.10.129.1',
  locator: 'RV 10.129.1',
};

test('citation label prefers canonical locator', () => {
  assert.equal(
    citationLabel(nasadiya),
    'RV 10.129.1 — Rig Veda selected hymns (English)',
  );
  assert.equal(citationLabel({ chunk_id: 'x', text: 'a', title: 'Gītā' }), 'Gītā');
});

test('document href anchors the verse and keeps highlight fallback', () => {
  const href = documentHref(nasadiya);
  assert.ok(href);
  assert.ok(href.includes('/documento/doc-rv'));
  assert.ok(href.includes('highlight='));
  assert.ok(href.endsWith('#RV.10.129.1'));
  assert.equal(documentHref({ chunk_id: 'x', text: 'a' }), null);
});

test('hit heading stays the work title', () => {
  assert.equal(hitHeading(nasadiya), 'Rig Veda selected hymns (English)');
});

test('display text strips locator prefixes', () => {
  const text = displayHitText({
    ...nasadiya,
    text: '[RV 10.129.1] Then was not non-existent.\n\n[RV 10.129.2] Death was not then.',
  });
  assert.equal(text, 'Then was not non-existent.\n\nDeath was not then.');
});

test('copied citation leads with the locator', () => {
  const copied = copyCitationText(nasadiya);
  assert.ok(copied.startsWith('"Then was not non-existent nor existent."'));
  assert.ok(copied.includes('RV 10.129.1'));
  assert.ok(copied.includes('public-domain'));
});
