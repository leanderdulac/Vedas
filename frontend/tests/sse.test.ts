import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSseParser } from '../src/api/sse.ts';

test('preserves events at every possible two-chunk split', () => {
  const stream = 'event: token\ndata: {"text":"ātman"}\n\nevent: done\ndata: {}\n\n';
  for (let split = 0; split <= stream.length; split++) {
    const events: string[][] = [];
    const parse = createSseParser((name, data) => events.push([name, data]));
    parse(stream.slice(0, split));
    parse(stream.slice(split));
    assert.deepEqual(events, [['token', '{"text":"ātman"}'], ['done', '{}']]);
  }
});
test('handles CRLF, multiline data, comments and one-character chunks', () => {
  const events: string[][] = [];
  const parse = createSseParser((name, data) => events.push([name, data]));
  for (const char of ': ping\r\nevent: token\r\ndata: first\r\ndata:  second\r\n\r\ndata: third\r\n\r\n') parse(char);
  assert.deepEqual(events, [['token', 'first\n second'], ['message', 'third']]);
});
test('does not dispatch unfinished events', () => {
  const events: string[][] = [];
  const parse = createSseParser((name, data) => events.push([name, data]));
  parse('event: done\ndata: {}\n');
  assert.deepEqual(events, []);
  parse('\n');
  assert.deepEqual(events, [['done', '{}']]);
});
