/** Preserve event state across arbitrary network chunk boundaries. */
export function createSseParser(dispatch: (name: string, data: string) => void) {
  let buffer = '';
  let eventName = 'message';
  let dataLines: string[] = [];

  return (chunk: string) => {
    buffer += chunk;
    let end: number;
    while ((end = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, end).replace(/\r$/, '');
      buffer = buffer.slice(end + 1);
      if (line === '') {
        if (dataLines.length) dispatch(eventName, dataLines.join('\n'));
        dataLines = [];
        eventName = 'message';
      } else if (line.startsWith('event:')) {
        eventName = line.slice(6).replace(/^ /, '') || 'message';
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^ /, ''));
      } else if (line === 'data') {
        dataLines.push('');
      }
    }
  };
}
