// SRT / WebVTT parsing (a port of parse_subtitles in facial/transcript.py).

const TS = String.raw`(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})`;
const CUE = new RegExp(`${TS}\\s*-->\\s*${TS}`);
const TAGS = /<[^>]+>|\{[^}]*\}/g;

const seconds = (h, m, s, frac) => Number(h || 0) * 3600 + Number(m) * 60 + Number(s) + Number(frac.padEnd(3, "0")) / 1000;

export function parseSubtitles(text) {
  const segments = [];
  for (const block of text.replace(/\r\n/g, "\n").split(/\n\s*\n/)) {
    const lines = block.trim().split("\n").map((l) => l.trim());
    const i = lines.findIndex((l) => CUE.test(l));
    if (i < 0) continue;
    const g = lines[i].match(CUE).slice(1);
    const body = lines.slice(i + 1).filter(Boolean).map((l) => l.replace(TAGS, "")).join(" ").trim();
    if (body) segments.push({ start: seconds(...g.slice(0, 4)), end: seconds(...g.slice(4)), text: body });
  }
  return segments.sort((a, b) => a.start - b.start);
}
