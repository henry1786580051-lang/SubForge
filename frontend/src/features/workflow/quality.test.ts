import { describe, it, expect } from "vitest";
import { analyzeSubtitleQuality } from "./quality";
import type { SubtitleSegment } from "@/lib/api";
const sub = (id: number, start: string, end: string, text = "Hello", translated = ""): SubtitleSegment => ({ id, start, end, text, translated });
describe("local subtitle checks", () => {
  it("keeps continuous timing separate from anomalies", () => {
    const result = analyzeSubtitleQuality([sub(1,"00:00:00,000","00:00:02,000"),sub(2,"00:00:02,000","00:00:04,000")]);
    expect(result.ids.continuous).toEqual([1]); expect(result.ids.timeline).toEqual([]);
    expect(result.translated).toBe(0); expect(result.ids.missing).toEqual([1,2]);
  });
  it("detects malformed, reversed and overlapping timestamps", () => {
    const result = analyzeSubtitleQuality([sub(1,"oops","00:00:02,000"),sub(2,"00:00:04,000","00:00:03,000"),sub(3,"00:00:05,000","00:00:08,000"),sub(4,"00:00:07,000","00:00:09,000")]);
    expect(result.ids.timeline).toEqual([1,2,3]);
  });
  it("uses translated display text and separates long duration from reading speed", () => {
    const result = analyzeSubtitleQuality([sub(1,"00:00:00,000","00:00:01,000","Hi","这是一个需要更多时间阅读的中文字幕"),sub(2,"00:00:02,000","00:00:12,000")]);
    expect(result.ids.reading).toEqual([1]); expect(result.ids.long).toEqual([2]); expect(result.translated).toBe(1);
  });
  it("recomputes after edits without retaining stale warnings", () => {
    const result = analyzeSubtitleQuality([sub(1,"00:00:00,000","00:00:05,000","Hi","你好")]);
    expect(result.ids.reading).toEqual([]); expect(result.ids.missing).toEqual([]);
  });
});
