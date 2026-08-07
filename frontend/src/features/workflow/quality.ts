import type { SubtitleSegment } from "@/lib/api";
import { parseSrtTime } from "@/lib/format";

export interface SubtitleQuality {
  overlaps: number[];
  longDurations: number[];
  tightGaps: number[];
  emptyTranslations: number;
  emptyTranslationIds: number[];
}

export function analyzeSubtitleQuality(subtitles: SubtitleSegment[]): SubtitleQuality {
  const overlaps: number[] = [];
  const longDurations: number[] = [];
  const tightGaps: number[] = [];
  const emptyTranslationIds: number[] = [];

  subtitles.forEach((subtitle, index) => {
    const start = parseSrtTime(subtitle.start);
    const end = parseSrtTime(subtitle.end);
    const next = subtitles[index + 1];
    const duration = end - start;
    if (duration > 7.5) longDurations.push(subtitle.id);
    if (!subtitle.translated.trim()) emptyTranslationIds.push(subtitle.id);
    if (next) {
      const nextStart = parseSrtTime(next.start);
      const gap = nextStart - end;
      if (gap < -0.01) overlaps.push(subtitle.id);
      if (gap >= 0 && gap < 0.08) tightGaps.push(subtitle.id);
    }
  });

  return {
    overlaps,
    longDurations,
    tightGaps,
    emptyTranslations: emptyTranslationIds.length,
    emptyTranslationIds,
  };
}
