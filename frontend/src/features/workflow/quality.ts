import type { SubtitleSegment } from "@/lib/api";

export type QualityKey = "timeline" | "reading" | "missing" | "continuous" | "long";
export const QUALITY_LABELS: Record<QualityKey, string> = {
  timeline: "时间轴异常", reading: "阅读负担", missing: "待补译", continuous: "连续衔接", long: "较长显示",
};
export interface SubtitleQuality {
  total: number;
  translated: number;
  ids: Record<QualityKey, number[]>;
  reasons: Record<QualityKey, Record<number, string>>;
}
function seconds(value: string): number {
  const match = /^(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})$/.exec(value);
  return match ? +match[1] * 3600 + +match[2] * 60 + +match[3] + +match[4] / 1000 : NaN;
}
export function analyzeSubtitleQuality(subtitles: SubtitleSegment[]): SubtitleQuality {
  const ids: SubtitleQuality["ids"] = { timeline: [], reading: [], missing: [], continuous: [], long: [] };
  const reasons: SubtitleQuality["reasons"] = { timeline: {}, reading: {}, missing: {}, continuous: {}, long: {} };
  const add = (key: QualityKey, id: number, reason: string) => {
    if (!ids[key].includes(id)) ids[key].push(id);
    reasons[key][id] = [reasons[key][id], reason].filter(Boolean).join("；");
  };
  subtitles.forEach((sub, index) => {
    const start = seconds(sub.start), end = seconds(sub.end), duration = end - start;
    if (!sub.translated.trim()) add("missing", sub.id, "此条尚无译文");
    if (!Number.isFinite(duration) || duration <= 0) {
      add("timeline", sub.id, "时间格式无效，或结束时间不晚于开始时间");
      return;
    }
    if (duration > 7.5) add("long", sub.id, `显示 ${duration.toFixed(1)} 秒，仅供参考`);
    const text = sub.translated.trim() || sub.text.trim();
    const cjk = (text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || []).length;
    const count = [...text.replace(/\s/g, "")].length;
    const limit = cjk > count / 2 ? 9 : 20;
    if (count / duration > limit) add("reading", sub.id, `${sub.translated.trim() ? "译文" : "原文"}约 ${(count / duration).toFixed(1)} 字符/秒，高于本地参考值 ${limit}；请结合播放复核`);
    const next = subtitles[index + 1];
    if (next) {
      const gap = seconds(next.start) - end;
      if (gap < -0.01) add("timeline", sub.id, `与下一条重叠 ${(-gap).toFixed(2)} 秒；多说话人场景可能合理`);
      if (gap >= 0 && gap < 0.08) add("continuous", sub.id, `与下一条间隔 ${Math.round(gap * 1000)} 毫秒，不视为错误`);
    }
  });
  return { total: subtitles.length, translated: subtitles.length - ids.missing.length, ids, reasons };
}
