"use client";
import { useState } from "react";
import { QUALITY_LABELS, type QualityKey, type SubtitleQuality } from "@/features/workflow/quality";

export function SubtitleChecks({ quality, onSelect, active, processing = false }: {
  quality: SubtitleQuality; onSelect?: (key: QualityKey) => void; active?: QualityKey | null; processing?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!quality.total) return <p className="check-note">导入或生成字幕后显示本地检查结果。</p>;
  const percent = Math.round(quality.translated / quality.total * 100);
  const row = (key: QualityKey) => {
    const count = quality.ids[key].length;
    const warn = count > 0 && (key === "timeline" || key === "reading");
    return <button key={key} type="button" className="check-row" data-active={active === key} disabled={!count || !onSelect || processing} onClick={() => onSelect?.(key)} aria-pressed={active === key}>
      <span className="check-status" data-warn={warn} aria-hidden="true">{warn ? "!" : "·"}</span>
      <span className="check-label">{QUALITY_LABELS[key]}</span>
      <span className="check-count">{count.toLocaleString()}</span>
      <span className="check-chevron" aria-hidden="true">{count && onSelect ? "›" : ""}</span>
    </button>;
  };
  return <div className="subtitle-checks">
    <div className="check-progress"><span>翻译进度</span><strong>{processing ? "处理中" : quality.translated === 0 ? "尚无译文" : `${percent}%`}</strong></div>
    <p className="check-note">{quality.translated.toLocaleString()} / {quality.total.toLocaleString()} 条已有译文</p>
    <div className="check-track" role="progressbar" aria-label="翻译进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ transform: `scaleX(${percent / 100})` }} /></div>
    <div className="check-list">{(["timeline", "reading", "missing"] as QualityKey[]).map(row)}</div>
    <button className="check-disclosure" type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>参考项<span aria-hidden="true" style={{ transform: expanded ? "rotate(90deg)" : undefined }}>›</span></button>
    {expanded && <div className="check-details">{row("continuous")}{row("long")}<p className="check-note">连续衔接和较长显示仅供参考。阅读负担按实际显示文本估算，不代表翻译质量。</p></div>}
    <p className="check-note check-scope">本地基础检查 · 随编辑更新<br />语音覆盖与语义质量未在此检查。</p>
  </div>;
}
