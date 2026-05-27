"use client";

import { useState } from "react";
import { useAppStore } from "@/store/appStore";
import { useTaskMonitor } from "@/lib/useTaskMonitor";
import { configApi } from "@/lib/api";

export function ConfigPanel() {
  const { videoFile, subtitleFile, config, setConfig, isProcessing, setIsProcessing, taskProgress, taskMessage, error, setError, step } = useAppStore();
  const { startTask, cancelTask } = useTaskMonitor();
  const [promptExpanded, setPromptExpanded] = useState(false);

  const handleStart = async () => {
    if (!videoFile) { setError("请先导入视频文件"); return; }
    setIsProcessing(true); setError(null);
    if (!subtitleFile) {
      await startTask("transcribe", {
        file_path: videoFile,
        model: config.transcribeModel,
        language: config.sourceLanguage,
      });
    } else {
      await startTask("subtitle", {
        subtitle_file: subtitleFile,
        target_language: config.targetLanguage,
        translator: config.translator,
        need_optimize: config.needOptimize,
        need_translate: config.needTranslate,
        need_reflect: config.needReflect,
        custom_prompt: config.customPrompt || undefined,
      });
    }
  };

  return (
    <div className="border-t border-[var(--border)] bg-[var(--surface)] relative" style={{ boxShadow: "0 -1px 3px rgba(0,0,0,0.04)" }}>
      {isProcessing && (
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-[rgba(0,0,0,0.04)]">
          <div className="h-full bg-[var(--accent)] transition-all duration-500 progress-glow" style={{ width: `${taskProgress}%` }} />
        </div>
      )}

      {error && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-red-50 border border-red-200 text-red-600 text-[11px] flex items-center gap-2" style={{ boxShadow: "var(--shadow)" }}>
          <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 ml-1">&times;</button>
        </div>
      )}

      <div className="px-5 py-2.5 flex items-center gap-4">

        {step === "subtitle" && (
          <>
            <ToggleSwitch label="优化字幕" checked={config.needOptimize} onChange={(v) => setConfig({ needOptimize: v })} disabled={isProcessing} />
            <ToggleSwitch label="翻译字幕" checked={config.needTranslate} onChange={(v) => setConfig({ needTranslate: v })} disabled={isProcessing} />
          </>
        )}

        {isProcessing && (
          <div className="flex items-center gap-2 ml-2">
            <span className="text-[10px] text-[var(--text-secondary)]">{taskMessage}</span>
            <span className="text-[10px] text-[var(--text-muted)] tabular-nums">{taskProgress}%</span>
          </div>
        )}

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          {isProcessing ? (
            <button onClick={cancelTask} className="px-3 py-1.5 text-[11px] rounded-full bg-red-50 text-red-600 hover:bg-red-100 transition-all border border-red-200 font-medium">取消</button>
          ) : (
            <>
              <button onClick={() => {
                useAppStore.getState().setVideoFile(null);
                useAppStore.getState().setFileInfo(null);
                useAppStore.getState().setSubtitleFile(null);
                useAppStore.getState().setSubtitles([]);
                useAppStore.getState().setStep("import");
              }} className="px-3 py-1.5 text-[11px] rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[rgba(0,0,0,0.04)] transition-all border border-[var(--border)]">
                重置
              </button>
              <button onClick={handleStart} disabled={!videoFile}
                className="px-4 py-1.5 text-[11px] rounded-full bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-all font-medium disabled:opacity-30 disabled:cursor-not-allowed"
                style={{ boxShadow: "0 2px 8px rgba(212,149,106,0.25)" }}>
                {step === "transcribe" ? "开始转录" : config.needTranslate ? "开始翻译" : "处理字幕"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ToggleSwitch({ label, checked, onChange, disabled }: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button onClick={() => !disabled && onChange(!checked)} className={`flex items-center gap-2 group ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}>
      <div className={`w-7 h-[15px] rounded-full transition-all duration-200 relative flex items-center ${
        checked ? "bg-[var(--accent)]/20" : "bg-[rgba(0,0,0,0.06)]"
      }`}>
        <div className={`absolute w-[11px] h-[11px] rounded-full transition-all duration-200 shadow-sm ${
          checked ? "left-[13px] bg-[var(--accent)]" : "left-[2px] bg-[var(--text-muted)]"
        }`} />
      </div>
      <span className="text-[11px] text-[var(--text-muted)] group-hover:text-[var(--text-secondary)] transition-colors">{label}</span>
    </button>
  );
}
