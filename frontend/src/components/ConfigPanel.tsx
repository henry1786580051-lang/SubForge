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
    if (step === "transcribe" || !subtitleFile) {
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
    <div className="border-t border-border bg-gradient-to-b from-surface to-[rgba(0,0,0,0.01)] relative shadow-sm">
      {isProcessing && (
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-[rgba(0,0,0,0.04)]">
          <div className="h-full bg-accent transition-all duration-500 progress-glow" style={{ width: `${taskProgress}%` }} />
        </div>
      )}

      {error && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-red-50 border border-red-200 text-red-600 text-[12px] flex items-center gap-2 shadow-sm">
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
            <span className="text-[11px] text-text-secondary">{taskMessage}</span>
            <span className="text-[11px] text-text-muted tabular-nums">{taskProgress}%</span>
          </div>
        )}

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          {isProcessing ? (
            <button onClick={cancelTask} className="px-3 py-1.5 text-[12px] rounded-full bg-red-50 text-red-600 hover:bg-red-100 transition-all border border-red-200 font-medium btn-press">取消</button>
          ) : (
            <>
              <button onClick={() => {
                cancelTask();
                useAppStore.getState().setVideoFile(null);
                useAppStore.getState().setFileInfo(null);
                useAppStore.getState().setSubtitleFile(null);
                useAppStore.getState().setSubtitles([]);
                useAppStore.getState().setStep("import");
                useAppStore.getState().setCurrentTaskId(null);
                useAppStore.getState().setTaskState(0, "", "idle");
              }} className="px-3 py-1.5 text-[12px] rounded-md text-text-secondary hover:text-text-primary hover:bg-surface-hover transition-all border border-border btn-press">
                重置
              </button>
              <button onClick={handleStart} disabled={!videoFile}
                className="px-4 py-1.5 text-[12px] rounded-full bg-accent text-white hover:bg-accent-hover transition-all font-medium disabled:opacity-30 disabled:cursor-not-allowed btn-press shadow-md">
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
        checked ? "bg-accent/20" : "bg-[rgba(0,0,0,0.06)]"
      }`}>
        <div className={`absolute w-[11px] h-[11px] rounded-full transition-all duration-200 shadow-sm ${
          checked ? "left-[13px] bg-accent" : "left-[2px] bg-text-muted"
        }`} />
      </div>
      <span className="text-[12px] text-text-muted group-hover:text-text-secondary transition-colors">{label}</span>
    </button>
  );
}
