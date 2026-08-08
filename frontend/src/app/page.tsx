"use client";

import { useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { SettingsPanel } from "@/components/SettingsPanel";
import { LlmLogsPanel } from "@/components/LlmLogsPanel";
import { ToastContainer } from "@/components/Toast";
import { WorkflowWorkspace } from "@/components/WorkflowWorkspace";
import { useTaskMonitor } from "@/lib/useTaskMonitor";
import { useAppStore, WorkflowStep } from "@/store/appStore";
import { healthApi, configApi } from "@/lib/api";

const WORKFLOW_STEPS: { id: WorkflowStep; label: string }[] = [
  { id: "import", label: "导入素材" },
  { id: "transcribe", label: "语音转录" },
  { id: "subtitle", label: "字幕处理" },
];

export default function Home() {
  const {
    step,
    setStep,
    activeView,
    backendOnline,
    setBackendOnline,
    setConfigLoaded,
    taskStatus,
    taskMessage,
  } = useAppStore();
  const taskControls = useTaskMonitor();

  const currentIdx = WORKFLOW_STEPS.findIndex((s) => s.id === step);

  useEffect(() => {
    const check = () => {
      healthApi
        .check()
        .then((data) => {
          setBackendOnline(true);
          useAppStore.getState().setFfmpegOk(data.ffmpeg && data.ffprobe);
        })
        .catch(() => setBackendOnline(false));
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, [setBackendOnline]);

  // Load backend config on startup to sync with store
  useEffect(() => {
    configApi
      .get()
      .then((data: Record<string, unknown>) => {
        useAppStore.getState().setConfig({
          transcribeModel: (data.transcribe_model as string) || "whisper_cpp",
          sourceLanguage: (data.source_language as string) || "auto",
          targetLanguage: (data.target_language as string) || "chinese",
          translator: (data.translator as string) || "bing",
          llmProvider: (data.llm_provider as string) || "custom",
          llmModel: (data.llm_model as string) || "",
          needOptimize: (data.need_optimize as boolean) ?? true,
          needTranslate: (data.need_translate as boolean) ?? true,
          needReflect: (data.need_reflect as boolean) ?? false,
          customPrompt: (data.custom_prompt as string) || "",
          whisperModelSize: (data.whisper_model_size as string) || "large-v3",
          whisperxAlignmentStrategy:
            (data.whisperx_alignment_strategy as "auto" | "manual") || "auto",
          whisperxAlignModel:
            (data.whisperx_align_model as string) || "WAV2VEC2_ASR_LARGE_LV60K_960H",
          whisperxBatchSize: Number(data.whisperx_batch_size || 8),
          whisperxSupported: (data.whisperx_supported as boolean) ?? true,
          enableAudioEnhancement: (data.enable_audio_enhancement as boolean) ?? true,
          speakerDiarization:
            (data.speaker_diarization as "off" | "two" | "auto" | "fixed") || "off",
          speakerCount: Number(data.speaker_count || 2),
        });
        setConfigLoaded(true);
      })
      .catch(() => {
        setConfigLoaded(false);
        useAppStore.getState().setError("无法读取转录配置，请确认后端服务正常");
      });
  }, [setConfigLoaded]);

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar />
      <ToastContainer />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-5 py-3 border-b border-border bg-surface/95 shadow-sm">
          <div className="flex items-center">
            {WORKFLOW_STEPS.map((s, idx) => (
              <div key={s.id} className="flex items-center">
                <button
                  onClick={() => { setStep(s.id); useAppStore.getState().setActiveView("workflow"); }}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] transition-all duration-300 btn-press ${
                    step === s.id && activeView === "workflow"
                      ? "bg-accent-dim text-accent font-medium"
                      : idx < currentIdx
                      ? "text-text-secondary"
                      : "text-text-muted"
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] transition-all duration-300 ${
                      step === s.id && activeView === "workflow"
                        ? "bg-accent text-white"
                        : idx < currentIdx
                        ? "bg-accent-dim text-accent"
                        : "bg-[rgba(0,0,0,0.04)] text-text-muted border border-border"
                    }`}
                  >
                    {idx < currentIdx ? (
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      idx + 1
                    )}
                  </div>
                  <span className="hidden sm:inline">{s.label}</span>
                </button>
                {idx < WORKFLOW_STEPS.length - 1 && (
                  <div className="flex items-center px-1">
                    <div
                      className={`w-8 h-px transition-colors duration-300 ${
                        idx < currentIdx ? "bg-accent/30" : "bg-border"
                      }`}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3">
            {taskStatus === "running" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-accent-dim">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <span className="text-[11px] text-accent font-medium">
                  {taskMessage || "处理中..."}
                </span>
              </div>
            )}
            {taskStatus === "completed" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200">
                <svg className="w-3 h-3 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-[11px] text-emerald-700 font-medium">完成</span>
              </div>
            )}
            {taskStatus === "failed" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-red-50 border border-red-200">
                <span className="text-[11px] text-red-600 font-medium">失败</span>
              </div>
            )}

            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[rgba(0,0,0,0.03)]">
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-red-400"}`} />
              <span className="text-[11px] text-text-muted">
                {backendOnline ? "在线" : "离线"}
              </span>
            </div>
          </div>
        </header>

        {/* Main content */}
        {activeView === "settings" ? (
          <div className="flex-1 overflow-hidden">
            <SettingsPanel />
          </div>
        ) : activeView === "llm-logs" ? (
          <div className="flex-1 overflow-hidden">
            <LlmLogsPanel />
          </div>
        ) : (
          <WorkflowWorkspace {...taskControls} />
        )}
      </div>
    </div>
  );
}
