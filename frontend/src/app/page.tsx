"use client";

import { useDesktopChrome } from "@/lib/desktopChrome";
import { Icon } from "@/components/Icon";
import { useUiStore } from "@/store/uiStore";
import { useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { SettingsPanel } from "@/components/SettingsPanel";
import { LlmLogsPanel } from "@/components/LlmLogsPanel";
import { FreeModelsPanel } from "@/components/FreeModelsPanel";
import { ToastContainer } from "@/components/Toast";
import { WorkflowWorkspace } from "@/components/WorkflowWorkspace";
import { useTaskMonitor } from "@/lib/useTaskMonitor";
import { useAppStore } from "@/store/appStore";
import { healthApi, configApi } from "@/lib/api";

export default function Home() {
  const {
    step,
    videoFile,
    subtitleFile,
    activeView,
    backendOnline,
    setBackendOnline,
    setConfigLoaded,
    taskStatus,
    taskMessage,
  } = useAppStore();
  const taskControls = useTaskMonitor();
  useDesktopChrome(taskControls.cancelTask);

  const { inspectorOpen, toggleInspector, appearance, nativeToolbar } = useUiStore();
  useEffect(() => {
    try {
      const saved = localStorage.getItem("subforge.appearance");
      if (saved === "light" || saved === "dark") useUiStore.getState().setAppearance(saved);
    } catch { /* Appearance can still follow the system without storage access. */ }
  }, []);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => { document.documentElement.dataset.appearance = appearance === "system" ? (media.matches ? "dark" : "light") : appearance; };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [appearance]);
  const filename = (subtitleFile || videoFile)?.split(/[\\/]/).pop();
  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key === ",") { event.preventDefault(); useAppStore.getState().setActiveView("settings"); }
      if (event.key.toLowerCase() === "o") {
        event.preventDefault();
        useAppStore.getState().setActiveView("workflow");
        useAppStore.getState().setStep("import");
        useUiStore.getState().requestOpen();
      }
    };
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, []);

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
          detectAdditionalLanguages:
            (data.detect_additional_languages as boolean) ?? false,
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
    <div className="app-shell flex h-dvh overflow-hidden">
      <Sidebar />
      <ToastContainer />

      <div className="min-w-0 flex-1 flex flex-col overflow-hidden">
        {!nativeToolbar && <header className="app-toolbar glass-surface">
          <div className="flex min-w-0 items-center gap-3">
            <Icon icon="solar:document-text-linear" width={19} className="shrink-0 text-text-muted" />
            <span className="truncate text-[13px] font-medium" title={filename}>{filename || "未命名项目"}</span>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            <span className="toolbar-status" role="status">
              <span className={`status-dot ${backendOnline ? "bg-emerald-500" : "bg-amber-500"}`} />
              {taskStatus === "running" ? (taskMessage || "处理中") : taskStatus === "failed" ? "任务未完成" : taskStatus === "completed" ? "任务已完成" : backendOnline ? "就绪" : "正在连接服务"}
            </span>
            {taskStatus === "running" && <button className="subtle-button" onClick={() => void taskControls.cancelTask()}>取消任务</button>}
            {activeView === "workflow" && step !== "import" && <button className="toolbar-button" onClick={toggleInspector} aria-pressed={inspectorOpen} title="显示或隐藏处理选项">
              <Icon icon="solar:sidebar-minimalistic-linear" width={18} />处理选项
            </button>}
          </div>
        </header>}

        {/* Main content */}
        {activeView === "settings" ? (
          <div className="flex-1 overflow-hidden">
            <SettingsPanel />
          </div>
        ) : activeView === "llm-logs" ? (
          <div className="flex-1 overflow-hidden">
            <LlmLogsPanel />
          </div>
        ) : activeView === "free-models" ? (
          <div className="flex-1 overflow-hidden">
            <FreeModelsPanel />
          </div>
        ) : (
          <WorkflowWorkspace {...taskControls} />
        )}
      </div>
    </div>
  );
}
