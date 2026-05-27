"use client";

import { useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { VideoPanel } from "@/components/VideoPanel";
import { SubtitlePanel } from "@/components/SubtitlePanel";
import { ConfigPanel } from "@/components/ConfigPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { LlmLogsPanel } from "@/components/LlmLogsPanel";
import { ToastContainer } from "@/components/Toast";
import { useAppStore, WorkflowStep } from "@/store/appStore";
import { healthApi } from "@/lib/api";

const WORKFLOW_STEPS: { id: WorkflowStep; label: string }[] = [
  { id: "import", label: "导入媒体" },
  { id: "transcribe", label: "语音转录" },
  { id: "subtitle", label: "字幕处理" },
];

export default function Home() {
  const { step, setStep, activeView, backendOnline, setBackendOnline, taskStatus, taskMessage } =
    useAppStore();

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

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <ToastContainer />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-5 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]" style={{ boxShadow: "var(--shadow)" }}>
          <div className="flex items-center">
            {WORKFLOW_STEPS.map((s, idx) => (
              <div key={s.id} className="flex items-center">
                <button
                  onClick={() => { setStep(s.id); useAppStore.getState().setActiveView("workflow"); }}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px] transition-all duration-300 ${
                    step === s.id && activeView === "workflow"
                      ? "bg-[var(--accent-dim)] text-[var(--accent)] font-medium"
                      : idx < currentIdx
                      ? "text-[var(--text-secondary)]"
                      : "text-[var(--text-muted)]"
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] transition-all duration-300 ${
                      step === s.id && activeView === "workflow"
                        ? "bg-[var(--accent)] text-white"
                        : idx < currentIdx
                        ? "bg-[var(--accent-dim)] text-[var(--accent)]"
                        : "bg-[rgba(0,0,0,0.04)] text-[var(--text-muted)] border border-[var(--border)]"
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
                        idx < currentIdx ? "bg-[var(--accent)]/30" : "bg-[var(--border)]"
                      }`}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3">
            {taskStatus === "running" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[var(--accent-dim)]">
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
                <span className="text-[10px] text-[var(--accent)] font-medium">
                  {taskMessage || "处理中..."}
                </span>
              </div>
            )}
            {taskStatus === "completed" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200">
                <svg className="w-3 h-3 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-[10px] text-emerald-700 font-medium">完成</span>
              </div>
            )}
            {taskStatus === "failed" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-red-50 border border-red-200">
                <span className="text-[10px] text-red-600 font-medium">失败</span>
              </div>
            )}

            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[rgba(0,0,0,0.03)]">
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-red-400"}`} />
              <span className="text-[10px] text-[var(--text-muted)]">
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
          <>
            <div className="flex-1 flex overflow-hidden">
              <div className="w-1/2 border-r border-[var(--border)] overflow-hidden">
                <VideoPanel />
              </div>
              <div className="w-1/2 overflow-hidden">
                <SubtitlePanel />
              </div>
            </div>
            <ConfigPanel />
          </>
        )}
      </div>
    </div>
  );
}
