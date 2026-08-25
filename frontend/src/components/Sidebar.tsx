"use client";

import { useAppStore, WorkflowStep } from "@/store/appStore";

const APP_VERSION = process.env.NEXT_PUBLIC_APP_VERSION || "1.1.17";

const steps: { id: WorkflowStep; label: string; icon: string }[] = [
  { id: "import", label: "导入", icon: "M12 4v16m-8-8h16" },
  { id: "transcribe", label: "转录", icon: "M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4-8a4 4 0 018 0v1a4 4 0 01-8 0v-1z" },
  { id: "subtitle", label: "字幕", icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" },
];

export function Sidebar() {
  const { step, setStep, sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useAppStore();

  return (
    <aside
      className={`h-dvh flex flex-col border-r border-border bg-surface transition-all duration-300 relative ${
        sidebarCollapsed ? "w-16" : "w-56"
      }`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 p-4 border-b border-border">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-dim to-[rgba(37,99,235,0.14)] flex items-center justify-center shrink-0">
          <svg className="w-4.5 h-4.5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
          </svg>
        </div>
        {!sidebarCollapsed && (
          <div className="flex flex-col min-w-0">
            <span className="font-semibold text-[13px] tracking-wide text-text-primary truncate">
              SubForge
            </span>
            <span className="text-[11px] text-text-muted">AI 视频字幕工具</span>
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-7 w-6 h-6 rounded-full bg-surface border border-border flex items-center justify-center text-text-muted hover:text-text-secondary shadow-sm transition-all z-10"
      >
        <svg className={`w-3 h-3 transition-transform duration-300 ${sidebarCollapsed ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Workflow Steps */}
      <nav className="flex-1 p-2.5 space-y-0.5">
        {!sidebarCollapsed && (
          <span className="text-[11px] text-text-muted uppercase tracking-wider px-3 py-2 block">
            工作流程
          </span>
        )}
        {steps.map((s) => (
          <button
            key={s.id}
            onClick={() => { setStep(s.id); setActiveView("workflow"); }}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-all duration-200 btn-press ${
              step === s.id && activeView === "workflow"
                ? "nav-item-active"
                : "text-text-muted hover:text-text-secondary hover:bg-surface-hover"
            }`}
          >
            <div className="relative">
              <svg className="w-[18px] h-[18px] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={s.icon} />
              </svg>
              {!sidebarCollapsed && step === s.id && activeView === "workflow" && (
                <div className="absolute -left-[13px] top-1/2 -translate-y-1/2 w-[2px] h-4 bg-accent rounded-full" />
              )}
            </div>
            {!sidebarCollapsed && <span className="flex-1 text-left">{s.label}</span>}
          </button>
        ))}

        {!sidebarCollapsed && (
          <>
            <div className="h-px bg-border my-2 mx-3" />
            <button
              onClick={() => setActiveView("llm-logs")}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-all btn-press ${
                activeView === "llm-logs"
                  ? "nav-item-active"
                  : "text-text-muted hover:text-text-secondary hover:bg-surface-hover"
              }`}
            >
              <svg className="w-[18px] h-[18px] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>LLM 日志</span>
            </button>
          </>
        )}
      </nav>

      {/* Bottom */}
      <div className="p-2.5 border-t border-border space-y-0.5">
        <button
          onClick={() => setActiveView(activeView === "settings" ? "workflow" : "settings")}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-all btn-press ${
            activeView === "settings"
              ? "nav-item-active"
              : "text-text-muted hover:text-text-secondary hover:bg-surface-hover"
          }`}
        >
          <svg className="w-[18px] h-[18px] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          {!sidebarCollapsed && <span>设置</span>}
        </button>

        {!sidebarCollapsed && (
          <div className="px-3 pt-2">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent/50" />
              <span className="text-[11px] text-text-muted">v{APP_VERSION}</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
