"use client";
import { Icon } from "@/components/Icon";
import { useAppStore, type WorkflowStep } from "@/store/appStore";
const stages: { id: WorkflowStep; label: string; icon: string }[] = [
  { id: "import", label: "导入素材", icon: "solar:folder-open-linear" },
  { id: "transcribe", label: "语音转录", icon: "solar:microphone-linear" },
  { id: "subtitle", label: "字幕工作区", icon: "solar:document-text-linear" },
];
export function Sidebar() {
  const { step, setStep, sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useAppStore();
  const item = (label: string, icon: string, active: boolean, action: () => void) => (
    <button key={label} onClick={action} title={label} aria-label={label} aria-current={active ? "page" : undefined}
      className={`sidebar-item ${active ? "nav-item-active" : ""}`}>
      <Icon icon={icon} width={19} />{!sidebarCollapsed && <span>{label}</span>}
    </button>
  );
  return <aside className={`app-sidebar glass-surface ${sidebarCollapsed ? "is-collapsed" : ""}`}>
    <div className="sidebar-brand"><span className="brand-mark"><Icon icon="solar:subtitles-linear" width={23} /></span>
      {!sidebarCollapsed && <span><strong>SubForge</strong><small>字幕工作室</small></span>}
    </div>
    <nav className="flex-1 p-2.5 space-y-1" aria-label="工作区">
      {!sidebarCollapsed && <p className="sidebar-caption">工作区</p>}
      {stages.map((s) => item(s.label, s.icon, activeView === "workflow" && step === s.id, () => { setStep(s.id); setActiveView("workflow"); }))}
      <div className="h-5" />
      {!sidebarCollapsed && <p className="sidebar-caption">资源</p>}
      {item("免费模型", "solar:gift-linear", activeView === "free-models", () => setActiveView("free-models"))}
    </nav>
    <div className="p-2.5 space-y-1 border-t border-border">
      {item("诊断日志", "solar:code-square-linear", activeView === "llm-logs", () => setActiveView("llm-logs"))}
      {item("设置", "solar:settings-linear", activeView === "settings", () => setActiveView("settings"))}
      {item(sidebarCollapsed ? "展开侧边栏" : "收起侧边栏", "solar:sidebar-minimalistic-linear", false, toggleSidebar)}
      {!sidebarCollapsed && <p className="px-3 pt-2 text-[11px] text-text-muted">SubForge {process.env.NEXT_PUBLIC_APP_VERSION || "1.3.0"}</p>}
    </div>
  </aside>;
}
