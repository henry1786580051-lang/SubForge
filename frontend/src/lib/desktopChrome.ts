"use client";

import { useEffect } from "react";
import { useAppStore } from "../store/appStore";
import { useUiStore } from "../store/uiStore";

export type DesktopCapabilities = {
  toolbar: boolean;
  liquid_glass?: boolean;
  reduce_transparency?: boolean;
  reduce_motion?: boolean;
  increase_contrast?: boolean;
};
type DesktopApi = {
  get_desktop_state?: () => Promise<DesktopCapabilities>;
  sync_desktop_state?: (state: ReturnType<typeof desktopState>) => Promise<DesktopCapabilities>;
};

export function desktopState(app = useAppStore.getState(), ui = useUiStore.getState()) {
  return {
    title: (app.subtitleFile || app.videoFile)?.split(/[\\/]/).pop() || "SubForge",
    status: app.taskStatus === "running" ? (app.taskMessage || "处理中") : app.taskStatus === "failed" ? "任务未完成" : app.taskStatus === "completed" ? "任务已完成" : app.backendOnline ? "字幕工作室 · 就绪" : "正在连接服务",
    appearance: ui.appearance,
    can_export: !!app.subtitleFile && app.subtitles.length > 0 && !app.isProcessing,
    can_inspect: app.activeView === "workflow" && app.step !== "import",
    inspector_open: ui.inspectorOpen,
    running: app.taskStatus === "running",
  };
}

export function handleDesktopCommand(command: unknown, cancelTask: () => Promise<void>) {
  const app = useAppStore.getState();
  const ui = useUiStore.getState();
  switch (command) {
    case "sidebar": app.toggleSidebar(); break;
    case "import": app.setActiveView("workflow"); app.setStep("import"); ui.requestOpen(); break;
    case "inspector": if (app.activeView === "workflow" && app.step !== "import") ui.toggleInspector(); break;
    case "export":
      if (!desktopState().can_export) break;
      app.setActiveView("workflow"); app.setStep("subtitle"); ui.requestExport(); break;
    case "cancel": if (app.taskStatus === "running") void cancelTask(); break;
  }
}

export function useDesktopChrome(cancelTask: () => Promise<void>) {
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let previous = "";
    let connected = false;
    const api = () => (window as unknown as { pywebview?: { api?: DesktopApi } }).pywebview?.api;
    const apply = (caps: DesktopCapabilities) => {
      if (disposed) return;
      connected = caps.toolbar;
      useUiStore.getState().setNativeToolbar(caps.toolbar);
      const root = document.documentElement;
      root.dataset.nativeToolbar = String(caps.toolbar);
      root.dataset.reduceTransparency = String(!!caps.reduce_transparency);
      root.dataset.reduceMotion = String(!!caps.reduce_motion);
      root.dataset.increaseContrast = String(!!caps.increase_contrast);
    };
    const sync = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!connected || disposed) return;
        const state = desktopState();
        const serialized = JSON.stringify(state);
        if (serialized === previous) return;
        previous = serialized;
        api()?.sync_desktop_state?.(state).then(apply).catch(() => { previous = ""; });
      }, 80);
    };
    const connect = () => {
      api()?.get_desktop_state?.().then((caps) => { apply(caps); if (caps.toolbar) { previous = ""; sync(); } }).catch(() => {});
    };
    const command = (event: Event) => handleDesktopCommand((event as CustomEvent).detail, cancelTask);
    const changed = (event: Event) => apply((event as CustomEvent<DesktopCapabilities>).detail);
    window.addEventListener("subforge:command", command);
    window.addEventListener("subforge:desktop", changed);
    window.addEventListener("pywebviewready", connect);
    const unsubscribeApp = useAppStore.subscribe(sync);
    const unsubscribeUi = useUiStore.subscribe(sync);
    connect();
    // The first bridge-ready event can precede React hydration.
    const retry = window.setInterval(() => { if (!connected) connect(); }, 1500);
    return () => {
      disposed = true; clearTimeout(timer); clearInterval(retry);
      unsubscribeApp(); unsubscribeUi();
      window.removeEventListener("subforge:command", command);
      window.removeEventListener("subforge:desktop", changed);
      window.removeEventListener("pywebviewready", connect);
    };
  }, [cancelTask]);
}
