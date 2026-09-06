import { beforeEach, expect, it, vi } from "vitest";
import { desktopState, handleDesktopCommand } from "./desktopChrome";
import { useAppStore } from "../store/appStore";
import { useUiStore } from "../store/uiStore";

beforeEach(() => {
  useAppStore.setState(useAppStore.getInitialState());
  useUiStore.setState(useUiStore.getInitialState());
});

it("does not export an empty or processing document", () => {
  const cancel = vi.fn();
  handleDesktopCommand("export", cancel);
  expect(useUiStore.getState().exportRequested).toBe(false);
  useAppStore.setState({ subtitleFile: "/private/example.srt", subtitles: [{ id: 1, start: "0", end: "1", text: "Hello", translated: "你好" }], isProcessing: true });
  handleDesktopCommand("export", cancel);
  expect(useUiStore.getState().exportRequested).toBe(false);
  useAppStore.setState({ isProcessing: false, activeView: "settings" });
  handleDesktopCommand("export", cancel);
  expect(useAppStore.getState().step).toBe("subtitle");
  expect(useAppStore.getState().activeView).toBe("workflow");
  expect(useUiStore.getState().exportRequested).toBe(true);
  expect(desktopState().title).toBe("example.srt");
});

it("only cancels running tasks and ignores unknown commands", () => {
  const cancel = vi.fn().mockResolvedValue(undefined);
  handleDesktopCommand("cancel", cancel);
  handleDesktopCommand("execute", cancel);
  expect(cancel).not.toHaveBeenCalled();
  useAppStore.setState({ taskStatus: "running" });
  handleDesktopCommand("cancel", cancel);
  expect(cancel).toHaveBeenCalledOnce();
});

it("routes import from settings and keeps the inspector disabled on import", () => {
  useAppStore.setState({ activeView: "settings", step: "subtitle" });
  handleDesktopCommand("import", vi.fn());
  expect(useAppStore.getState().activeView).toBe("workflow");
  expect(useAppStore.getState().step).toBe("import");
  expect(useUiStore.getState().openRequested).toBe(true);
  handleDesktopCommand("inspector", vi.fn());
  expect(useUiStore.getState().inspectorOpen).toBe(true);
});
