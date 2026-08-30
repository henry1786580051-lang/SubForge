import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SubtitleFile, SubtitleSegment, TaskInfo } from "./api";

// Exercise the hook's transport callbacks without mounting a browser or opening
// a real backend connection. Effects are installed in the same order as React.
const harness = vi.hoisted(() => ({
  effects: [] as Array<() => void | (() => void)>,
  refs: [] as Array<{ current: unknown }>,
  store: {
    currentTaskId: "task-1" as string | null,
    subtitles: [] as SubtitleSegment[],
    status: "idle",
    taskStatus: "idle", isProcessing: false,
    setCurrentTaskId: vi.fn(), setTaskState: vi.fn(), setTaskAttention: vi.fn(),
    setSubtitleFile: vi.fn(), setSubtitles: vi.fn(), setError: vi.fn(), setIsProcessing: vi.fn(),
  },
  load: vi.fn(), getTask: vi.fn(), cancel: vi.fn(), start: vi.fn(),
}));

vi.mock("react", () => ({
  useRef: (current: unknown) => {
    const ref = { current };
    harness.refs.push(ref);
    return ref;
  },
  useEffect: (effect: () => void | (() => void)) => harness.effects.push(effect),
  useCallback: (callback: unknown) => callback,
}));
vi.mock("@/store/appStore", () => ({
  useAppStore: Object.assign(() => harness.store, { getState: () => harness.store }),
}));
vi.mock("@/lib/api", () => ({
  API_BASE: "http://localhost:8000",
  tasksApi: { get: harness.getTask, cancel: harness.cancel },
  subtitlesApi: { load: harness.load },
  transcribeApi: { start: harness.start }, subtitleApi: { start: harness.start },
}));
vi.mock("@/lib/taskPreview", async () => import("./taskPreview"));

import { useTaskMonitor } from "./useTaskMonitor";

const segment = (text: string): SubtitleSegment => ({
  id: 1, start: "00:00:00.000", end: "00:00:01.000", text, translated: "",
});
const task = (patch: Partial<TaskInfo> = {}): TaskInfo => ({
  id: harness.store.currentTaskId!, type: "subtitle", status: "running", progress: 50,
  message: "working", ...patch,
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
const subtitleFile = (text: string): SubtitleFile => ({
  file_path: "/tmp/file.srt", format: "srt", count: 1, segments: [segment(text)],
});

const sockets: TestWebSocket[] = [];
class TestWebSocket {
  static OPEN = 1;
  readyState = TestWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor() { sockets.push(this); }
}

describe("useTaskMonitor ordering", () => {
  let dispatch: (update: TaskInfo) => void;
  let cleanup: Array<() => void>;
  let controls: ReturnType<typeof useTaskMonitor>;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    harness.load.mockReset();
    harness.getTask.mockReset();
    harness.cancel.mockReset();
    harness.start.mockReset();
    harness.effects.length = 0;
    harness.refs.length = 0;
    sockets.length = 0;
    vi.stubGlobal("WebSocket", TestWebSocket);
    harness.store.currentTaskId = "task-1";
    harness.store.subtitles = [];
    harness.store.status = "idle";
    harness.store.taskStatus = "idle";
    harness.store.isProcessing = false;
    harness.store.setCurrentTaskId.mockImplementation((id) => { harness.store.currentTaskId = id; });
    harness.store.setIsProcessing.mockImplementation((value) => { harness.store.isProcessing = value; });
    harness.store.setSubtitles.mockImplementation((segments) => { harness.store.subtitles = segments; });
    harness.store.setTaskState.mockImplementation((_progress, _message, status) => {
      harness.store.status = status; harness.store.taskStatus = status;
    });
    controls = useTaskMonitor();
    cleanup = harness.effects.map((effect) => effect()).filter((result): result is () => void => typeof result === "function");
    dispatch = harness.refs.find((ref) => typeof ref.current === "function")!.current as typeof dispatch;
  });
  afterEach(() => {
    cleanup.forEach((effect) => effect());
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("never replaces the final editor with a late partial file or running poll", async () => {
    const pending = deferred<SubtitleFile>();
    harness.load.mockReturnValue(pending.promise);
    dispatch(task({ subtitle_file: "/tmp/partial.srt" }));
    dispatch(task({ status: "completed", result: { subtitle_file: "/tmp/final.srt", segments: [segment("final")] } }));
    pending.resolve(subtitleFile("stale"));
    await Promise.resolve();
    dispatch(task({ progress: 80 }));
    expect(harness.store.subtitles).toEqual([segment("final")]);
    expect(harness.store.status).toBe("completed");
    expect(harness.load).toHaveBeenCalledTimes(1);
  });

  it("does not read partial files when delta previews are available", () => {
    dispatch(task({ subtitle_file: "/tmp/partial.srt", preview_revision: 1,
      preview_delta: { mode: "replace", total: 1, segments: [segment("live")] } }));
    dispatch(task({ subtitle_file: "/tmp/partial.srt", preview_revision: 1, progress: 60 }));
    expect(harness.load).not.toHaveBeenCalled();
    expect(harness.store.subtitles).toEqual([segment("live")]);
  });

  it("keeps an older backend file read valid across identical polls", async () => {
    const pending = deferred<SubtitleFile>();
    harness.load.mockReturnValue(pending.promise);
    dispatch(task({ subtitle_file: "/tmp/partial.srt" }));
    dispatch(task({ subtitle_file: "/tmp/partial.srt" }));
    pending.resolve(subtitleFile("legacy live"));
    await Promise.resolve();
    expect(harness.store.subtitles).toEqual([segment("legacy live")]);
    expect(harness.load).toHaveBeenCalledTimes(1);
  });

  it("drops partial reads when a newer revision arrives", async () => {
    const pending = deferred<SubtitleFile>();
    harness.load.mockReturnValue(pending.promise);
    dispatch(task({ subtitle_file: "/tmp/partial.srt" }));
    dispatch(task({ preview_revision: 2, preview_segments: [segment("new preview")] }));
    pending.resolve(subtitleFile("stale"));
    await Promise.resolve();
    expect(harness.store.subtitles).toEqual([segment("new preview")]);
  });

  it("resets revision tracking on task switches outside startTask", () => {
    dispatch(task({ preview_revision: 99, preview_segments: [segment("old task")] }));
    harness.store.currentTaskId = "task-2";
    dispatch(task({ preview_revision: 1, preview_segments: [segment("new task")] }));
    expect(harness.store.subtitles).toEqual([segment("new task")]);
  });

  it("ignores repeated completion after a user edits the result", () => {
    const done = task({ status: "completed", result: { subtitle_file: "/tmp/final.srt", segments: [segment("final")] } });
    dispatch(done);
    harness.store.subtitles = [segment("user edit")];
    dispatch(done);
    expect(harness.store.subtitles).toEqual([segment("user edit")]);
  });

  it("does not apply an old task's recovery after switching tasks", async () => {
    const pending = deferred<SubtitleFile>();
    harness.load.mockReturnValue(pending.promise);
    dispatch(task({ status: "failed", result: { recovery_file: "/tmp/recovery.srt" } }));
    harness.store.currentTaskId = "task-2";
    dispatch(task({ preview_revision: 1, preview_segments: [segment("new task")] }));
    pending.resolve(subtitleFile("old recovery"));
    await Promise.resolve();
    expect(harness.store.subtitles).toEqual([segment("new task")]);
  });

  it("does not apply a final file load after the user has edited the preview", async () => {
    const pending = deferred<SubtitleFile>();
    harness.load.mockReturnValue(pending.promise);
    dispatch(task({ status: "completed", result: { subtitle_file: "/tmp/final.srt" } }));
    harness.store.subtitles = [segment("user correction")];
    pending.resolve(subtitleFile("machine final"));
    await Promise.resolve();
    expect(harness.store.subtitles).toEqual([segment("user correction")]);
  });

  it("does not clear a new task when an old cancellation returns late", async () => {
    const pending = deferred<unknown>();
    harness.cancel.mockReturnValue(pending.promise);
    const cancelling = controls.cancelTask();
    harness.store.currentTaskId = "task-2";
    dispatch(task({ preview_revision: 1, preview_segments: [segment("new task")] }));
    pending.resolve({ status: "cancelled" });
    await cancelling;
    expect(harness.store.currentTaskId).toBe("task-2");
    expect(harness.store.status).toBe("running");
  });

  it("cancels a newly created backend task if the user cancelled during startup", async () => {
    harness.store.currentTaskId = null;
    const pending = deferred<{ task_id: string }>();
    harness.start.mockReturnValue(pending.promise);
    const starting = controls.startTask("subtitle", {});
    await controls.cancelTask();
    pending.resolve({ task_id: "late-task" });
    await starting;
    expect(harness.cancel).toHaveBeenCalledWith("late-task");
    expect(harness.store.currentTaskId).toBeNull();
    expect(harness.store.isProcessing).toBe(false);
  });

  it("does not submit duplicate starts while the first request is pending", async () => {
    harness.store.currentTaskId = null;
    const pending = deferred<{ task_id: string }>();
    harness.start.mockReturnValue(pending.promise);
    const first = controls.startTask("subtitle", {});
    const second = controls.startTask("subtitle", {});
    pending.resolve({ task_id: "one-task" });
    await first;
    await second;
    expect(harness.start).toHaveBeenCalledTimes(1);
  });

  it("keeps monitoring a task when cancellation was not acknowledged", async () => {
    dispatch(task());
    harness.store.isProcessing = true;
    harness.cancel.mockRejectedValue(new Error("connection lost"));
    harness.getTask.mockResolvedValue(task());
    await controls.cancelTask();
    expect(harness.store.currentTaskId).toBe("task-1");
    expect(harness.store.isProcessing).toBe(true);
    expect(harness.store.setError).toHaveBeenCalledWith(expect.stringContaining("connection lost"));
  });

  it("keeps a completed result when completion races with cancellation", async () => {
    const pending = deferred<unknown>();
    harness.cancel.mockReturnValue(pending.promise);
    const cancelling = controls.cancelTask();
    dispatch(task({ status: "completed", result: { subtitle_file: "/tmp/final.srt", segments: [segment("complete")] } }));
    pending.reject(new Error("task already finished"));
    await cancelling;
    expect(harness.store.status).toBe("completed");
    expect(harness.store.subtitles).toEqual([segment("complete")]);
    expect(harness.store.currentTaskId).toBe("task-1");
    expect(harness.store.setError).not.toHaveBeenCalled();
  });

  it("reads the terminal result if a late cancellation is rejected", async () => {
    harness.cancel.mockRejectedValue(new Error("task already finished"));
    harness.getTask.mockResolvedValue(task({ status: "completed", result: { subtitle_file: "/tmp/final.srt", segments: [segment("complete")] } }));
    await controls.cancelTask();
    expect(harness.store.status).toBe("completed");
    expect(harness.store.subtitles).toEqual([segment("complete")]);
    expect(harness.store.setError).not.toHaveBeenCalled();
  });

  it("does not reconnect when an old socket closes after its replacement opens", async () => {
    const old = sockets[0];
    const connect = harness.refs.filter((ref) => typeof ref.current === "function")[1].current as () => void;
    connect();
    expect(sockets).toHaveLength(2);
    old.onclose?.();
    await vi.advanceTimersByTimeAsync(3000);
    expect(sockets).toHaveLength(2);
    sockets[1].onclose?.();
    await vi.advanceTimersByTimeAsync(3000);
    expect(sockets).toHaveLength(3);
  });

  it("ignores the previous completed task while a new task is starting", async () => {
    dispatch(task({ status: "completed", result: { subtitle_file: "/tmp/old.srt", segments: [segment("old")] } }));
    const pending = deferred<{ task_id: string }>();
    harness.start.mockReturnValue(pending.promise);
    const starting = controls.startTask("subtitle", {});
    expect(harness.store.currentTaskId).toBeNull();
    dispatch(task({ id: "task-1", status: "completed", result: { subtitle_file: "/tmp/old.srt", segments: [segment("stale")] } }));
    pending.resolve({ task_id: "new-task" });
    await starting;
    expect(harness.store.currentTaskId).toBe("new-task");
    expect(harness.store.status).toBe("running");
  });
});
