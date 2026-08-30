"use client";

import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/store/appStore";
import {
  tasksApi,
  transcribeApi,
  subtitleApi,
  subtitlesApi,
  API_BASE,
  type TaskInfo,
} from "@/lib/api";
import { mergeTaskPreview } from "@/lib/taskPreview";

export type TaskStarter = (
  type: "transcribe" | "subtitle",
  payload: Record<string, unknown>
) => Promise<void>;

function getWebSocketBase(): string {
  if (API_BASE) return API_BASE.replace(/^http/, "ws");
  if (typeof window === "undefined") return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

function subscribeToTask(ws: WebSocket, taskId: string | null) {
  if (taskId && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "subscribe", task_id: taskId }));
  }
}

export function useTaskMonitor() {
  const {
    currentTaskId,
    setCurrentTaskId,
    setTaskState,
    setTaskAttention,
    setSubtitleFile,
    setSubtitles,
    setError,
    setIsProcessing,
  } = useAppStore();

  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const aliveRef = useRef(true);
  const handleTaskUpdateRef = useRef<(task: TaskInfo) => void>(() => {});
  const connectWsRef = useRef<() => void>(() => {});

  // Track task update signature. Partial SRT files are updated in-place, and
  // optimization/translation often changes text without changing segment count.
  const lastPartialKeyRef = useRef<string | null>(null);
  const previewRevisionRef = useRef(0);
  const updateGuardRef = useRef({ taskId: "", terminal: false, generation: 0 });
  const actionGenerationRef = useRef(0);
  const pendingStartRef = useRef<number | null>(null);

  function handleTaskUpdate(task: TaskInfo) {
    // Guard against stale task updates (e.g., after cancel)
    const store = useAppStore.getState();
    if (task.id !== store.currentTaskId) return;
    const guard = updateGuardRef.current;
    if (guard.taskId !== task.id) {
      guard.taskId = task.id;
      guard.terminal = false;
      guard.generation++;
      previewRevisionRef.current = 0;
      lastPartialKeyRef.current = null;
    }
    // Terminal updates are sticky: an in-flight poll must not restart a task or
    // replace the completed editor with an older partial file.
    if (guard.terminal) return;
    guard.terminal = ["completed", "failed", "cancelled"].includes(task.status);
    const partialKey = `${task.subtitle_file}:${task.progress}:${task.message}`;
    const legacyPreview = task.status === "running" && task.subtitle_file
      && !task.preview_revision && !task.preview_delta && !task.preview_segments;
    if (guard.terminal || (task.preview_revision || 0) > previewRevisionRef.current
      || (legacyPreview && partialKey !== lastPartialKeyRef.current)) {
      guard.generation++;
    }
    const generation = guard.generation;
    let expectedSubtitles = store.subtitles;
    const isCurrentRead = () => aliveRef.current
      && task.id === useAppStore.getState().currentTaskId
      && generation === updateGuardRef.current.generation
      && expectedSubtitles === useAppStore.getState().subtitles;

    const uiStatus = task.status === "cancelled" ? "idle" : task.status;
    setTaskState(
      task.progress,
      task.message,
      uiStatus as "idle" | "running" | "completed" | "failed"
    );
    setTaskAttention(task.attention || null);

    // Prefer the WebSocket snapshot. Reading an SRT while a worker is replacing
    // it can return stale or partially written content in packaged builds.
    const previewRevision = task.preview_revision || 0;
    const previewUpdate = mergeTaskPreview(
      useAppStore.getState().subtitles,
      task,
      previewRevisionRef.current
    );
    if (previewUpdate) {
      setSubtitles(previewUpdate.segments);
      previewRevisionRef.current = previewUpdate.revision;
    }
    expectedSubtitles = useAppStore.getState().subtitles;

    // Compatibility fallback for older backends that only expose a partial file.
    if (legacyPreview && task.subtitle_file) {
      if (partialKey === lastPartialKeyRef.current) return;
      lastPartialKeyRef.current = partialKey;
      subtitlesApi
        .load(task.subtitle_file)
        .then((subFile) => {
          if (!isCurrentRead()) return;
          if (!task.preview_segments) setSubtitles(subFile.segments);
        })
        .catch(() => {});
    }

    if (task.status === "completed") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);
      setTaskAttention(null);

      const result = task.result;
      if (task.type === "transcribe" && result?.subtitle_file) {
        setSubtitleFile(result.subtitle_file as string);
        if (Array.isArray(result.segments)) {
          setSubtitles(result.segments as import("@/lib/api").SubtitleSegment[]);
          return;
        }
        subtitlesApi
          .load(result.subtitle_file as string)
          .then((subFile) => {
            if (!isCurrentRead()) return;
            setSubtitles(subFile.segments);
          })
          .catch((err) => {
            if (!isCurrentRead()) return;
            setError(err instanceof Error ? err.message : "Failed to load subtitles");
          });
      }
      if (task.type === "subtitle" && result?.subtitle_file) {
        setSubtitleFile(result.subtitle_file as string);
        if (Array.isArray(result.segments)) {
          setSubtitles(result.segments as import("@/lib/api").SubtitleSegment[]);
          return;
        }
        if (Array.isArray(task.preview_segments)) {
          setSubtitles(task.preview_segments);
          previewRevisionRef.current = Math.max(previewRevisionRef.current, previewRevision);
          return;
        }
        const completedRevision = Number(result.preview_revision || 0);
        if (completedRevision > 0 && completedRevision <= previewRevisionRef.current) {
          return;
        }
        subtitlesApi
          .load(result.subtitle_file as string)
          .then((subFile) => {
            if (!isCurrentRead()) return;
            setSubtitles(subFile.segments);
          })
          .catch((err) => {
            if (!isCurrentRead()) return;
            setError(err instanceof Error ? err.message : "Failed to load translated subtitles");
          });
      }
    }

    if (task.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);
      setTaskAttention(null);
      setError(task.error || "Task failed");
      const recoveryFile = task.result?.recovery_file;
      if (typeof recoveryFile === "string" && recoveryFile) {
        setSubtitleFile(recoveryFile);
        if (!task.preview_segments) {
          subtitlesApi
            .load(recoveryFile)
            .then((subFile) => {
              if (!isCurrentRead()) return;
              setSubtitles(subFile.segments);
            })
            .catch(() => {});
        }
      }
    }

    if (task.status === "cancelled") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);
      setTaskAttention(null);
      setCurrentTaskId(null);
    }
  }

  // Keep ref in sync with latest handleTaskUpdate
  useEffect(() => {
    handleTaskUpdateRef.current = handleTaskUpdate;
  });

  // Connect WebSocket for real-time updates
  const connectWs = useCallback(() => {
    // Close existing connection first
    if (wsRef.current) {
      const previous = wsRef.current;
      wsRef.current = null;
      previous.close();
    }
    try {
      const ws = new WebSocket(`${getWebSocketBase()}/ws/tasks`);
      ws.onopen = () => {
        subscribeToTask(ws, useAppStore.getState().currentTaskId);
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "task_update" && msg.data) {
            const task: TaskInfo = msg.data;
            const store = useAppStore.getState();
            if (task.id === store.currentTaskId) {
              handleTaskUpdateRef.current(task);
            }
          }
        } catch {
          // ignore
        }
      };
      ws.onerror = () => {};
      ws.onclose = () => {
        if (aliveRef.current && wsRef.current === ws) {
          wsRef.current = null;
          wsTimeoutRef.current = setTimeout(() => connectWsRef.current(), 3000);
        }
      };
      wsRef.current = ws;
    } catch {
      // WebSocket not available
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    connectWsRef.current = connectWs;
    connectWs();
    return () => {
      aliveRef.current = false;
      if (wsTimeoutRef.current) clearTimeout(wsTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connectWs]);

  useEffect(() => {
    if (wsRef.current) subscribeToTask(wsRef.current, currentTaskId);
  }, [currentTaskId]);

  // Poll task status as fallback
  useEffect(() => {
    if (!currentTaskId) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    let pending = false;
    pollRef.current = setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const task = await tasksApi.get(currentTaskId);
        handleTaskUpdateRef.current(task);
      } catch {
        // ignore poll errors
      } finally {
        pending = false;
      }
    }, 1000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [currentTaskId]);

  const startTask = useCallback(
    async (
      type: "transcribe" | "subtitle",
      payload: Record<string, unknown>
    ) => {
      const store = useAppStore.getState();
      if (pendingStartRef.current !== null
        || (store.currentTaskId && store.taskStatus === "running")) return;
      const generation = ++actionGenerationRef.current;
      pendingStartRef.current = generation;
      const isCurrentStart = () => aliveRef.current
        && generation === actionGenerationRef.current
        && useAppStore.getState().currentTaskId === null;
      // Detach the old task before awaiting the new ID, including any late reads.
      updateGuardRef.current.generation++;
      setCurrentTaskId(null);
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(true);
      setError(null);
      setTaskState(0, "Starting...", "running");
      setTaskAttention(null);
      lastPartialKeyRef.current = null;
      previewRevisionRef.current = 0;

      try {
        let result: { task_id: string };
        if (type === "transcribe") {
          result = await transcribeApi.start(
            payload as { file_path: string; model?: string; language?: string }
          );
        } else {
          result = await subtitleApi.start(
            payload as Parameters<typeof subtitleApi.start>[0]
          );
        }
        if (isCurrentStart()) {
          setCurrentTaskId(result.task_id);
        } else {
          // Cancelling the pending HTTP request alone cannot stop a backend task
          // that was already created. Retire its late ID instead of attaching it.
          try {
            await tasksApi.cancel(result.task_id);
          } catch (err) {
            if (aliveRef.current) {
              setError(`无法确认旧任务已取消：${err instanceof Error ? err.message : result.task_id}`);
            }
          }
        }
      } catch (err) {
        if (!isCurrentStart()) return;
        setError(err instanceof Error ? err.message : "Failed to start task");
        setTaskState(0, "", "idle");
        setIsProcessing(false);
      } finally {
        if (pendingStartRef.current === generation) pendingStartRef.current = null;
      }
    },
    [setCurrentTaskId, setError, setTaskAttention, setTaskState, setIsProcessing]
  );

  const cancelTask = useCallback(async () => {
    const taskId = useAppStore.getState().currentTaskId;
    const generation = ++actionGenerationRef.current;
    pendingStartRef.current = null;
    const isCurrentCancellation = () => aliveRef.current
      && generation === actionGenerationRef.current
      && taskId === useAppStore.getState().currentTaskId
      && !(updateGuardRef.current.taskId === taskId && updateGuardRef.current.terminal);
    if (taskId) {
      try {
        await tasksApi.cancel(taskId);
      } catch (err) {
        if (!isCurrentCancellation()) return;
        // Completion may have won the race. Otherwise keep monitoring instead
        // of presenting an unacknowledged cancellation as a stopped worker.
        try {
          const latest = await tasksApi.get(taskId);
          if (!isCurrentCancellation()) return;
          handleTaskUpdateRef.current(latest);
          if (["completed", "failed", "cancelled"].includes(latest.status)) return;
        } catch {
          // Retain the current task so a later poll can reconnect.
        }
        if (isCurrentCancellation()) {
          setError(`取消任务失败：${err instanceof Error ? err.message : "请重试"}`);
        }
        return;
      }
    }
    if (!isCurrentCancellation()) return;
    updateGuardRef.current.generation++;
    if (pollRef.current) clearInterval(pollRef.current);
    setIsProcessing(false);
    setTaskAttention(null);
    setTaskState(0, "", "idle");
    setCurrentTaskId(null);
  }, [setIsProcessing, setTaskAttention, setTaskState, setCurrentTaskId, setError]);

  return { startTask, cancelTask };
}
