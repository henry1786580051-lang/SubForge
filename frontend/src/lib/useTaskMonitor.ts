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

  function handleTaskUpdate(task: TaskInfo) {
    // Guard against stale task updates (e.g., after cancel)
    const store = useAppStore.getState();
    if (task.id !== store.currentTaskId) return;

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
    const hasNewPreview = previewRevision > previewRevisionRef.current;
    if (task.status === "running" && task.preview_segments && hasNewPreview) {
      setSubtitles(task.preview_segments);
      previewRevisionRef.current = previewRevision;
    } else if (task.status === "running" && task.preview_delta && hasNewPreview) {
      const delta = task.preview_delta;
      const current = useAppStore.getState().subtitles;
      if (delta.mode === "replace") {
        setSubtitles(delta.segments);
      } else if (delta.mode === "append") {
        setSubtitles([...current, ...delta.segments]);
      } else {
        const changed = new Map(delta.segments.map((segment) => [segment.id, segment]));
        setSubtitles(current.map((segment) => {
          const patch = changed.get(segment.id);
          return patch ? { ...segment, ...patch } : segment;
        }));
      }
      previewRevisionRef.current = previewRevision;
    }

    // Compatibility fallback for older backends that only expose a partial file.
    if (task.status === "running" && task.subtitle_file) {
      const partialKey = `${task.subtitle_file}:${task.progress}:${task.message}`;
      if (partialKey === lastPartialKeyRef.current) return;
      lastPartialKeyRef.current = partialKey;
      subtitlesApi
        .load(task.subtitle_file)
        .then((subFile) => {
          if (task.id !== useAppStore.getState().currentTaskId) return;
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
            if (task.id !== useAppStore.getState().currentTaskId) return;
            setSubtitles(subFile.segments);
          })
          .catch((err) => {
            if (task.id !== useAppStore.getState().currentTaskId) return;
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
            if (task.id !== useAppStore.getState().currentTaskId) return;
            setSubtitles(subFile.segments);
          })
          .catch((err) => {
            if (task.id !== useAppStore.getState().currentTaskId) return;
            setError(err instanceof Error ? err.message : "Failed to load translated subtitles");
          });
      }
    }

    if (task.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);
      setTaskAttention(null);
      setError(task.error || "Task failed");
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
      wsRef.current.close();
      wsRef.current = null;
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
        if (aliveRef.current) {
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

    pollRef.current = setInterval(async () => {
      try {
        const task = await tasksApi.get(currentTaskId);
        handleTaskUpdateRef.current(task);
      } catch {
        // ignore poll errors
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
        setCurrentTaskId(result.task_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to start task");
        setTaskState(0, "", "idle");
        setIsProcessing(false);
      }
    },
    [setCurrentTaskId, setError, setTaskAttention, setTaskState, setIsProcessing]
  );

  const cancelTask = useCallback(async () => {
    const taskId = useAppStore.getState().currentTaskId;
    if (taskId) {
      try {
        await tasksApi.cancel(taskId);
      } catch {
        // ignore
      }
    }
    if (pollRef.current) clearInterval(pollRef.current);
    setIsProcessing(false);
    setTaskAttention(null);
    setTaskState(0, "", "idle");
    setCurrentTaskId(null);
  }, [setIsProcessing, setTaskAttention, setTaskState, setCurrentTaskId]);

  return { startTask, cancelTask };
}
