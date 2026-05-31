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

export function useTaskMonitor() {
  const {
    currentTaskId,
    setCurrentTaskId,
    setTaskState,
    setSubtitleFile,
    setSubtitles,
    setStep,
    setError,
    setIsProcessing,
  } = useAppStore();

  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const aliveRef = useRef(true);
  const handleTaskUpdateRef = useRef<(task: TaskInfo) => void>(() => {});

  // Track last loaded partial file path + segment count to detect content changes
  const lastPartialRef = useRef<string | null>(null);
  const lastPartialCountRef = useRef<number>(0);

  function handleTaskUpdate(task: TaskInfo) {
    // Guard against stale task updates (e.g., after cancel)
    const store = useAppStore.getState();
    if (task.id !== store.currentTaskId) return;

    setTaskState(
      task.progress,
      task.message,
      task.status as "idle" | "running" | "completed" | "failed"
    );

    // Load partial subtitle results during processing (real-time preview)
    // Same file is updated in-place, so reload when segment count changes
    if (task.status === "running" && task.subtitle_file) {
      subtitlesApi
        .load(task.subtitle_file)
        .then((subFile) => {
          if (subFile.count !== lastPartialCountRef.current) {
            lastPartialCountRef.current = subFile.count;
            lastPartialRef.current = task.subtitle_file ?? null;
            setSubtitles(subFile.segments);
            if (!store.subtitleFile) setStep("subtitle");
          }
        })
        .catch(() => {});
    }

    if (task.status === "completed") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);

      const result = task.result;
      if (task.type === "transcribe" && result?.subtitle_file) {
        setSubtitleFile(result.subtitle_file as string);
        subtitlesApi
          .load(result.subtitle_file as string)
          .then((subFile) => {
            setSubtitles(subFile.segments);
            setStep("subtitle");
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : "Failed to load subtitles");
          });
      }
      if (task.type === "subtitle" && result?.subtitle_file) {
        setSubtitleFile(result.subtitle_file as string);
        subtitlesApi
          .load(result.subtitle_file as string)
          .then((subFile) => {
            setSubtitles(subFile.segments);
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : "Failed to load translated subtitles");
          });
      }
    }

    if (task.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      setIsProcessing(false);
      setError(task.error || "Task failed");
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
      ws.onopen = () => {};
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
          wsTimeoutRef.current = setTimeout(connectWs, 3000);
        }
      };
      wsRef.current = ws;
    } catch {
      // WebSocket not available
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    connectWs();
    return () => {
      aliveRef.current = false;
      if (wsTimeoutRef.current) clearTimeout(wsTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connectWs]);

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
      lastPartialRef.current = null;
      lastPartialCountRef.current = 0;

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
    [setCurrentTaskId, setError, setTaskState, setIsProcessing]
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
    setTaskState(0, "", "idle");
    setCurrentTaskId(null);
  }, [setIsProcessing, setTaskState, setCurrentTaskId]);

  return { startTask, cancelTask };
}
