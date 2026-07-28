import { create } from "zustand";
import type { FileInfo, SubtitleSegment, TaskAttention } from "@/lib/api";

export type WorkflowStep = "import" | "transcribe" | "subtitle";

export interface Toast {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}
export type ActiveView = "workflow" | "settings" | "llm-logs";

interface AppState {
  // Current workflow step
  step: WorkflowStep;
  setStep: (step: WorkflowStep) => void;

  // Active view
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;

  // Video file
  videoFile: string | null;
  setVideoFile: (path: string | null) => void;
  fileInfo: FileInfo | null;
  setFileInfo: (info: FileInfo | null) => void;

  // Subtitle data
  subtitleFile: string | null;
  setSubtitleFile: (path: string | null) => void;
  subtitles: SubtitleSegment[];
  setSubtitles: (segments: SubtitleSegment[]) => void;
  updateSubtitle: (id: number, field: "text" | "translated", value: string) => void;
  selectedIds: Set<number>;
  toggleSelect: (id: number) => void;
  selectAll: () => void;
  deselectAll: () => void;

  // Current task
  currentTaskId: string | null;
  setCurrentTaskId: (id: string | null) => void;
  taskProgress: number;
  taskMessage: string;
  taskStatus: "idle" | "running" | "completed" | "failed";
  setTaskState: (progress: number, message: string, status: AppState["taskStatus"]) => void;
  taskAttention: TaskAttention | null;
  setTaskAttention: (attention: TaskAttention | null) => void;

  // Processing state
  isProcessing: boolean;
  setIsProcessing: (v: boolean) => void;

  // Config
  config: {
    transcribeModel: string;
    sourceLanguage: string;
    targetLanguage: string;
    translator: string;
    llmModel: string;
    needOptimize: boolean;
    needTranslate: boolean;
    needReflect: boolean;
    customPrompt: string;
    whisperModelSize: string;
    whisperxAlignmentStrategy: "auto" | "manual";
    whisperxAlignModel: string;
    whisperxBatchSize: number;
    whisperxSupported: boolean;
    enableAudioEnhancement: boolean;
    speakerDiarization: "off" | "two" | "auto" | "fixed";
    speakerCount: number;
  };
  setConfig: (config: Partial<AppState["config"]>) => void;

  // Backend status
  backendOnline: boolean;
  setBackendOnline: (v: boolean) => void;

  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // Error
  error: string | null;
  setError: (msg: string | null) => void;

  // Toast
  toasts: Toast[];
  _toastId: number;
  addToast: (msg: string, type?: "success" | "error" | "info") => void;
  removeToast: (id: number) => void;

  // FFmpeg status
  ffmpegOk: boolean;
  setFfmpegOk: (v: boolean) => void;

  // Video playback
  currentTime: number;
  setCurrentTime: (t: number) => void;
  isPlaying: boolean;
  setIsPlaying: (v: boolean) => void;
  seekToTime: number | null;
  setSeekToTime: (t: number | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  step: "import",
  setStep: (step) => set({ step }),

  activeView: "workflow",
  setActiveView: (view) => set({ activeView: view }),

  videoFile: null,
  setVideoFile: (path) => set({ videoFile: path }),
  fileInfo: null,
  setFileInfo: (info) => set({ fileInfo: info }),

  subtitleFile: null,
  setSubtitleFile: (path) => set({ subtitleFile: path }),
  subtitles: [],
  setSubtitles: (segments) => set({ subtitles: segments }),
  updateSubtitle: (id, field, value) =>
    set((state) => ({
      subtitles: state.subtitles.map((s) =>
        s.id === id ? { ...s, [field]: value } : s
      ),
    })),
  selectedIds: new Set(),
  toggleSelect: (id) =>
    set((state) => {
      const next = new Set(state.selectedIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { selectedIds: next };
    }),
  selectAll: () =>
    set((state) => ({
      selectedIds: new Set(state.subtitles.map((s) => s.id)),
    })),
  deselectAll: () => set({ selectedIds: new Set() }),

  currentTaskId: null,
  setCurrentTaskId: (id) => set({ currentTaskId: id }),
  taskProgress: 0,
  taskMessage: "",
  taskStatus: "idle",
  setTaskState: (progress, message, status) =>
    set({ taskProgress: progress, taskMessage: message, taskStatus: status }),
  taskAttention: null,
  setTaskAttention: (attention) => set({ taskAttention: attention }),

  isProcessing: false,
  setIsProcessing: (v) => set({ isProcessing: v }),

  config: {
    transcribeModel: "whisperx",
    sourceLanguage: "auto",
    targetLanguage: "chinese",
    translator: "bing",
    llmModel: "gpt-4o-mini",
    needOptimize: true,
    needTranslate: true,
    needReflect: false,
    customPrompt: "",
    whisperModelSize: "large-v3",
    whisperxAlignmentStrategy: "auto",
    whisperxAlignModel: "WAV2VEC2_ASR_LARGE_LV60K_960H",
    whisperxBatchSize: 8,
    whisperxSupported: true,
    enableAudioEnhancement: true,
    speakerDiarization: "off",
    speakerCount: 2,
  },
  setConfig: (partial) =>
    set((state) => ({ config: { ...state.config, ...partial } })),

  backendOnline: false,
  setBackendOnline: (v) => set({ backendOnline: v }),

  sidebarCollapsed: false,
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

  error: null,
  setError: (msg) => set({ error: msg }),

  toasts: [],
  _toastId: 0,
  addToast: (message, type = "info") =>
    set((state) => {
      const id = state._toastId + 1;
      setTimeout(() => {
        useAppStore.getState().removeToast(id);
      }, 4000);
      return { toasts: [...state.toasts, { id, message, type }], _toastId: id };
    }),
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  ffmpegOk: false,
  setFfmpegOk: (v) => set({ ffmpegOk: v }),

  currentTime: 0,
  setCurrentTime: (t) => set({ currentTime: t }),
  isPlaying: false,
  setIsPlaying: (v) => set({ isPlaying: v }),
  seekToTime: null,
  setSeekToTime: (t) => set({ seekToTime: t }),
}));
