export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8000"
    : "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  const text = await res.text();
  if (!text) return undefined as unknown as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`Invalid JSON response from ${path}`);
  }
}

// Tasks
export const tasksApi = {
  list: () => request<TaskInfo[]>("/api/tasks/"),
  get: (id: string) => request<TaskInfo>(`/api/tasks/${id}`),
  cancel: (id: string) => request(`/api/tasks/${id}/cancel`, { method: "POST" }),
};

// Files
export const filesApi = {
  upload: async (file: File): Promise<{ file_path: string; filename: string }> => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/api/files/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Upload failed");
    return res.json();
  },
  info: (path: string) =>
    request<FileInfo>(`/api/files/info?path=${encodeURIComponent(path)}`),
  thumbnailUrl: (path: string) =>
    `${API_BASE}/api/files/thumbnail?path=${encodeURIComponent(path)}`,
  streamUrl: (path: string) =>
    `${API_BASE}/api/files/stream?path=${encodeURIComponent(path)}`,
};

// Subtitles
export const subtitlesApi = {
  load: (path: string) =>
    request<SubtitleFile>(`/api/subtitles/load?path=${encodeURIComponent(path)}`),
  exportUrl: (path: string, format: string, mode: string = "bilingual") =>
    `${API_BASE}/api/subtitles/export?path=${encodeURIComponent(path)}&format=${format}&mode=${mode}`,
  exportPost: async (segments: SubtitleSegment[], format: string, mode: string, filename: string): Promise<Blob> => {
    const res = await fetch(`${API_BASE}/api/subtitles/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments, format, mode, filename }),
    });
    if (!res.ok) throw new Error(`Export failed: ${res.status}`);
    return res.blob();
  },
  save: (file_path: string, segments: SubtitleSegment[]) =>
    request<{ status: string; file_path: string; count: number }>("/api/subtitles/save", {
      method: "POST",
      body: JSON.stringify({ file_path, segments }),
    }),
};

// Transcribe
export const transcribeApi = {
  start: (data: { file_path: string; model?: string; language?: string; audio_track?: number; device?: string; n_threads?: number; compute_type?: string }) =>
    request<{ task_id: string; status: string }>("/api/transcribe/start", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  listModels: () => request<AsrModelInfo[]>("/api/transcribe/models"),
  modelStatus: () => request<AsrModelStatus>("/api/transcribe/model-status"),
  testModel: () => request<AsrModelTestResult>("/api/transcribe/test-model", { method: "POST" }),
  hardware: () => request<{ platform: string; arch: string; chip: string; device: string; n_threads: number; compute_type: string; gpu: string }>("/api/transcribe/hardware"),
  downloadModel: (model_id: string) =>
    request<{ task_id?: string; status: string; path?: string }>("/api/transcribe/download-model", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),
  deleteModel: (model_id: string) =>
    request<{ status: string; model_id: string; path?: string }>("/api/transcribe/delete-model", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),
};

// Subtitle processing
export const subtitleApi = {
  start: (data: {
    subtitle_file: string;
    target_language?: string;
    translator?: string;
    llm_model?: string;
    need_optimize?: boolean;
    need_translate?: boolean;
    need_reflect?: boolean;
    custom_prompt?: string;
  }) =>
    request<{ task_id: string; status: string }>("/api/subtitle/start", {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

// Config
export const configApi = {
  get: () => request<Record<string, unknown>>("/api/config/"),
  update: (key: string, value: unknown) =>
    request("/api/config/", {
      method: "POST",
      body: JSON.stringify({ key, value }),
    }),
  switchLlmProvider: (data: {
    provider: string;
    current_base_url: string;
    current_api_key: string;
    current_model: string;
  }) =>
    request<{
      status: string;
      provider: string;
      base_url: string;
      api_key_configured: boolean;
      model: string;
    }>("/api/config/llm-provider", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  fetchModels: () => request<{ models: string[]; error?: string }>("/api/config/models"),
  testLlm: () => request<{ ok: boolean; model?: string; error?: string }>("/api/config/test-llm"),
  testWhisper: () => request<{ ok: boolean; error?: string }>("/api/config/test-whisper"),
  fetchWhisperModels: () => request<{ models: string[]; error?: string }>("/api/config/whisper-models"),
};

// LLM Logs
export const llmLogsApi = {
  list: (page = 1, search = "") =>
    request<{ groups: LlmLogGroup[]; total: number; page: number; pages: number }>(
      `/api/llm-logs/?page=${page}&search=${encodeURIComponent(search)}`
    ),
  detail: (id: string) =>
    request<LlmLogGroup>(`/api/llm-logs/${encodeURIComponent(id)}`),
  clear: () => request("/api/llm-logs/", { method: "DELETE" }),
};

// Health
export const healthApi = {
  check: () => request<{ status: string; version: string; ffmpeg: boolean; ffprobe: boolean }>("/api/health"),
};

// Types
export interface TaskInfo {
  id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
  subtitle_file?: string | null;
  preview_segments?: SubtitleSegment[] | null;
  preview_revision?: number;
  preview_delta?: {
    mode: "append" | "patch" | "replace";
    segments: SubtitleSegment[];
    total: number;
  } | null;
}

export interface FileInfo {
  file_path: string;
  filename: string;
  duration: number;
  size: number;
  bit_rate: number;
  video: {
    width: number;
    height: number;
    codec: string;
    fps: string;
  } | null;
  audio_tracks: {
    index: number;
    codec: string;
    channels: number;
    sample_rate: number;
    language: string;
  }[];
}

export interface SubtitleSegment {
  id: number;
  start: string;
  end: string;
  text: string;
  translated: string;
  speaker?: string;
}

export interface SubtitleFile {
  file_path: string;
  format: string;
  segments: SubtitleSegment[];
  count: number;
}

export interface AsrModelInfo {
  id: string;
  name: string;
  category: "whisper_cpp" | "whisperx" | string;
  type: "ggml" | "mlx" | "alignment" | string;
  size: string;
  downloaded: boolean;
  downloadable?: boolean;
  deletable?: boolean;
  path: string;
  align_model?: string;
  language?: string;
  value?: string;
  selected?: boolean;
  state?: "ready" | "missing" | "on_demand" | string;
  detail?: string;
  resolved_model?: string;
}

export interface AsrModelStatus {
  engine: string;
  engine_name: string;
  model_id: string;
  model_value: string;
  model_name: string;
  resolved_model: string;
  model_path: string;
  model_ready: boolean;
  model_state: "ready" | "missing" | "on_demand" | string;
  model_message: string;
  alignment_model: string;
  alignment_path: string;
  alignment_ready: boolean;
  platform_supported: boolean;
  runtime_ready: boolean;
  testable: boolean;
}

export interface AsrModelTestResult extends AsrModelStatus {
  ok: boolean;
  error?: string;
  elapsed_seconds?: number;
  transcript?: string;
  segment_count?: number;
}

export interface LlmLogEntry {
  timestamp?: string;
  task_id?: string;
  file_name?: string;
  stage?: string;
  model?: string;
  duration_ms?: number;
  tokens?: number;
  request?: unknown;
  response?: unknown;
  status?: number;
  error?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  batch?: string;
}

export interface LlmLogGroup {
  id: string;
  task_id?: string;
  file_name?: string;
  started_at?: string;
  ended_at?: string;
  stages: string[];
  models: string[];
  request_count: number;
  error_count: number;
  duration_ms: number;
  tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
  entries: LlmLogEntry[];
}
