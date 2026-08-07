"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import { useAppStore, type WorkflowStep } from "@/store/appStore";
import {
  configApi,
  filesApi,
  openNativeFile,
  subtitlesApi,
  tasksApi,
  transcribeApi,
  type AsrModelInfo,
  type FileInfo,
  type SubtitleSegment,
} from "@/lib/api";
import { formatDuration, formatSize } from "@/lib/format";
import { SubtitlePanel } from "@/components/SubtitlePanel";
import {
  ASR_ENGINES,
  SOURCE_LANGUAGES,
  STEP_META,
  TARGET_LANGUAGES,
  TRANSCRIBE_STAGES,
  TRANSLATORS,
} from "@/features/workflow/catalog";
import {
  analyzeSubtitleQuality,
  type SubtitleQuality,
} from "@/features/workflow/quality";

type TaskStarter = (
  type: "transcribe" | "subtitle",
  payload: Record<string, unknown>
) => Promise<void>;
type AppConfig = ReturnType<typeof useAppStore.getState>["config"];

interface WorkflowWorkspaceProps {
  startTask: TaskStarter;
  cancelTask: () => Promise<void>;
}

export function WorkflowWorkspace({ startTask, cancelTask }: WorkflowWorkspaceProps) {
  const { step } = useAppStore();

  return (
    <main className="flex-1 min-h-0 bg-background">
      {step === "import" && <ImportWorkspace />}
      {step === "transcribe" && (
        <TranscribeWorkspace startTask={startTask} cancelTask={cancelTask} />
      )}
      {step === "subtitle" && (
        <SubtitleWorkspace startTask={startTask} cancelTask={cancelTask} />
      )}
    </main>
  );
}

function ImportWorkspace() {
  const {
    backendOnline,
    ffmpegOk,
    fileInfo,
    setFileInfo,
    setSubtitles,
    setSubtitleFile,
    setVideoFile,
    subtitleFile,
    subtitles,
    videoFile,
  } = useAppStore();
  const [dragType, setDragType] = useState<"media" | "subtitle" | null>(null);
  const [uploading, setUploading] = useState<"media" | "subtitle" | null>(null);
  const mediaInputRef = useRef<HTMLInputElement>(null);
  const subtitleInputRef = useRef<HTMLInputElement>(null);

  const loadMedia = useCallback(
    async (file: File) => {
      setUploading("media");
      try {
        if (
          file.size > 1024 * 1024 * 1024 &&
          typeof window !== "undefined" &&
          "pywebview" in window
        ) {
          throw new Error("大于 1GB 的素材请使用“选择素材”按钮导入，避免在应用内复制整份文件");
        }
        const uploaded = await filesApi.upload(file);
        setVideoFile(uploaded.file_path);
        const info = await filesApi.info(uploaded.file_path);
        setFileInfo(info);
        useAppStore.getState().addToast("素材已导入", "success");
      } catch (err) {
        useAppStore
          .getState()
          .setError(err instanceof Error ? err.message : "素材导入失败");
      } finally {
        setUploading(null);
      }
    },
    [setFileInfo, setVideoFile]
  );

  const loadSubtitle = useCallback(
    async (file: File) => {
      setUploading("subtitle");
      try {
        const uploaded = await filesApi.upload(file);
        const loaded = await subtitlesApi.load(uploaded.file_path);
        setSubtitleFile(loaded.file_path);
        setSubtitles(loaded.segments);
        useAppStore.getState().addToast("字幕已导入", "success");
      } catch (err) {
        useAppStore
          .getState()
          .setError(err instanceof Error ? err.message : "字幕导入失败");
      } finally {
        setUploading(null);
      }
    },
    [setSubtitleFile, setSubtitles]
  );

  const chooseMedia = useCallback(async () => {
    try {
      const selected = await openNativeFile("media");
      if (!selected.available) {
        mediaInputRef.current?.click();
        return;
      }
      if (!selected.path) return;
      setUploading("media");
      setVideoFile(selected.path);
      const info = await filesApi.info(selected.path);
      setFileInfo(info);
      useAppStore.getState().addToast("素材已导入", "success");
    } catch (err) {
      useAppStore
        .getState()
        .setError(err instanceof Error ? err.message : "素材导入失败");
    } finally {
      setUploading(null);
    }
  }, [setFileInfo, setVideoFile]);

  const chooseSubtitle = useCallback(async () => {
    try {
      const selected = await openNativeFile("subtitle");
      if (!selected.available) {
        subtitleInputRef.current?.click();
        return;
      }
      if (!selected.path) return;
      setUploading("subtitle");
      const loaded = await subtitlesApi.load(selected.path);
      setSubtitleFile(loaded.file_path);
      setSubtitles(loaded.segments);
      useAppStore.getState().addToast("字幕已导入", "success");
    } catch (err) {
      useAppStore
        .getState()
        .setError(err instanceof Error ? err.message : "字幕导入失败");
    } finally {
      setUploading(null);
    }
  }, [setSubtitleFile, setSubtitles]);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      setDragType(null);
      if (!file) return;
      if (/\.(srt|vtt|ass)$/i.test(file.name)) {
        void loadSubtitle(file);
      } else {
        void loadMedia(file);
      }
    },
    [loadMedia, loadSubtitle]
  );

  const checks = [
    {
      label: "后端服务",
      value: backendOnline ? "在线" : "离线",
      ok: backendOnline,
      icon: "solar:server-square-bold-duotone",
    },
    {
      label: "FFmpeg",
      value: ffmpegOk ? "可用" : "未就绪",
      ok: ffmpegOk,
      icon: "solar:videocamera-record-bold-duotone",
    },
    {
      label: "音轨",
      value: fileInfo ? `${fileInfo.audio_tracks.length} 条` : "待检测",
      ok: !!fileInfo && fileInfo.audio_tracks.length > 0,
      icon: "solar:soundwave-bold-duotone",
    },
    {
      label: "字幕",
      value: subtitles.length > 0 ? `${subtitles.length} 条` : "可选",
      ok: subtitles.length > 0,
      neutral: subtitles.length === 0,
      icon: "solar:document-text-bold-duotone",
    },
  ];

  return (
    <WorkspaceFrame meta={STEP_META.import}>
      <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_360px] gap-5 max-xl:grid-cols-1">
        <section
          onDragOver={(event) => {
            event.preventDefault();
            setDragType("media");
          }}
          onDragLeave={() => setDragType(null)}
          onDrop={handleDrop}
          className={`relative min-h-0 overflow-hidden rounded-2xl border bg-surface shadow-sm transition-all ${
            dragType
              ? "border-accent bg-accent-dim"
              : "border-border hover:border-border-active"
          }`}
        >
          <input
            ref={mediaInputRef}
            type="file"
            accept="video/*,audio/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void loadMedia(file);
            }}
          />
          <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-accent via-emerald-500 to-sky-500" />
          <div className="flex h-full min-h-[330px] flex-col p-5">
            <div className="flex items-start justify-between gap-6">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
                  准备素材
                </p>
                <h2 className="mt-2 max-w-[620px] text-[27px] font-semibold leading-tight text-text-primary">
                  选择要处理的视频或音频
                </h2>
              </div>
              <button
                onClick={() => void chooseMedia()}
                className="inline-flex shrink-0 items-center gap-2 rounded-full bg-accent px-4 py-2 text-[13px] font-medium text-white shadow-md transition hover:bg-accent-hover disabled:opacity-50"
                disabled={uploading === "media"}
              >
                <Icon icon={uploading === "media" ? "solar:refresh-bold" : "solar:upload-bold"} className={uploading === "media" ? "animate-spin" : ""} width={17} />
                选择文件
              </button>
            </div>

            <div className="mt-5 grid flex-1 grid-cols-[minmax(280px,0.78fr)_minmax(320px,1fr)] gap-5 max-lg:grid-cols-1">
              <div className="flex min-h-[220px] flex-col justify-between rounded-2xl border border-dashed border-border bg-background/70 p-4">
                <div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface text-accent shadow-sm">
                    <Icon icon="solar:folder-with-files-bold-duotone" width={27} />
                  </div>
                  <h3 className="mt-4 text-[18px] font-semibold text-text-primary">
                    {videoFile ? "当前素材" : "拖入视频或音频"}
                  </h3>
                  <p className="mt-2 text-[13px] leading-6 text-text-secondary">
                    {videoFile
                      ? videoFile.split("/").pop()
                      : "支持 MP4、MOV、MKV、MP3 和 WAV。导入后将自动读取时长、画面和音轨信息。"}
                  </p>
                </div>
                <div className="mt-5 grid grid-cols-2 gap-2.5">
                  <MetricTile label="时长" value={fileInfo ? formatDuration(fileInfo.duration) : "--"} />
                  <MetricTile label="大小" value={fileInfo ? formatSize(fileInfo.size) : "--"} />
                  <MetricTile
                    label="视频"
                    value={fileInfo?.video ? `${fileInfo.video.width}x${fileInfo.video.height}` : "音频/未知"}
                  />
                  <MetricTile label="音轨" value={fileInfo ? `${fileInfo.audio_tracks.length}` : "--"} />
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-surface-raised p-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-[14px] font-semibold text-text-primary">文件检查</h3>
                  <span className="rounded-full border border-border bg-background px-2.5 py-1 text-[11px] text-text-muted">
                    {fileInfo ? "文件已就绪" : "等待导入"}
                  </span>
                </div>
                <div className="mt-4 space-y-2.5">
                  {checks.map((check) => (
                    <CheckRow key={check.label} {...check} />
                  ))}
                </div>
                <div className="mt-4 rounded-xl bg-background p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[13px] font-medium text-text-primary">已有字幕</p>
                      <p className="mt-1 text-[12px] text-text-muted">
                        {subtitleFile ? subtitleFile.split("/").pop() : "导入 SRT、VTT 或 ASS，可直接断句、翻译和审校"}
                      </p>
                    </div>
                    <button
                      onClick={() => void chooseSubtitle()}
                      className="rounded-full border border-border bg-surface px-3 py-1.5 text-[12px] text-text-secondary transition hover:border-border-active hover:text-text-primary"
                    >
                      导入字幕
                    </button>
                  </div>
                  <input
                    ref={subtitleInputRef}
                    type="file"
                    accept=".srt,.vtt,.ass"
                    className="hidden"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void loadSubtitle(file);
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </section>

        <aside className="flex min-h-0 flex-col gap-5 overflow-auto pr-1">
          <Panel title="处理流程" icon="solar:map-point-wave-bold-duotone">
            <div className="space-y-4">
              <FlowStep index="1" title="导入" active done={!!videoFile || !!subtitleFile} />
              <FlowStep index="2" title="转录" active={!!videoFile} done={!!subtitleFile && subtitles.length > 0} />
              <FlowStep index="3" title="断句与翻译" active={!!subtitleFile} done={subtitles.some((s) => s.translated)} />
              <FlowStep index="4" title="导出" active={subtitles.length > 0} />
            </div>
          </Panel>

          <Panel title="音轨信息" icon="solar:soundwave-square-bold-duotone">
            {fileInfo?.audio_tracks.length ? (
              <div className="space-y-2">
                {fileInfo.audio_tracks.map((track) => (
                  <div key={track.index} className="rounded-xl border border-border bg-background p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[13px] font-medium text-text-primary">音轨 {track.index + 1}</span>
                      <span className="rounded-full bg-surface px-2 py-0.5 text-[10px] text-text-muted">
                        {track.language || "未标注"}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-text-muted">
                      <span>{track.codec}</span>
                      <span>{track.channels} ch</span>
                      <span>{track.sample_rate} Hz</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState icon="solar:soundwave-bold-duotone" title="导入文件后显示音轨信息" minHeight="min-h-[220px]" />
            )}
          </Panel>
        </aside>
      </div>
    </WorkspaceFrame>
  );
}

function TranscribeWorkspace({ startTask, cancelTask }: WorkflowWorkspaceProps) {
  const {
    config,
    configLoaded,
    fileInfo,
    isProcessing,
    setConfig,
    setError,
    setIsProcessing,
    setStep,
    subtitles,
    taskMessage,
    taskAttention,
    taskProgress,
    taskStatus,
    videoFile,
    currentTaskId,
    addToast,
  } = useAppStore();
  const [hardware, setHardware] = useState<{
    chip: string;
    device: string;
    n_threads: number;
    compute_type: string;
    gpu: string;
  } | null>(null);
  const [models, setModels] = useState<AsrModelInfo[]>([]);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [huggingfaceToken, setHuggingfaceToken] = useState("");
  const [huggingfaceTokenConfigured, setHuggingfaceTokenConfigured] = useState(false);
  const [resolvingAlignment, setResolvingAlignment] = useState(false);

  useEffect(() => {
    void transcribeApi.hardware().then(setHardware).catch(() => {});
    void transcribeApi.listModels().then(setModels).catch(() => {});
    void configApi
      .get()
      .then((data) => {
        setHuggingfaceToken("");
        setHuggingfaceTokenConfigured(Boolean(data.huggingface_token_configured));
      })
      .catch(() => {});
  }, []);

  const saveConfig = useCallback(
    async (key: string, value: string | number | boolean) => {
      const map: Record<string, keyof AppConfig> = {
        transcribe_model: "transcribeModel",
        source_language: "sourceLanguage",
        whisper_model_size: "whisperModelSize",
        whisperx_alignment_strategy: "whisperxAlignmentStrategy",
        whisperx_align_model: "whisperxAlignModel",
        whisperx_batch_size: "whisperxBatchSize",
        enable_audio_enhancement: "enableAudioEnhancement",
        speaker_diarization: "speakerDiarization",
        speaker_count: "speakerCount",
      };
      const mapped = map[key];
      if (mapped) setConfig({ [mapped]: value });
      try {
        await configApi.update(key, value);
        return true;
      } catch (err) {
        setError(err instanceof Error ? err.message : "配置保存失败");
        return false;
      }
    },
    [setConfig, setError]
  );

  const startTranscribe = useCallback(async () => {
    if (!configLoaded || !config.transcribeModel) {
      setError("转录配置仍在加载，请稍后重试");
      return;
    }
    if (!videoFile) {
      setError("请先导入视频或音频文件");
      setStep("import");
      return;
    }
    setIsProcessing(true);
    await startTask("transcribe", {
      file_path: videoFile,
      model: config.transcribeModel,
      language: config.sourceLanguage,
    });
  }, [config.sourceLanguage, config.transcribeModel, configLoaded, setError, setIsProcessing, setStep, startTask, videoFile]);

  const currentModels = useMemo(() => {
    if (config.transcribeModel === "whisper_cpp") {
      return models.filter((model) => model.category === "whisper_cpp");
    }
    if (config.transcribeModel === "whisperx") {
      return models.filter(
        (model) => model.category === "whisperx" && ["mlx", "ctranslate2"].includes(model.type)
      );
    }
    return [];
  }, [config.transcribeModel, models]);

  const alignmentModels = useMemo(
    () => models.filter((model) => model.category === "whisperx" && model.type === "alignment"),
    [models]
  );
  const speakerModels = useMemo(
    () => models.filter(
      (model) => model.category === "whisperx" && ["diarization", "speaker_verification"].includes(model.type)
    ),
    [models]
  );
  const diarizationModels = useMemo(
    () => speakerModels.filter((model) => model.type === "diarization"),
    [speakerModels]
  );
  const diarizationReady =
    config.speakerDiarization === "off" ||
    diarizationModels.some((model) => model.downloaded);
  const selectedModel = currentModels.find(
    (model) => (model.value || model.id) === config.whisperModelSize || model.selected
  );
  const alignmentLanguage = config.sourceLanguage === "nb" ? "no" : config.sourceLanguage;
  const selectedAlignModel = alignmentModels.find((model) =>
    config.whisperxAlignmentStrategy === "manual"
      ? model.align_model === config.whisperxAlignModel || model.id === config.whisperxAlignModel
      : alignmentLanguage !== "auto" && model.language === alignmentLanguage
  );
  const quality = useMemo(() => analyzeSubtitleQuality(subtitles), [subtitles]);

  const downloadModel = useCallback(async (modelId: string): Promise<boolean> => {
    setDownloadingModel(modelId);
    setDownloadProgress((prev) => ({ ...prev, [modelId]: 0 }));
    try {
      const requestedModel = models.find((model) => model.id === modelId);
      if (requestedModel?.type === "diarization" && huggingfaceToken.trim()) {
        await configApi.update("huggingface_token", huggingfaceToken.trim());
        setHuggingfaceTokenConfigured(true);
        setHuggingfaceToken("");
      }
      const result = await transcribeApi.downloadModel(modelId);
      if (result.status === "already_exists") {
        setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloaded: true } : m)));
        return true;
      }
      const downloadTaskId = result.task_id;
      if (downloadTaskId) {
        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          const task = await tasksApi.get(downloadTaskId);
          setDownloadProgress((prev) => ({ ...prev, [modelId]: task.progress }));
          if (task.status === "completed") {
            setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloaded: true } : m)));
            return true;
          }
          if (task.status === "failed" || task.status === "cancelled") {
            throw new Error(task.error || "模型下载失败");
          }
        }
      }
      return result.status === "completed";
    } catch (err) {
      setError(err instanceof Error ? err.message : "模型下载失败");
      return false;
    } finally {
      setDownloadingModel(null);
    }
  }, [huggingfaceToken, models, setError]);

  const resolveAlignmentDecision = useCallback(async (action: "continue" | "ignore") => {
    if (!currentTaskId || resolvingAlignment) return;
    setResolvingAlignment(true);
    try {
      await transcribeApi.resolveAlignmentDecision(currentTaskId, action);
      addToast(
        action === "continue"
          ? "已保留该语种转录，并使用句段级时间轴继续"
          : "已忽略缺少对齐模型的语种",
        "info"
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法继续转录任务");
    } finally {
      setResolvingAlignment(false);
    }
  }, [addToast, currentTaskId, resolvingAlignment, setError]);

  const downloadMissingAlignmentModels = useCallback(async () => {
    if (
      !currentTaskId ||
      resolvingAlignment ||
      taskAttention?.type !== "missing_alignment_models" ||
      taskAttention.source_mode !== "auto"
    ) return;
    setResolvingAlignment(true);
    try {
      for (const model of taskAttention.models) {
        const downloaded = await downloadModel(model.model_id);
        if (!downloaded) return;
      }
      await transcribeApi.resolveAlignmentDecision(currentTaskId, "retry");
      const refreshed = await transcribeApi.listModels();
      setModels(refreshed);
      addToast("对齐模型已就绪，正在继续生成词级时间轴", "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "模型下载完成后无法恢复任务");
    } finally {
      setResolvingAlignment(false);
    }
  }, [addToast, currentTaskId, downloadModel, resolvingAlignment, setError, taskAttention]);

  const missingAlignmentModels =
    taskAttention?.type === "missing_alignment_models" && taskAttention.source_mode === "auto"
      ? taskAttention.models
      : [];
  const canIgnoreMissingLanguages =
    missingAlignmentModels.length > 0 &&
    missingAlignmentModels.every((model) => model.ranges.length > 0);

  return (
    <WorkspaceFrame meta={STEP_META.transcribe}>
      <div className="grid h-full min-h-0 grid-cols-[minmax(480px,1fr)_minmax(360px,0.72fr)] gap-5 max-xl:grid-cols-1">
        <section className="grid min-h-0 grid-rows-[minmax(420px,1fr)_auto] gap-5">
          <Panel
            title={subtitles.length ? `转录结果 · ${subtitles.length} 条` : "实时转录结果"}
            icon="solar:playlist-bold-duotone"
            fill
          >
            <LiveSubtitleList subtitles={subtitles} isLive={taskStatus === "running"} />
          </Panel>
          <Panel title="时间轴质量" icon="solar:shield-warning-bold-duotone">
            <QualitySummary quality={quality} compact />
          </Panel>
        </section>

        <aside className="flex min-h-0 flex-col gap-5 overflow-auto pr-1">
          <Panel title="识别引擎" icon="solar:tuning-square-2-bold-duotone">
            <div className="grid grid-cols-2 gap-2.5">
              {ASR_ENGINES.map((engine) => {
                const unsupported = engine.id === "whisperx" && !config.whisperxSupported;
                return (
                <button
                  key={engine.id}
                  disabled={unsupported}
                  onClick={() => void saveConfig("transcribe_model", engine.id)}
                  className={`flex min-h-[100px] flex-col rounded-lg border p-3.5 text-left transition-[border-color,background-color,transform] duration-200 active:translate-y-px ${
                    config.transcribeModel === engine.id
                      ? "border-accent bg-accent-dim text-accent"
                      : unsupported
                      ? "cursor-not-allowed border-border bg-background text-text-muted opacity-45"
                      : "border-border bg-background text-text-secondary hover:border-border-active"
                  }`}
                >
                  <div className="flex min-h-5 items-center gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                      <Icon icon={engine.icon} width={18} />
                    </span>
                    <span className="text-[13px] font-semibold leading-5">{engine.name}</span>
                  </div>
                  <p className="mt-2 min-h-8 text-[10px] leading-4 text-text-muted">{unsupported ? "当前平台不支持" : engine.desc}</p>
                </button>
              )})}
            </div>
          </Panel>

          <Panel title="说话人识别" icon="solar:users-group-rounded-bold-duotone">
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[12px] font-semibold text-text-primary">说话人识别</p>
                  <p className="mt-1 text-[10px] leading-4 text-text-muted">
                    区分双人或多人对话，并使用原始音轨保护较弱说话人的语音覆盖率。
                  </p>
                </div>
                <span className={`shrink-0 rounded-md px-2 py-1 text-[9px] font-semibold ${
                  config.speakerDiarization === "off"
                    ? "bg-background text-text-muted"
                    : diarizationReady
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-amber-50 text-amber-700"
                }`}>
                  {config.speakerDiarization === "off"
                    ? "未启用"
                    : diarizationReady
                    ? "已就绪"
                    : "需要模型"}
                </span>
              </div>

              <div className="grid grid-cols-4 gap-2" role="group" aria-label="多人语音识别模式">
                {([["off", "关闭"], ["two", "双人"], ["auto", "自动"], ["fixed", "指定人数"]] as const).map(([value, label]) => (
                  <button
                    key={value}
                    onClick={() => void saveConfig("speaker_diarization", value)}
                    className={`h-9 rounded-md border text-[11px] font-medium transition-colors ${
                      config.speakerDiarization === value
                        ? "border-accent bg-accent-dim text-accent"
                        : "border-border bg-background text-text-secondary hover:border-border-active"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {config.speakerDiarization === "fixed" && (
                <label className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2">
                  <span>
                    <span className="block text-[11px] font-semibold text-text-primary">说话人数</span>
                    <span className="mt-0.5 block text-[9px] text-text-muted">已知人数时可避免自动模式过度聚类</span>
                  </span>
                  <input
                    type="number"
                    min={2}
                    max={10}
                    step={1}
                    value={config.speakerCount}
                    onChange={(event) => {
                      const count = Math.max(2, Math.min(10, Number(event.target.value) || 2));
                      void saveConfig("speaker_count", count);
                    }}
                    className="input-field h-9 w-20 text-center"
                    aria-label="说话人数"
                  />
                </label>
              )}

              {config.speakerDiarization !== "off" && (
                <div className="space-y-3 border-t border-border pt-3">
                  {speakerModels.map((model) => (
                    <ModelRow
                      key={model.id}
                      model={model}
                      active={Boolean(model.downloaded)}
                      downloading={downloadingModel === model.id}
                      progress={downloadProgress[model.id]}
                      onSelect={() => undefined}
                      onDownload={() => void downloadModel(model.id)}
                    />
                  ))}
                  {!speakerModels.length && (
                    <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[10px] leading-4 text-amber-800">
                      未找到 Community-1 模型配置，请检查后端模型清单。
                    </p>
                  )}
                  <label className="block space-y-1.5">
                    <span className="text-[11px] font-medium text-text-muted">Hugging Face Token</span>
                    <input
                      type="password"
                      value={huggingfaceToken}
                      onChange={(event) => setHuggingfaceToken(event.target.value)}
                      onBlur={(event) => {
                        const token = event.target.value.trim();
                        if (token) {
                          void saveConfig("huggingface_token", token).then((saved) => {
                            if (saved) {
                              setHuggingfaceTokenConfigured(true);
                              setHuggingfaceToken("");
                            }
                          });
                        }
                      }}
                      placeholder={huggingfaceTokenConfigured ? "已保存，需要更换时重新填写" : "首次下载 Community-1 时填写 hf_..."}
                      autoComplete="off"
                      className="input-field"
                    />
                  </label>
                  <p className="text-[10px] leading-4 text-text-muted">
                    多人模式固定使用原始音轨，不执行 DeepFilterNet 候选比较或降噪。
                  </p>
                </div>
              )}
            </div>
          </Panel>

          <Panel title="识别模型" icon="solar:layers-bold-duotone">
            {config.transcribeModel === "whisper_api" ? (
              <EmptyState icon="solar:cloud-bold-duotone" title="云端模型在设置页配置" />
            ) : (
              <div className="space-y-4">
                <div>
                  <FieldLabel
                    label="当前转录模型"
                    value={selectedModel?.downloaded ? "本地可用" : selectedModel?.state === "on_demand" ? "首次使用下载" : "未下载"}
                  />
                  <div className="mt-2 grid grid-cols-2 gap-2 max-md:grid-cols-1">
                    {currentModels.map((model) => (
                      <ModelChip
                        key={model.id}
                        model={model}
                        active={config.whisperModelSize === (model.value || model.id) || Boolean(model.selected)}
                        downloading={downloadingModel === model.id}
                        progress={downloadProgress[model.id]}
                        onSelect={() => void saveConfig("whisper_model_size", model.value || model.id)}
                        onDownload={() => void downloadModel(model.id)}
                      />
                    ))}
                  </div>
                </div>

                {config.transcribeModel === "whisperx" && (
                  <div>
                    <FieldLabel label="时间轴对齐" value="按源语言自动匹配" />
                    <div className="mt-2 flex items-center justify-between gap-3 rounded-lg border border-border bg-background/70 p-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${
                          selectedAlignModel?.downloaded ? "bg-emerald-50 text-emerald-600" : "bg-accent-dim text-accent"
                        }`}>
                          <Icon icon="solar:align-bottom-linear" className="h-4 w-4" />
                        </span>
                        <div className="min-w-0">
                          <p className="text-[11px] font-semibold text-text-primary">
                            {config.whisperxAlignmentStrategy === "manual"
                              ? selectedAlignModel?.language_name || "手动指定模型"
                              : config.sourceLanguage === "auto"
                                ? "识别语言后自动匹配"
                                : selectedAlignModel?.language_name || "该语言暂无默认模型"}
                          </p>
                          <p className="mt-0.5 truncate text-[9px] text-text-muted">
                            {selectedAlignModel
                              ? `${selectedAlignModel.downloaded ? "本地已就绪" : "尚未下载"} · ${selectedAlignModel.align_model}`
                              : config.sourceLanguage === "auto"
                                ? "首次使用对应语言时按需加载"
                                : "当前语言没有推荐模型，请在设置中手动选择"}
                          </p>
                        </div>
                      </div>
                      {selectedAlignModel && !selectedAlignModel.downloaded && (
                        <button
                          onClick={() => void downloadModel(selectedAlignModel.id)}
                          disabled={downloadingModel === selectedAlignModel.id}
                          className="shrink-0 rounded-md bg-accent px-2.5 py-1.5 text-[10px] font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-60"
                        >
                          {downloadingModel === selectedAlignModel.id
                            ? `${downloadProgress[selectedAlignModel.id] ?? 0}%`
                            : "下载"}
                        </button>
                      )}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <label className="space-y-1.5">
                    <span className="text-[11px] font-medium text-text-muted">源语言</span>
                    <select
                      value={config.sourceLanguage}
                      onChange={(event) => void saveConfig("source_language", event.target.value)}
                      className="input-field"
                    >
                      {SOURCE_LANGUAGES.map(([id, label]) => (
                        <option key={id} value={id}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="space-y-1.5">
                    <span className="text-[11px] font-medium text-text-muted">批处理</span>
                    <input
                      type="number"
                      min={1}
                      max={16}
                      value={config.whisperxBatchSize}
                      onChange={(event) =>
                        void saveConfig("whisperx_batch_size", Number(event.target.value) || 4)
                      }
                      className="input-field"
                    />
                  </label>
                </div>

                <ToggleLine
                  label="DeepFilterNet 音频增强"
                  description={
                    config.speakerDiarization === "off"
                      ? "适用于单人录音；多人模式会自动跳过"
                      : "多人模式已自动跳过，以保护所有说话人"
                  }
                  checked={
                    config.speakerDiarization === "off" && config.enableAudioEnhancement
                  }
                  disabled={config.speakerDiarization !== "off"}
                  onChange={(value) => void saveConfig("enable_audio_enhancement", value)}
                />
              </div>
            )}
          </Panel>

          <Panel title="硬件状态" icon="solar:cpu-bold-duotone">
            <div className="grid grid-cols-2 gap-3">
              <MetricTile label="芯片" value={hardware?.chip || "检测中"} wide />
              <MetricTile label="加速" value={hardware?.gpu || "检测中"} wide />
              <MetricTile label="线程" value={hardware ? String(hardware.n_threads) : "--"} />
              <MetricTile label="计算" value={hardware?.compute_type || "--"} />
            </div>
          </Panel>

          {missingAlignmentModels.length > 0 && (
            <section
              className="rounded-lg border border-amber-200 bg-amber-50 p-4"
              aria-live="polite"
              aria-label="缺少语言对齐模型"
            >
              <div className="flex items-start gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-amber-100 text-amber-700">
                  <Icon icon="solar:download-minimalistic-bold-duotone" width={20} />
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="text-[13px] font-semibold text-amber-950">
                    自动检测到新的语种
                  </h3>
                  <p className="mt-1 text-[10px] leading-4 text-amber-800">
                    转录内容已经保留。下载对应的对齐模型可生成更准确的词级时间轴；直接继续则对这些语种使用句段时间轴。
                  </p>
                </div>
              </div>

              <div className="mt-3 space-y-2">
                {missingAlignmentModels.map((model) => {
                  const confidence = Math.round(Math.max(0, Math.min(1, model.confidence)) * 100);
                  const ranges = model.ranges.slice(0, 3).map(
                    (range) => `${formatDuration(range.start)}–${formatDuration(range.end)}`
                  );
                  return (
                    <div
                      key={model.model_id}
                      className="rounded-md border border-amber-200 bg-white/75 px-3 py-2.5"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-[11px] font-semibold text-text-primary">
                            {model.language_name}
                            <span className="ml-1.5 font-normal text-text-muted">{model.language.toUpperCase()}</span>
                          </p>
                          <p className="mt-0.5 truncate text-[9px] text-text-muted">
                            {model.model_name} · {model.size || "大小未知"}
                          </p>
                        </div>
                        <span className="shrink-0 text-[10px] font-semibold text-amber-700">
                          {model.ranges.length > 0 && confidence > 0
                            ? `${confidence}%`
                            : "主要语言"}
                        </span>
                      </div>
                      <p className="mt-1.5 text-[9px] leading-4 text-text-muted">
                        {ranges.length > 0
                          ? `${ranges.join("、")}${model.ranges.length > ranges.length ? ` 等 ${model.ranges.length} 处` : ""}`
                          : "主要语言的完整时间轴对齐"}
                      </p>
                      {downloadingModel === model.model_id && (
                        <div className="mt-2 h-1 overflow-hidden rounded-full bg-amber-100">
                          <div
                            className="h-full bg-amber-500 transition-[width]"
                            style={{ width: `${downloadProgress[model.model_id] || 0}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={resolvingAlignment}
                  onClick={() => void downloadMissingAlignmentModels()}
                  className="h-9 rounded-md bg-amber-700 px-3 text-[10px] font-semibold text-white transition-colors hover:bg-amber-800 disabled:cursor-wait disabled:opacity-60"
                >
                  {resolvingAlignment && downloadingModel ? "正在下载" : "下载并继续"}
                </button>
                <button
                  type="button"
                  disabled={resolvingAlignment}
                  onClick={() => void resolveAlignmentDecision("continue")}
                  className="h-9 rounded-md border border-amber-300 bg-white px-3 text-[10px] font-semibold text-amber-900 transition-colors hover:bg-amber-100 disabled:cursor-wait disabled:opacity-60"
                >
                  使用句段时间轴
                </button>
                {canIgnoreMissingLanguages && (
                  <button
                    type="button"
                    disabled={resolvingAlignment}
                    onClick={() => void resolveAlignmentDecision("ignore")}
                    className="col-span-2 h-8 text-[9px] font-medium text-text-muted transition-colors hover:text-amber-900 disabled:cursor-wait disabled:opacity-60"
                  >
                    忽略这些外语片段
                  </button>
                )}
              </div>
            </section>
          )}

          <TaskActionCard
            title="转录任务"
            description={
              !configLoaded
                ? "正在加载识别配置"
                : videoFile
                  ? videoFile.split(/[\\/]/).pop() || videoFile
                  : "请先导入视频或音频文件"
            }
            primaryLabel={taskStatus === "running" ? "转录中" : "开始转录"}
            disabled={!videoFile || !configLoaded || !config.transcribeModel || isProcessing || !diarizationReady}
            progress={taskProgress}
            message={taskMessage}
            running={isProcessing}
            stages={TRANSCRIBE_STAGES}
            currentStage={taskMessage}
            onPrimary={startTranscribe}
            onCancel={cancelTask}
          />

          {fileInfo && <MediaCompactInfo info={fileInfo} />}
        </aside>
      </div>
    </WorkspaceFrame>
  );
}

function SubtitleWorkspace({ startTask, cancelTask }: WorkflowWorkspaceProps) {
  const {
    config,
    isProcessing,
    setConfig,
    setError,
    setIsProcessing,
    setStep,
    subtitleFile,
    subtitles,
    taskMessage,
    taskProgress,
    taskStatus,
  } = useAppStore();
  const [promptFocused, setPromptFocused] = useState(false);
  const [subtitleFocusRequest, setSubtitleFocusRequest] = useState<{
    id: number;
    token: number;
  } | null>(null);
  const promptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const focusTokenRef = useRef(0);
  const quality = useMemo(() => analyzeSubtitleQuality(subtitles), [subtitles]);
  const translatedCount = subtitles.filter((sub) => sub.translated.trim()).length;
  const completion = subtitles.length ? Math.round((translatedCount / subtitles.length) * 100) : 0;

  const jumpToNextEmptyTranslation = useCallback(() => {
    const ids = quality.emptyTranslationIds;
    if (!ids.length) return;
    const currentIndex = subtitleFocusRequest
      ? ids.indexOf(subtitleFocusRequest.id)
      : -1;
    const nextId = ids[(currentIndex + 1) % ids.length];
    focusTokenRef.current += 1;
    setSubtitleFocusRequest({ id: nextId, token: focusTokenRef.current });
  }, [quality.emptyTranslationIds, subtitleFocusRequest]);

  useEffect(
    () => () => {
      if (promptTimerRef.current) clearTimeout(promptTimerRef.current);
    },
    []
  );

  const saveConfig = useCallback(
    async (key: string, value: string | boolean) => {
      const map: Record<string, keyof AppConfig> = {
        target_language: "targetLanguage",
        translator: "translator",
        llm_model: "llmModel",
        need_optimize: "needOptimize",
        need_translate: "needTranslate",
        need_reflect: "needReflect",
        custom_prompt: "customPrompt",
      };
      const mapped = map[key];
      if (mapped) setConfig({ [mapped]: value });
      try {
        await configApi.update(key, value);
      } catch (err) {
        setError(err instanceof Error ? err.message : "配置保存失败");
      }
    },
    [setConfig, setError]
  );

  const startSubtitle = useCallback(async () => {
    if (!subtitleFile) {
      setError("请先导入或生成字幕文件");
      setStep("import");
      return;
    }
    setIsProcessing(true);
    await startTask("subtitle", {
      subtitle_file: subtitleFile,
      target_language: config.targetLanguage,
      translator: config.translator,
      llm_model: config.llmModel,
      need_optimize: config.needOptimize,
      need_translate: config.needTranslate,
      need_reflect: config.needReflect,
      custom_prompt: config.customPrompt || undefined,
    });
  }, [
    config.customPrompt,
    config.llmModel,
    config.needOptimize,
    config.needReflect,
    config.needTranslate,
    config.targetLanguage,
    config.translator,
    setError,
    setIsProcessing,
    setStep,
    startTask,
    subtitleFile,
  ]);

  return (
    <WorkspaceFrame meta={STEP_META.subtitle}>
      <div className="grid h-full min-h-0 grid-cols-[minmax(560px,1fr)_360px] gap-5 max-xl:grid-cols-1">
        <section className="min-h-0 overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
          <SubtitlePanel
            focusRequest={subtitleFocusRequest}
            showPrompt={false}
            showTranslateActions={false}
          />
        </section>

        <aside className="flex min-h-0 flex-col gap-5 overflow-auto pr-1">
          <Panel title="处理模式" icon="solar:magic-stick-3-bold-duotone">
            <div className="grid grid-cols-2 gap-2">
              <ToggleCard
                title="优化断句"
                desc="按语义调整分段和字幕长度"
                checked={config.needOptimize}
                icon="solar:scissors-bold-duotone"
                onChange={(value) => void saveConfig("need_optimize", value)}
              />
              <ToggleCard
                title="翻译"
                desc="生成目标语言字幕"
                checked={config.needTranslate}
                icon="solar:translation-bold-duotone"
                onChange={(value) => void saveConfig("need_translate", value)}
              />
            </div>
          </Panel>

          <Panel title="翻译配置" icon="solar:chat-round-like-bold-duotone">
            <div className="space-y-3">
              <label className="space-y-1.5">
                <span className="text-[11px] font-medium text-text-muted">目标语言</span>
                <select
                  value={config.targetLanguage}
                  onChange={(event) => void saveConfig("target_language", event.target.value)}
                  className="input-field"
                >
                  {TARGET_LANGUAGES.map(([id, label]) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1.5">
                <span className="text-[11px] font-medium text-text-muted">翻译服务</span>
                <select
                  value={config.translator}
                  onChange={(event) => void saveConfig("translator", event.target.value)}
                  className="input-field"
                >
                  {TRANSLATORS.map(([id, label]) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1.5">
                <span className="text-[11px] font-medium text-text-muted">LLM 模型</span>
                <input
                  value={config.llmModel}
                  onChange={(event) => setConfig({ llmModel: event.target.value })}
                  onBlur={(event) => void saveConfig("llm_model", event.target.value)}
                  className="input-field"
                  placeholder="mimo-v2.5-pro"
                />
              </label>
              <div className="pt-3">
                <ToggleLine
                  label="翻译复核"
                  description="完成初译后再次检查完整性和表达"
                  checked={config.needReflect}
                  onChange={(value) => void saveConfig("need_reflect", value)}
                />
              </div>
            </div>
          </Panel>

          <Panel title="翻译要求" icon="solar:pen-new-square-bold-duotone">
            <textarea
              value={config.customPrompt}
              onFocus={() => setPromptFocused(true)}
              onBlur={(event) => {
                setPromptFocused(false);
                void saveConfig("custom_prompt", event.target.value);
              }}
              onChange={(event) => {
                const value = event.target.value;
                setConfig({ customPrompt: value });
                if (promptTimerRef.current) clearTimeout(promptTimerRef.current);
                promptTimerRef.current = setTimeout(() => {
                  void configApi.update("custom_prompt", value);
                }, 700);
              }}
              placeholder="例如：保留汽车品牌、车型和技术名词；中文表达自然，不遗漏限定信息。"
              className={`h-28 w-full resize-none rounded-xl border bg-background p-3 text-[12px] leading-5 text-text-primary outline-none transition ${
                promptFocused ? "border-border-active shadow-[0_0_0_3px_rgba(37,99,235,0.08)]" : "border-border"
              }`}
            />
          </Panel>

          <Panel title="字幕质量" icon="solar:chart-2-bold-duotone">
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between text-[12px]">
                  <span className="text-text-muted">翻译完成度</span>
                  <span className="font-mono text-text-primary">{completion}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-background">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${completion}%` }} />
                </div>
              </div>
              <QualitySummary
                quality={quality}
                compact
                onEmptyTranslationsClick={jumpToNextEmptyTranslation}
              />
            </div>
          </Panel>

          <TaskActionCard
            title="字幕处理任务"
            description={subtitleFile ? subtitleFile.split("/").pop() || subtitleFile : "请先导入或生成字幕"}
            primaryLabel={taskStatus === "running" ? "处理中" : config.needTranslate ? "开始翻译" : "开始断句"}
            disabled={!subtitleFile || isProcessing}
            progress={taskProgress}
            message={taskMessage}
            running={isProcessing}
            stages={["读取字幕", "调整断句", "检查原文", "生成翻译", "质量复核", "保存结果"]}
            currentStage={taskMessage}
            onPrimary={startSubtitle}
            onCancel={cancelTask}
          />
        </aside>
      </div>
    </WorkspaceFrame>
  );
}

function WorkspaceFrame({
  children,
  meta,
}: {
  children: React.ReactNode;
  meta: (typeof STEP_META)[WorkflowStep];
}) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-5">
      <div className="mb-5 flex shrink-0 items-end justify-between gap-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
            <Icon icon={meta.icon} width={17} />
            {meta.eyebrow}
          </div>
          <h1 className="mt-2 text-[28px] font-semibold leading-tight text-text-primary">{meta.title}</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-6 text-text-secondary">{meta.description}</p>
        </div>
        <TaskStatusPill />
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}

function TaskStatusPill() {
  const { backendOnline, taskMessage, taskProgress, taskStatus } = useAppStore();
  if (taskStatus === "running") {
    return (
      <div className="flex shrink-0 items-center gap-3 rounded-full border border-accent/20 bg-accent-dim px-3 py-2">
        <span className="h-2 w-2 rounded-full bg-accent animate-pulse" />
        <span className="max-w-[260px] truncate text-[12px] font-medium text-accent">
          {taskMessage || "处理中"}
        </span>
        <span className="font-mono text-[11px] text-accent">{taskProgress}%</span>
      </div>
    );
  }
  return (
    <div className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-surface px-3 py-2 text-[12px] text-text-muted">
      <span className={`h-2 w-2 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-red-500"}`} />
      {backendOnline ? "后端在线" : "后端离线"}
    </div>
  );
}

function Panel({
  children,
  fill,
  icon,
  title,
}: {
  children: React.ReactNode;
  fill?: boolean;
  icon: string;
  title: string;
}) {
  return (
    <section className={`rounded-2xl border border-border bg-surface p-4 shadow-sm ${fill ? "flex min-h-0 flex-1 flex-col overflow-hidden" : ""}`}>
      <div className="mb-3.5 flex min-h-8 shrink-0 items-center gap-2.5">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-dim text-accent">
          <Icon icon={icon} width={18} />
        </span>
        <h2 className="text-[14px] font-semibold text-text-primary">{title}</h2>
      </div>
      <div className={fill ? "min-h-0 flex-1 overflow-hidden" : ""}>{children}</div>
    </section>
  );
}

function MetricTile({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`rounded-xl border border-border bg-surface p-3 ${wide ? "col-span-2" : ""}`}>
      <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-text-muted">{label}</p>
      <p className="mt-1 truncate text-[13px] font-semibold text-text-primary">{value}</p>
    </div>
  );
}

function CheckRow({
  icon,
  label,
  neutral,
  ok,
  value,
}: {
  icon: string;
  label: string;
  neutral?: boolean;
  ok: boolean;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-border bg-background p-3">
      <div className="flex items-center gap-3">
        <Icon icon={icon} width={18} className={ok ? "text-emerald-600" : neutral ? "text-text-muted" : "text-red-500"} />
        <span className="text-[13px] font-medium text-text-primary">{label}</span>
      </div>
      <span
        className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${
          ok
            ? "bg-emerald-50 text-emerald-700"
            : neutral
            ? "bg-surface text-text-muted"
            : "bg-red-50 text-red-600"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function FlowStep({
  active,
  done,
  index,
  title,
}: {
  active?: boolean;
  done?: boolean;
  index: string;
  title: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold ${
          done
            ? "bg-emerald-500 text-white"
            : active
            ? "bg-accent text-white"
            : "bg-background text-text-muted"
        }`}
      >
        {done ? <Icon icon="solar:check-read-bold" width={14} /> : index}
      </div>
      <span className={`text-[13px] ${active || done ? "font-medium text-text-primary" : "text-text-muted"}`}>
        {title}
      </span>
    </div>
  );
}

function EmptyState({
  icon,
  minHeight = "min-h-[140px]",
  title,
}: {
  icon: string;
  minHeight?: string;
  title: string;
}) {
  return (
    <div className={`flex h-full ${minHeight} flex-col items-center justify-center rounded-xl border border-dashed border-border bg-background/70 p-6 text-center`}>
      <Icon icon={icon} width={28} className="text-text-muted" />
      <p className="mt-2 text-[12px] text-text-muted">{title}</p>
    </div>
  );
}

function FieldLabel({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[11px] font-semibold text-text-secondary">{label}</span>
      <span className="shrink-0 rounded-md bg-background px-2 py-1 text-[10px] font-medium text-text-muted">{value}</span>
    </div>
  );
}

function ModelChip({
  active,
  downloading,
  model,
  onDownload,
  onSelect,
  progress,
}: {
  active: boolean;
  downloading: boolean;
  model: AsrModelInfo;
  onDownload: () => void;
  onSelect: () => void;
  progress?: number;
}) {
  const downloadable = model.downloadable !== false;
  return (
    <button
      onClick={model.downloaded || !downloadable ? onSelect : onDownload}
      className={`group flex min-h-10 min-w-0 items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left text-[11px] transition-[border-color,background-color,transform] duration-200 active:translate-y-px ${
        active
          ? "border-accent bg-accent-dim text-accent"
          : "border-border bg-background text-text-secondary hover:border-border-active"
      }`}
    >
      <span className="flex min-w-0 items-baseline gap-2">
        <span className="truncate font-semibold">{model.name || model.id.split("/").pop() || model.id}</span>
        <span className="shrink-0 text-[10px] text-text-muted">{model.size}</span>
      </span>
      <span className="flex shrink-0 items-center gap-1 text-[10px] font-medium">
        {model.downloaded ? (
          <>
            <span className="text-emerald-700">可用</span>
            <Icon icon="solar:check-circle-bold" width={14} className="text-emerald-600" />
          </>
        ) : downloading ? (
          <span className="font-mono text-accent">{progress ?? 0}%</span>
        ) : !downloadable ? (
          <span className="text-text-muted">{model.state === "on_demand" ? "首次使用下载" : "不可下载"}</span>
        ) : (
          <span className="text-accent opacity-80 group-hover:opacity-100">下载</span>
        )}
      </span>
    </button>
  );
}

function ModelRow({
  active,
  downloading,
  model,
  onDownload,
  onSelect,
  progress,
}: {
  active: boolean;
  downloading: boolean;
  model: AsrModelInfo;
  onDownload: () => void;
  onSelect: () => void;
  progress?: number;
}) {
  return (
    <div className={`rounded-xl border p-3 ${active ? "border-accent bg-accent-dim" : "border-border bg-background"}`}>
      <div className="flex items-center justify-between gap-3">
        <button onClick={onSelect} className="min-w-0 text-left">
          <p className="truncate text-[13px] font-semibold text-text-primary">{model.name || model.id}</p>
          <p className="mt-0.5 text-[11px] text-text-muted">{model.size} · {model.type}</p>
        </button>
        <button
          onClick={model.downloaded ? onSelect : onDownload}
          className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${
            model.downloaded ? "bg-emerald-50 text-emerald-700" : "bg-accent text-white"
          }`}
        >
          {model.downloaded ? "可用" : downloading ? `${progress ?? 0}%` : "下载"}
        </button>
      </div>
    </div>
  );
}

function ToggleLine({
  checked,
  description,
  disabled = false,
  label,
  onChange,
}: {
  checked: boolean;
  description?: string;
  disabled?: boolean;
  label: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-4 rounded-xl border border-border bg-background p-3 text-left transition hover:border-border-active disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:border-border"
    >
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-text-primary">{label}</span>
        {description && (
          <span className="mt-0.5 block text-[10px] leading-4 text-text-muted">{description}</span>
        )}
      </span>
      <span className={`relative h-[22px] w-10 rounded-full transition ${checked ? "bg-accent" : "bg-black/10"}`}>
        <span
          className={`absolute top-[3px] h-4 w-4 rounded-full bg-white shadow-sm transition ${
            checked ? "left-[21px]" : "left-[3px]"
          }`}
        />
      </span>
    </button>
  );
}

function ToggleCard({
  checked,
  desc,
  icon,
  onChange,
  title,
}: {
  checked: boolean;
  desc: string;
  icon: string;
  onChange: (value: boolean) => void;
  title: string;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`rounded-xl border p-3 text-left transition ${
        checked ? "border-accent bg-accent-dim" : "border-border bg-background hover:border-border-active"
      }`}
    >
      <div className="flex items-center justify-between">
        <Icon icon={icon} width={19} className={checked ? "text-accent" : "text-text-muted"} />
        <span className={`h-2 w-2 rounded-full ${checked ? "bg-accent" : "bg-black/20"}`} />
      </div>
      <p className="mt-3 text-[13px] font-semibold text-text-primary">{title}</p>
      <p className="mt-1 text-[10px] leading-4 text-text-muted">{desc}</p>
    </button>
  );
}

function TaskActionCard({
  currentStage,
  description,
  disabled,
  message,
  onCancel,
  onPrimary,
  primaryLabel,
  progress,
  running,
  stages,
  title,
}: {
  currentStage: string;
  description: string;
  disabled: boolean;
  message: string;
  onCancel: () => Promise<void>;
  onPrimary: () => void;
  primaryLabel: string;
  progress: number;
  running: boolean;
  stages: readonly string[];
  title: string;
}) {
  return (
    <section className="rounded-2xl border border-border bg-text-primary p-4 text-white shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold">{title}</h2>
          <p className="mt-1 max-w-[280px] truncate text-[12px] text-white/60">{description}</p>
        </div>
        {running && <span className="rounded-full bg-white/10 px-2 py-1 font-mono text-[11px]">{progress}%</span>}
      </div>
      {running && (
        <div className="mt-4">
          <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-2 truncate text-[12px] text-white/70">{message || currentStage}</p>
          <div className="mt-3 grid grid-cols-3 gap-1.5">
            {stages.map((stage, index) => (
              <span
                key={stage}
                className={`rounded-full px-2 py-1 text-center text-[10px] ${
                  progress >= (index / stages.length) * 100 ? "bg-white/14 text-white" : "bg-white/6 text-white/45"
                }`}
              >
                {stage}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="mt-4 flex items-center gap-2">
        {running ? (
          <button
            onClick={() => void onCancel()}
            className="w-full rounded-full border border-white/20 px-4 py-2 text-[13px] font-medium text-white transition hover:bg-white/10"
          >
            取消任务
          </button>
        ) : (
          <button
            onClick={onPrimary}
            disabled={disabled}
            className="w-full rounded-full bg-accent px-4 py-2 text-[13px] font-semibold text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {primaryLabel}
          </button>
        )}
      </div>
    </section>
  );
}

function MediaCompactInfo({ info }: { info: FileInfo }) {
  return (
    <Panel title="素材摘要" icon="solar:video-library-bold-duotone">
      <div className="space-y-2 text-[12px]">
        <InfoLine label="文件" value={info.filename} />
        <InfoLine label="时长" value={formatDuration(info.duration)} />
        <InfoLine label="大小" value={formatSize(info.size)} />
        <InfoLine
          label="画面"
          value={info.video ? `${info.video.width}x${info.video.height} · ${info.video.codec}` : "无视频流"}
        />
      </div>
    </Panel>
  );
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg bg-background px-3 py-2">
      <span className="shrink-0 text-text-muted">{label}</span>
      <span className="min-w-0 truncate text-right font-medium text-text-primary">{value}</span>
    </div>
  );
}

function LiveSubtitleList({
  isLive = false,
  subtitles,
}: {
  isLive?: boolean;
  subtitles: SubtitleSegment[];
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const followTailRef = useRef(true);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container || !followTailRef.current) return;
    container.scrollTop = container.scrollHeight;
  }, [subtitles.length]);

  if (!subtitles.length) {
    return <EmptyState icon="solar:playlist-bold-duotone" title="转录开始后会显示实时字幕" />;
  }
  return (
    <div
      ref={scrollRef}
      className="h-full min-h-0 overflow-auto pr-1"
      onScroll={(event) => {
        const target = event.currentTarget;
        followTailRef.current = target.scrollHeight - target.scrollTop - target.clientHeight < 80;
      }}
    >
      <div className="space-y-2">
        {subtitles.map((sub) => (
          <div key={sub.id} className="rounded-xl border border-border bg-background p-3">
            <div className="mb-1 flex items-center gap-2 font-mono text-[10px] text-text-muted">
              <span>{String(sub.id).padStart(3, "0")}</span>
              <span>{sub.start}</span>
              <span>→</span>
              <span>{sub.end}</span>
            </div>
            <div className="flex items-start gap-2">
              {sub.speaker && (
                <span className="mt-0.5 shrink-0 rounded bg-accent-dim px-1.5 py-0.5 font-mono text-[10px] font-semibold text-accent">
                  {sub.speaker.match(/(\d+)\s*$/)?.[1]
                    ? `S${sub.speaker.match(/(\d+)\s*$/)?.[1]}`
                    : sub.speaker}
                </span>
              )}
              <p className="text-[12px] leading-5 text-text-primary">{sub.text}</p>
            </div>
          </div>
        ))}
        {isLive && (
          <div className="flex h-9 items-center justify-center gap-2 text-[11px] text-text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
            正在接收转录结果
          </div>
        )}
      </div>
    </div>
  );
}

function QualitySummary({
  compact,
  onEmptyTranslationsClick,
  quality,
}: {
  compact?: boolean;
  onEmptyTranslationsClick?: () => void;
  quality: SubtitleQuality;
}) {
  const items = [
    { label: "重叠", value: quality.overlaps.length, tone: quality.overlaps.length ? "bad" : "good" },
    { label: "过长", value: quality.longDurations.length, tone: quality.longDurations.length ? "warn" : "good" },
    { label: "紧贴", value: quality.tightGaps.length, tone: quality.tightGaps.length ? "warn" : "good" },
    {
      label: "空译文",
      value: quality.emptyTranslations,
      tone: quality.emptyTranslations ? "warn" : "good",
      onClick: onEmptyTranslationsClick,
    },
  ];
  return (
    <div className="space-y-3">
      <div className={`grid ${compact ? "grid-cols-4 max-sm:grid-cols-2" : "grid-cols-2"} gap-2`}>
        {items.map((item) => {
          const interactive = Boolean(item.onClick && item.value > 0);
          return (
            <button
              key={item.label}
              type="button"
              disabled={!interactive}
              onClick={interactive ? item.onClick : undefined}
              aria-label={interactive ? `定位空译文，共 ${item.value} 条` : undefined}
              title={interactive ? `定位空译文，共 ${item.value} 条；再次点击查看下一条` : undefined}
              className={`relative rounded-xl border border-border bg-background p-3 text-left transition-[border-color,background-color,transform] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 ${
                interactive
                  ? "cursor-pointer hover:border-amber-300 hover:bg-amber-50/60 active:translate-y-px"
                  : "cursor-default"
              }`}
            >
              <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-text-muted">
                {item.label}
              </p>
              {interactive && (
                <Icon
                  icon="solar:map-arrow-right-bold"
                  width={14}
                  className="absolute right-3 top-3 text-amber-600"
                />
              )}
              <p
                className={`mt-1 font-mono text-[20px] font-semibold ${
                  item.tone === "bad"
                    ? "text-red-600"
                    : item.tone === "warn"
                      ? "text-amber-600"
                      : "text-emerald-600"
                }`}
              >
                {item.value}
              </p>
            </button>
          );
        })}
      </div>
      {!compact && (
        <div className="rounded-xl bg-background p-3">
          <p className="text-[12px] leading-5 text-text-muted">
            当前检测基于 SRT 时间轴和文本结构。无语音覆盖仍需要结合音频 VAD，但重叠、异常长段和过密间隔会优先暴露。
          </p>
        </div>
      )}
    </div>
  );
}
