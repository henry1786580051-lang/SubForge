"use client";

import { downloadLabel } from "@/lib/downloadProgress";
import type { TaskInfo } from "@/lib/api";
import { Panel, ToggleLine, TaskActionCard } from "@/components/WorkspaceControls";
import { importSubtitleDocument } from "@/lib/documentOperations";
import { modelPresentation } from "@/lib/modelPresentation";
import { InspectorDisclosure } from "@/components/InspectorDisclosure";
import { useUiStore } from "@/store/uiStore";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@/components/Icon";
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
import type { TaskStarter } from "@/lib/useTaskMonitor";
import { SubtitleChecks } from "@/components/SubtitleChecks";
import { SubtitlePanel } from "@/components/SubtitlePanel";
import {
  ASR_ENGINES,
  SOURCE_LANGUAGES,
  STEP_META,
  TARGET_LANGUAGES,
  TRANSLATORS,
} from "@/features/workflow/catalog";
import {
  analyzeSubtitleQuality,
  type QualityKey,
  QUALITY_LABELS,
} from "@/features/workflow/quality";
import { LLM_PROVIDERS } from "@/features/settings/catalog";

type AppConfig = ReturnType<typeof useAppStore.getState>["config"];

interface WorkflowWorkspaceProps {
  startTask: TaskStarter;
  cancelTask: () => Promise<void>;
}

export function WorkflowWorkspace({ startTask, cancelTask }: WorkflowWorkspaceProps) {
  const { step } = useAppStore();

  return (
    <main className="flex-1 min-h-0">
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
    fileInfo,
    setFileInfo,
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
        await importSubtitleDocument(async () => {
          const uploaded = await filesApi.upload(file);
          return subtitlesApi.load(uploaded.file_path);
        });
        useAppStore.getState().addToast("字幕已导入", "success");
      } catch (err) {
        useAppStore
          .getState()
          .setError(err instanceof Error ? err.message : "字幕导入失败");
      } finally {
        setUploading(null);
      }
    },
    []
  );

  const chooseMedia = useCallback(async (kind: "media" | "any" = "media") => {
    try {
      const selected = await openNativeFile(kind);
      if (!selected.available) {
        mediaInputRef.current?.click();
        return;
      }
      if (!selected.path) return;
      if (/\.(srt|vtt|ass)$/i.test(selected.path)) {
        setUploading("subtitle");
        await importSubtitleDocument(() => subtitlesApi.load(selected.path!));
        useAppStore.getState().addToast("字幕已导入", "success");
        return;
      }
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
      await importSubtitleDocument(() => subtitlesApi.load(selected.path!));
      useAppStore.getState().addToast("字幕已导入", "success");
    } catch (err) {
      useAppStore
        .getState()
        .setError(err instanceof Error ? err.message : "字幕导入失败");
    } finally {
      setUploading(null);
    }
  }, []);

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

  const openRequested = useUiStore((state) => state.openRequested);
  useEffect(() => {
    if (!openRequested) return;
    useUiStore.getState().consumeOpen();
    // This effect consumes an application command and opens the native file dialog.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void chooseMedia("any");
  }, [openRequested, chooseMedia]);

  return (
    <WorkspaceFrame meta={STEP_META.import}>
      <div className="import-workspace">
        <section className={`import-drop ${dragType ? "is-dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setDragType("media"); }}
          onDragLeave={() => setDragType(null)} onDrop={handleDrop} aria-busy={!!uploading}>
          <input ref={mediaInputRef} type="file" accept="video/*,audio/*" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadMedia(file); event.target.value = ""; }} />
          <input ref={subtitleInputRef} type="file" accept=".srt,.vtt,.ass" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadSubtitle(file); event.target.value = ""; }} />
          <span className="import-symbol"><Icon icon="solar:subtitles-linear" width={40} /></span>
          <h2>{uploading ? "正在读取文件…" : videoFile || subtitleFile ? "素材已准备好" : "从一份素材开始"}</h2>
          <p className="import-description">拖入视频、音频或字幕，开始转录、翻译与精校。</p>
          <div className="flex flex-wrap justify-center gap-3 mt-7">
            <button className="primary-button" disabled={!!uploading} onClick={() => void chooseMedia()}><Icon icon="solar:folder-open-linear" width={18} />选择视频或音频 <kbd>⌘O</kbd></button>
            <button className="toolbar-button" disabled={!!uploading} onClick={() => void chooseSubtitle()}>导入已有字幕</button>
          </div>
          <p className="mt-5 text-[12px] text-text-muted">MP4、MOV、MP3、WAV · SRT、VTT、ASS</p>
        </section>
        {(videoFile || subtitleFile) && <div className="import-files">
          {videoFile && <div className="import-file-row"><Icon icon="solar:video-library-bold-duotone" width={24} className="text-accent shrink-0" /><div className="min-w-0 flex-1"><strong>{fileInfo?.filename || videoFile.split("/").pop()}</strong><p>{fileInfo ? `${formatDuration(fileInfo.duration)} · ${formatSize(fileInfo.size)} · ${fileInfo.audio_tracks.length} 条音轨` : "正在读取媒体信息"}</p></div><button className="toolbar-button" disabled={!fileInfo} onClick={() => useAppStore.getState().setStep("transcribe")}>前往转录<Icon icon="solar:arrow-right-linear" width={16} /></button></div>}
          {subtitleFile && <div className="import-file-row"><Icon icon="solar:document-text-linear" width={24} className="text-accent shrink-0" /><div className="min-w-0 flex-1"><strong>{subtitleFile.split("/").pop()}</strong><p>{subtitles.length} 条字幕 · 可直接编辑与翻译</p></div><button className="toolbar-button" onClick={() => useAppStore.getState().setStep("subtitle")}>打开字幕<Icon icon="solar:arrow-right-linear" width={16} /></button></div>}
        </div>}
        <p className="import-footnote">从素材到成片，一处完成。导入 → 转录 → 字幕</p>
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
    platform: string;
    arch: string;
    chip: string;
    device: string;
    n_threads: number;
    compute_type: string;
    gpu: string;
  } | null>(null);
  const [models, setModels] = useState<AsrModelInfo[]>([]);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const downloadActive = useRef(false);
  const [downloadTask, setDownloadTask] = useState<TaskInfo | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number | undefined>>({});
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
        detect_additional_languages: "detectAdditionalLanguages",
        enable_audio_enhancement: "enableAudioEnhancement",
        speaker_diarization: "speakerDiarization",
        speaker_count: "speakerCount",
      };
      const mapped = map[key];
      try {
        await configApi.update(key, value);
        if (mapped) setConfig({ [mapped]: value });
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
    await startTask("transcribe", {
      file_path: videoFile,
      model: config.transcribeModel,
      language: config.sourceLanguage,
    });
  }, [config.sourceLanguage, config.transcribeModel, configLoaded, setError, setStep, startTask, videoFile]);

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
    (model) => (model.value || model.id) === config.whisperModelSize
  ) ?? currentModels.find((model) => model.selected);
  const alignmentLanguage = config.sourceLanguage === "nb" ? "no" : config.sourceLanguage;
  const selectedAlignModel = alignmentModels.find((model) =>
    config.whisperxAlignmentStrategy === "manual"
      ? model.align_model === config.whisperxAlignModel || model.id === config.whisperxAlignModel
      : alignmentLanguage !== "auto" && model.language === alignmentLanguage
  );
  const quality = useMemo(() => analyzeSubtitleQuality(subtitles), [subtitles]);

  const downloadModel = useCallback(async (modelId: string): Promise<boolean> => {
    if (downloadActive.current) return false;
    downloadActive.current = true;
    setDownloadingModel(modelId);
    setDownloadTask(null);
    setDownloadProgress((prev) => ({ ...prev, [modelId]: undefined }));
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
          setDownloadTask(task);
          setDownloadProgress((prev) => ({ ...prev, [modelId]: task.download?.progress ?? undefined }));
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
      downloadActive.current = false;
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
      !["auto", "hybrid"].includes(taskAttention.source_mode || "")
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
    taskAttention?.type === "missing_alignment_models" &&
    ["auto", "hybrid"].includes(taskAttention.source_mode || "")
      ? taskAttention.models
      : [];
  const hybridLanguageAttention = taskAttention?.source_mode === "hybrid";
  const canIgnoreMissingLanguages =
    missingAlignmentModels.length > 0 &&
    missingAlignmentModels.every((model) => model.ranges.length > 0);

  return (
    <WorkspaceFrame meta={STEP_META.transcribe}>
      <div className="workspace-layout">
        <section className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-3">
          <Panel
            title={subtitles.length ? `转录结果 · ${subtitles.length} 条` : "实时转录结果"}
            icon="solar:playlist-bold-duotone"
            fill
          >
            <LiveSubtitleList subtitles={subtitles} isLive={taskStatus === "running"} />
          </Panel>
          <Panel title="时间轴质量" icon="solar:shield-warning-bold-duotone">
            <SubtitleChecks quality={quality} processing={isProcessing} />
          </Panel>
        </section>

        <Inspector title="转录配置" footer={
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

            currentStage={taskMessage}
            onPrimary={startTranscribe}
            onCancel={cancelTask}
          />
        }>
      {downloadingModel && <div className="download-status" role="status">
        <strong>模型下载</strong><span>{downloadLabel(downloadTask)}</span>
      </div>}
          <Panel title="识别设置" icon="solar:microphone-linear">
            <label className="inspector-field">源语言
              <select value={config.sourceLanguage} className="input-field" onChange={(event) => void saveConfig("source_language", event.target.value)}>
                {SOURCE_LANGUAGES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
              </select>
            </label>
            <div className="inspector-model" title={config.whisperModelSize}>
              <span>{ASR_ENGINES.find((engine) => engine.id === config.transcribeModel)?.name || config.transcribeModel}</span>

            </div>
          </Panel>
          <InspectorDisclosure title={<>识别引擎</>}>
          <Panel title="识别引擎" icon="solar:tuning-square-2-bold-duotone">
            <div className="inspector-engines" role="group" aria-label="识别引擎">
              {ASR_ENGINES.map((engine) => {
                const unsupported = engine.id === "whisperx" && !config.whisperxSupported;
                return (
                <button
                  key={engine.id}
                  disabled={unsupported}
                  onClick={() => void saveConfig("transcribe_model", engine.id)}
                  className="inspector-engine"
                  aria-pressed={config.transcribeModel === engine.id}
                >
                  <span className="inspector-engine-icon"><Icon icon={engine.icon} width={18} /></span>
                  <span className="inspector-engine-copy">
                    <strong>{engine.name}</strong>
                    <small>{unsupported ? "当前平台不支持" : engine.desc}</small>
                  </span>
                  <svg className="inspector-engine-check" aria-hidden="true" width="16" height="16" viewBox="0 0 16 16"><path d="m3 8 3 3 7-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </button>
              )})}
            </div>
          </Panel>
          </InspectorDisclosure>

          <Panel title="识别模型" icon="solar:layers-bold-duotone">
            {config.transcribeModel === "whisper_api" ? (
              <EmptyState icon="solar:cloud-bold-duotone" title="云端模型在设置页配置" />
            ) : (
              <div className="space-y-4">
                <div>
                  <ModelSummary model={selectedModel} fallback={config.whisperModelSize} />
                  <InspectorDisclosure title={<>更换模型<span className="inspector-summary-value">{currentModels.length} 个选项</span></>}>
                    <div className="model-options">
                    {currentModels.map((model) => (
                      <ModelChip key={model.id} model={model} active={model.id === selectedModel?.id}
                        downloading={downloadingModel === model.id} progress={downloadProgress[model.id]}
                        onSelect={() => void saveConfig("whisper_model_size", model.value || model.id)}
                        onDownload={() => void downloadModel(model.id)} />
                    ))}
                    {!currentModels.length && <p className="model-option-meta">当前引擎暂无可切换模型。</p>}
                    </div>
                  </InspectorDisclosure>
                </div>

                <InspectorDisclosure title={<>高级识别选项</>}>
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
                            ? (downloadProgress[selectedAlignModel.id] == null ? "下载中" : `${downloadProgress[selectedAlignModel.id]}%`)
                            : "下载"}
                        </button>
                      )}
                    </div>
                  </div>
                )}

                <div className="inspector-form-rows">
                  <label className="inspector-field">
                    <span className="text-[11px] font-medium text-text-muted">批处理</span>
                    <select value={config.whisperxBatchSize}
                      onChange={(event) => void saveConfig("whisperx_batch_size", Number(event.target.value))}
                      className="input-field">
                      {Array.from({ length: 16 }, (_, index) => index + 1).map((size) => <option key={size} value={size}>{size}</option>)}
                    </select>
                  </label>
                </div>

                <ToggleLine
                  label="自动补录其他语言"
                  description={
                    config.sourceLanguage === "auto"
                      ? "自动语言模式已经包含多语言检测与按需对齐"
                      : hardware === null
                        ? "正在确认设备是否支持 MLX 多语言补录"
                      : hardware.platform !== "Darwin" || hardware.arch !== "arm64"
                        ? "当前仅支持 Apple Silicon 的 MLX Whisper"
                        : "保留所选主语言，并对稳定检测到的其他语言局部重识别"
                  }
                  checked={
                    config.sourceLanguage !== "auto" &&
                    hardware?.platform === "Darwin" &&
                    hardware.arch === "arm64" &&
                    config.detectAdditionalLanguages
                  }
                  disabled={
                    config.sourceLanguage === "auto" ||
                    hardware === null ||
                    hardware.platform !== "Darwin" ||
                    hardware.arch !== "arm64"
                  }
                  onChange={(value) => void saveConfig("detect_additional_languages", value)}
                />

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
                </InspectorDisclosure>
              </div>
            )}
          </Panel>

          <Panel title="说话人识别" icon="solar:users-group-rounded-bold-duotone">
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div>

                  <p className="mt-1 text-[10px] leading-4 text-text-muted">
                    区分对话角色，保留短促插话。
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

              <div className="inspector-segments" role="group" aria-label="多人语音识别模式">
                <span aria-hidden="true" className="inspector-segment-selection" style={{ transform: `translateX(${["off", "two", "auto", "fixed"].indexOf(config.speakerDiarization) * 100}%)` }} />
                {([["off", "关闭"], ["two", "双人"], ["auto", "自动"], ["fixed", "指定人数"]] as const).map(([value, label]) => (
                  <button
                    key={value}
                    onClick={() => void saveConfig("speaker_diarization", value)}
                    aria-pressed={config.speakerDiarization === value}
                    className="inspector-segment"
                  >
                    {label}
                  </button>
                ))}
              </div>

              {config.speakerDiarization === "auto" && (
                <p className="rounded-md bg-background px-3 py-2 text-[10px] leading-4 text-text-muted">
                  自动识别 1–10 位说话人
                </p>
              )}

              {config.speakerDiarization === "two" && (
                <p className="rounded-md bg-background px-3 py-2 text-[10px] leading-4 text-text-muted">
                  以两位主要对话者为基准，也允许广告或插播中出现短暂的第三声音。
                </p>
              )}

              {config.speakerDiarization === "fixed" && (
                <label className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2">
                  <span>
                    <span className="block text-[11px] font-semibold text-text-primary">说话人数</span>
                    <span className="mt-0.5 block text-[9px] text-text-muted">严格按指定人数聚类，适合人数确定且无插播的素材</span>
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
                <InspectorDisclosure open={!diarizationReady} title={<>{diarizationReady ? "模型与凭据 · 已就绪" : "配置说话人模型"}</>}><div className="space-y-3 pt-3">
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
                    多人模式使用原始音轨，以保护较弱声音。
                  </p>
                </div></InspectorDisclosure>
              )}
            </div>
          </Panel>

          <InspectorDisclosure title={<>硬件状态</>}>
          <Panel title="硬件状态" icon="solar:cpu-bold-duotone">
            <div className="grid grid-cols-2 gap-3">
              <MetricTile label="芯片" value={hardware?.chip || "检测中"} wide />
              <MetricTile label="加速" value={hardware?.gpu || "检测中"} wide />
              <MetricTile label="线程" value={hardware ? String(hardware.n_threads) : "--"} />
              <MetricTile label="计算" value={hardware?.compute_type || "--"} />
            </div>
          </Panel>
          </InspectorDisclosure>

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
                    {hybridLanguageAttention ? "主语言之外检测到新的语种" : "自动检测到新的语种"}
                  </h3>
                  <p className="mt-1 text-[10px] leading-4 text-amber-800">
                    {hybridLanguageAttention
                      ? "主语言转录已经保留。下载对应模型后，将只补录这些外语区间并生成词级时间轴。"
                      : "转录内容已经保留。下载对应的对齐模型可生成更准确的词级时间轴；直接继续则对这些语种使用句段时间轴。"}
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
                        <div className={`model-download-track${downloadProgress[model.model_id] == null ? " is-indeterminate" : ""}`}
                          role="progressbar" aria-label="对齐模型下载进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={downloadProgress[model.model_id]}>
                          <span style={{ width: downloadProgress[model.model_id] == null ? "35%" : `${downloadProgress[model.model_id]}%` }} />
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



          {fileInfo && <MediaCompactInfo info={fileInfo} />}
        </Inspector>
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
  const llmProviderName = useMemo(
    () =>
      LLM_PROVIDERS.find((provider) => provider.id === config.llmProvider)?.name ||
      config.llmProvider,
    [config.llmProvider]
  );
  const [checkSelection, setCheckSelection] = useState<{ key: QualityKey; file: string | null } | null>(null);
  const activeCheck = checkSelection?.file === subtitleFile ? checkSelection?.key : null;
  const checkIds = activeCheck ? quality.ids[activeCheck] : null;
  const navigateCheck = (direction: number, key = activeCheck) => {
    if (!key || !quality.ids[key].length) return;
    const ids = quality.ids[key];
    const current = subtitleFocusRequest ? ids.indexOf(subtitleFocusRequest.id) : -1;
    const index = current < 0 ? 0 : (current + direction + ids.length) % ids.length;
    setSubtitleFocusRequest({ id: ids[index], token: ++focusTokenRef.current });
  };
  const selectCheck = (key: QualityKey) => {
    useAppStore.getState().deselectAll();
    setCheckSelection({ key, file: subtitleFile });
    const id = quality.ids[key][0];
    if (id !== undefined) setSubtitleFocusRequest({ id, token: ++focusTokenRef.current });
  };

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
        need_optimize: "needOptimize",
        need_translate: "needTranslate",
        need_reflect: "needReflect",
        custom_prompt: "customPrompt",
      };
      const mapped = map[key];
      try {
        await configApi.update(key, value);
        if (mapped) setConfig({ [mapped]: value });
      } catch (err) {
        setError(err instanceof Error ? err.message : "配置保存失败");
      }
    },
    [setConfig, setError]
  );

  const startSubtitle = useCallback(async () => {
    try { await configApi.flush(); }
    catch (err) { setError(err instanceof Error ? err.message : "配置保存失败"); return; }
    const { config, subtitleFile, videoFile } = useAppStore.getState();
    if (!subtitleFile) {
      setError("请先导入或生成字幕文件");
      setStep("import");
      return;
    }
    await startTask("subtitle", {
      subtitle_file: subtitleFile,
      media_file: videoFile || undefined,
      target_language: config.targetLanguage,
      translator: config.translator,
      llm_provider: config.llmProvider,
      llm_model: config.llmModel,
      need_optimize: config.needOptimize,
      need_translate: config.needTranslate,
      need_reflect: config.needReflect,
      custom_prompt: config.customPrompt,
    });
  }, [setError, setStep, startTask]);

  return (
    <WorkspaceFrame meta={STEP_META.subtitle}>
      <div className="workspace-layout">
        <section className="min-h-0 overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
          <SubtitlePanel
            startTask={startTask}
            focusRequest={subtitleFocusRequest}
            checkFilter={activeCheck && !isProcessing ? {
              label: QUALITY_LABELS[activeCheck], ids: checkIds || [],
              reasons: quality.reasons[activeCheck],
              onPrevious: () => navigateCheck(-1), onNext: () => navigateCheck(1),
              onClear: () => { setCheckSelection(null); setSubtitleFocusRequest(null); },
            } : null}
            showPrompt={false}
            showTranslateActions={false}
          />
        </section>

        <Inspector title="字幕处理" footer={
          <TaskActionCard
            title="字幕处理任务"
            description={subtitleFile ? subtitleFile.split("/").pop() || subtitleFile : "请先导入或生成字幕"}
            primaryLabel={taskStatus === "running" ? "处理中" : config.needTranslate ? (config.needOptimize ? "优化并翻译" : "开始翻译") : "优化断句"}
            disabled={!subtitleFile || isProcessing || (!config.needOptimize && !config.needTranslate)}
            progress={taskProgress}
            message={taskMessage}
            running={isProcessing}

            currentStage={taskMessage}
            onPrimary={startSubtitle}
            onCancel={cancelTask}
          />
        }>
          <Panel title="处理步骤" icon="solar:magic-stick-3-bold-duotone">
            <div className="inspector-rows">
              <ToggleLine label="优化断句" description="按语义调整分段和字幕长度" checked={config.needOptimize} onChange={(value) => void saveConfig("need_optimize", value)} />
              <ToggleLine label="翻译" description="生成目标语言字幕" checked={config.needTranslate} onChange={(value) => void saveConfig("need_translate", value)} />
              <div>
                <ToggleLine
                  label="翻译复核"
                  description="完成初译后再次检查完整性和表达"
                  disabled={!config.needTranslate}
                  checked={config.needReflect}
                  onChange={(value) => void saveConfig("need_reflect", value)}
                />
              </div>
            </div>
          </Panel>

          <div className={`inspector-reveal ${config.needTranslate ? "is-open" : ""}`} inert={!config.needTranslate} aria-hidden={!config.needTranslate}>
            <div className="inspector-reveal-content">
          <Panel title="翻译配置" icon="solar:chat-round-like-bold-duotone">
            <div className="inspector-rows">
              <label className="inspector-field">
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
              <label className="inspector-field">
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
              <div className="inspector-field">
                <span className="text-[11px] font-medium text-text-muted">当前 LLM</span>
                <div className="inspector-current-model" title={config.llmModel}>
                  <strong>{llmProviderName}</strong>
                  <small>{config.llmModel || "未选择模型"}</small>
                </div>
              </div>

            </div>
          </Panel>

          <InspectorDisclosure title={<>翻译要求<span className="inspector-summary-value">{config.customPrompt.trim() ? "已自定义" : "默认"}</span></>}>
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
          </InspectorDisclosure>

            </div>
          </div>
          <Panel title="字幕检查" icon="solar:chart-2-bold-duotone">
            <SubtitleChecks quality={quality} active={activeCheck} onSelect={selectCheck} processing={isProcessing} />
          </Panel>


        </Inspector>
      </div>
    </WorkspaceFrame>
  );
}

function Inspector({ children, title, footer }: { children: React.ReactNode; title: string; footer: React.ReactNode }) {
  const { inspectorOpen: open, toggleInspector } = useUiStore();
  return <aside className="workspace-inspector" inert={!open} aria-label="处理选项">
    <header className="inspector-header">
      <div><h2>{title}</h2><p>修改后自动保存为后续任务默认值</p></div>
      <button type="button" className="inspector-close" onClick={() => { toggleInspector(); document.querySelector<HTMLElement>(".workspace-heading h1")?.focus(); }} aria-label="收起处理选项" title="收起处理选项"><Icon icon="solar:alt-arrow-right-linear" width={16} /></button>
    </header>
    <div className="inspector-scroll">{children}</div>
    <footer className="inspector-footer">{footer}</footer>
  </aside>;
}

function WorkspaceFrame({ children, meta }: { children: React.ReactNode; meta: (typeof STEP_META)[WorkflowStep] }) {
  const inspectorOpen = useUiStore((state) => state.inspectorOpen);
  return <div className={`workspace-frame ${inspectorOpen ? "" : "inspector-hidden"}`}>
    <div className="workspace-heading"><h1 tabIndex={-1}>{meta.title}</h1><p>{meta.description}</p></div>
    <div className="min-h-0 flex-1 workspace-content">{children}</div>
  </div>;
}

function MetricTile({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`rounded-xl border border-border bg-surface p-3 ${wide ? "col-span-2" : ""}`}>
      <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-text-muted">{label}</p>
      <p className="mt-1 truncate text-[13px] font-semibold text-text-primary">{value}</p>
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

function ModelSummary({ model, fallback }: { model?: AsrModelInfo; fallback: string }) {
  const display = modelPresentation(model ?? { name: fallback.split(/[\\/]/).pop() || "未选择模型", id: fallback, downloaded: false });
  return <div className="current-model-summary">
    <span className="model-eyebrow">当前转录模型</span>
    <strong>{display.title}</strong>
    <p>{[display.variant, model ? display.status : "状态待确认"].filter(Boolean).join(" · ")}</p>
    <small>{display.hint}</small>
    <InspectorDisclosure title={<>模型详情</>}>
      <dl className="model-details">
        <dt>运行方式</dt><dd>{model?.type.toUpperCase() || "由当前引擎决定"}</dd>
        <dt>资源大小</dt><dd>{model?.size || "未知"}</dd>
        <dt>原始名称</dt><dd>{model?.name || fallback}</dd>
        <dt>文件位置或标识</dt><dd>{model?.path || model?.value || model?.id || fallback}</dd>
      </dl>
    </InspectorDisclosure>
  </div>;
}

function ModelChip({ active, downloading, model, onDownload, onSelect, progress }: {
  active: boolean; downloading: boolean; model: AsrModelInfo;
  onDownload: () => void; onSelect: () => void; progress?: number;
}) {
  const display = modelPresentation(model);
  const canSelect = model.downloaded || model.state === "on_demand";
  const unavailable = !canSelect && model.downloadable === false;
  return <button type="button" className="model-option" aria-pressed={active}
    disabled={downloading || unavailable} onClick={canSelect ? onSelect : onDownload}>
    <span className="model-option-heading"><strong>{display.title}</strong><span className="model-option-action">{downloading ? "下载中" : canSelect ? (active ? "✓ 当前" : "") : unavailable ? "不可下载" : "下载"}</span></span>
    <span className="model-option-hint">{display.hint}</span>
    <span className="model-option-meta">{[display.variant, model.size, downloading ? (progress == null ? "正在下载" : `正在下载 ${Math.round(progress)}%`) : display.status].filter(Boolean).join(" · ")}</span>
    {downloading && <span className={`model-download-track${progress == null ? " is-indeterminate" : ""}`} role="progressbar" aria-label={`${display.title}下载进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: progress == null ? "35%" : `${Math.max(0, Math.min(100, progress))}%` }} /></span>}
  </button>;
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
          {model.downloaded ? "可用" : downloading ? (progress == null ? "下载中" : `${progress}%`) : "下载"}
        </button>
      </div>
    </div>
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
