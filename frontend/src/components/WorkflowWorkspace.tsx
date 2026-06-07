"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import { useAppStore, WorkflowStep } from "@/store/appStore";
import {
  configApi,
  filesApi,
  subtitlesApi,
  tasksApi,
  transcribeApi,
  type AsrModelInfo,
  type FileInfo,
  type SubtitleSegment,
} from "@/lib/api";
import { formatDuration, formatSize, parseSrtTime } from "@/lib/format";
import { VideoPanel } from "@/components/VideoPanel";
import { SubtitlePanel } from "@/components/SubtitlePanel";

type TaskStarter = (
  type: "transcribe" | "subtitle",
  payload: Record<string, unknown>
) => Promise<void>;
type AppConfig = ReturnType<typeof useAppStore.getState>["config"];

interface WorkflowWorkspaceProps {
  startTask: TaskStarter;
  cancelTask: () => Promise<void>;
}

const STEP_META: Record<
  WorkflowStep,
  { eyebrow: string; title: string; description: string; icon: string }
> = {
  import: {
    eyebrow: "01 / Source",
    title: "导入与预检",
    description: "先确认素材、音轨、字幕和运行环境，避免任务开始后才发现配置缺口。",
    icon: "solar:inbox-in-bold-duotone",
  },
  transcribe: {
    eyebrow: "02 / Alignment",
    title: "语音转录工作台",
    description: "围绕 WhisperX、MLX、forced alignment 和实时字幕质量来组织转录流程。",
    icon: "solar:microphone-3-bold-duotone",
  },
  subtitle: {
    eyebrow: "03 / Bilingual Edit",
    title: "智能断句与翻译",
    description: "在时间轴、原文、译文和质量提示之间快速审校，减少吞词和错位。",
    icon: "solar:subtitle-bold-duotone",
  },
};

const SOURCE_LANGUAGES = [
  ["auto", "自动"],
  ["en", "英文"],
  ["zh", "中文"],
  ["ja", "日文"],
  ["ko", "韩文"],
  ["fr", "法语"],
  ["de", "德语"],
  ["es", "西班牙语"],
  ["pt", "葡萄牙语"],
  ["ru", "俄语"],
];

const TARGET_LANGUAGES = [
  ["chinese", "中文"],
  ["english", "英文"],
  ["japanese", "日文"],
  ["korean", "韩文"],
  ["french", "法语"],
  ["german", "德语"],
  ["spanish", "西班牙语"],
  ["portuguese", "葡萄牙语"],
  ["russian", "俄语"],
];

const TRANSLATORS = [
  ["llm", "LLM"],
  ["bing", "Bing"],
  ["google", "Google"],
  ["deeplx", "DeepLX"],
];

const ASR_ENGINES = [
  {
    id: "whisperx",
    name: "WhisperX",
    desc: "Apple Silicon 专门优化 · MLX + forced alignment",
    icon: "solar:bolt-bold-duotone",
  },
  {
    id: "whisper_cpp",
    name: "Whisper.cpp",
    desc: "本地 GGML · Metal/CPU 路径",
    icon: "solar:cpu-bolt-bold-duotone",
  },
  {
    id: "faster_whisper",
    name: "FasterWhisper",
    desc: "CTranslate2 · CUDA 设备更合适",
    icon: "solar:speedometer-bold-duotone",
  },
  {
    id: "whisper_api",
    name: "Whisper API",
    desc: "云端识别 · 不占用本机算力",
    icon: "solar:cloud-bold-duotone",
  },
];

const TRANSCRIBE_STAGES = [
  "提取音频",
  "音频增强",
  "Whisper 转录",
  "词级对齐",
  "时间轴修正",
  "写入字幕",
];

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
    setStep,
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
        const uploaded = await filesApi.upload(file);
        setVideoFile(uploaded.file_path);
        const info = await filesApi.info(uploaded.file_path);
        setFileInfo(info);
        useAppStore.getState().addToast("素材已导入", "success");
        setStep("transcribe");
      } catch (err) {
        useAppStore
          .getState()
          .setError(err instanceof Error ? err.message : "素材导入失败");
      } finally {
        setUploading(null);
      }
    },
    [setFileInfo, setStep, setVideoFile]
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
        setStep("subtitle");
      } catch (err) {
        useAppStore
          .getState()
          .setError(err instanceof Error ? err.message : "字幕导入失败");
      } finally {
        setUploading(null);
      }
    },
    [setStep, setSubtitleFile, setSubtitles]
  );

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
                  Media Intake
                </p>
                <h2 className="mt-2 max-w-[620px] text-[27px] font-semibold leading-tight text-text-primary">
                  把视频、音频和已有字幕先归档到一个清晰的任务入口。
                </h2>
              </div>
              <button
                onClick={() => mediaInputRef.current?.click()}
                className="inline-flex shrink-0 items-center gap-2 rounded-full bg-accent px-4 py-2 text-[13px] font-medium text-white shadow-md transition hover:bg-accent-hover disabled:opacity-50"
                disabled={uploading === "media"}
              >
                <Icon icon={uploading === "media" ? "solar:refresh-bold" : "solar:upload-bold"} className={uploading === "media" ? "animate-spin" : ""} width={17} />
                选择素材
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
                      : "支持 MP4、MOV、MKV、MP3、WAV。导入后会自动读取音轨、时长、分辨率和体积。"}
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
                  <h3 className="text-[14px] font-semibold text-text-primary">导入预检</h3>
                  <span className="rounded-full border border-border bg-background px-2.5 py-1 text-[11px] text-text-muted">
                    {fileInfo ? "已读取素材" : "等待素材"}
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
                      <p className="text-[13px] font-medium text-text-primary">已有字幕文件</p>
                      <p className="mt-1 text-[12px] text-text-muted">
                        {subtitleFile ? subtitleFile.split("/").pop() : "可直接进入智能断句/翻译"}
                      </p>
                    </div>
                    <button
                      onClick={() => subtitleInputRef.current?.click()}
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
          <Panel title="任务路径" icon="solar:map-point-wave-bold-duotone">
            <div className="space-y-4">
              <FlowStep index="1" title="导入" active done={!!videoFile || !!subtitleFile} />
              <FlowStep index="2" title="转录" active={!!videoFile} done={!!subtitleFile && subtitles.length > 0} />
              <FlowStep index="3" title="断句/翻译" active={!!subtitleFile} done={subtitles.some((s) => s.translated)} />
              <FlowStep index="4" title="导出" active={subtitles.length > 0} />
            </div>
          </Panel>

          <Panel title="音轨明细" icon="solar:soundwave-square-bold-duotone">
            {fileInfo?.audio_tracks.length ? (
              <div className="space-y-2">
                {fileInfo.audio_tracks.map((track) => (
                  <div key={track.index} className="rounded-xl border border-border bg-background p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[13px] font-medium text-text-primary">Audio {track.index + 1}</span>
                      <span className="rounded-full bg-surface px-2 py-0.5 text-[10px] text-text-muted">
                        {track.language || "und"}
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
              <EmptyState icon="solar:soundwave-bold-duotone" title="暂无音轨信息" minHeight="min-h-[220px]" />
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
    fileInfo,
    isProcessing,
    setConfig,
    setError,
    setIsProcessing,
    setStep,
    subtitles,
    taskMessage,
    taskProgress,
    taskStatus,
    videoFile,
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

  useEffect(() => {
    void transcribeApi.hardware().then(setHardware).catch(() => {});
    void transcribeApi.listModels().then(setModels).catch(() => {});
  }, []);

  const saveConfig = useCallback(
    async (key: string, value: string | number | boolean) => {
      const map: Record<string, keyof AppConfig> = {
        transcribe_model: "transcribeModel",
        source_language: "sourceLanguage",
        whisper_model_size: "whisperModelSize",
        whisperx_align_model: "whisperxAlignModel",
        whisperx_batch_size: "whisperxBatchSize",
        enable_audio_enhancement: "enableAudioEnhancement",
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

  const startTranscribe = useCallback(async () => {
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
  }, [config.sourceLanguage, config.transcribeModel, setError, setIsProcessing, setStep, startTask, videoFile]);

  const currentModels = useMemo(() => {
    if (config.transcribeModel === "whisper_cpp") {
      return models.filter((model) => model.category === "whisper_cpp");
    }
    if (config.transcribeModel === "whisperx") {
      return models.filter((model) => model.category === "whisperx" && model.type !== "alignment");
    }
    return [];
  }, [config.transcribeModel, models]);

  const alignmentModels = useMemo(
    () => models.filter((model) => model.category === "whisperx" && model.type === "alignment"),
    [models]
  );
  const selectedModel = currentModels.find((model) => model.id === config.whisperModelSize);
  const selectedAlignModel = alignmentModels.find(
    (model) => model.align_model === config.whisperxAlignModel || model.id === config.whisperxAlignModel
  );
  const quality = useMemo(() => analyzeSubtitleQuality(subtitles), [subtitles]);

  const downloadModel = useCallback(async (modelId: string) => {
    setDownloadingModel(modelId);
    setDownloadProgress((prev) => ({ ...prev, [modelId]: 0 }));
    try {
      const result = await transcribeApi.downloadModel(modelId);
      if (result.status === "already_exists") {
        setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloaded: true } : m)));
        setDownloadingModel(null);
        return;
      }
      if (result.task_id) {
        const timer = setInterval(async () => {
          try {
            const task = await tasksApi.get(result.task_id!);
            setDownloadProgress((prev) => ({ ...prev, [modelId]: task.progress }));
            if (task.status === "completed" || task.status === "failed") {
              clearInterval(timer);
              if (task.status === "completed") {
                setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloaded: true } : m)));
              } else {
                setError(task.error || "模型下载失败");
              }
              setDownloadingModel(null);
            }
          } catch {
            clearInterval(timer);
            setDownloadingModel(null);
          }
        }, 1000);
        return;
      }
      setDownloadingModel(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "模型下载失败");
      setDownloadingModel(null);
    }
  }, [setError]);

  return (
    <WorkspaceFrame meta={STEP_META.transcribe}>
      <div className="grid h-full min-h-0 grid-cols-[minmax(420px,1fr)_minmax(360px,0.72fr)] gap-5 max-xl:grid-cols-1">
        <section className="grid min-h-0 grid-rows-[minmax(300px,0.62fr)_minmax(250px,0.38fr)] gap-5">
          <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
            <VideoPanel />
          </div>
          <div className="grid min-h-0 grid-cols-[minmax(280px,0.7fr)_minmax(260px,0.3fr)] gap-5 max-lg:grid-cols-1">
            <Panel title="实时字幕预览" icon="solar:playlist-bold-duotone" fill>
              <LiveSubtitleList subtitles={subtitles} />
            </Panel>
            <Panel title="时间轴质量" icon="solar:shield-warning-bold-duotone" fill>
              <QualitySummary quality={quality} />
            </Panel>
          </div>
        </section>

        <aside className="flex min-h-0 flex-col gap-5 overflow-auto pr-1">
          <Panel title="ASR 引擎" icon="solar:tuning-square-2-bold-duotone">
            <div className="grid grid-cols-2 gap-2">
              {ASR_ENGINES.map((engine) => (
                <button
                  key={engine.id}
                  onClick={() => void saveConfig("transcribe_model", engine.id)}
                  className={`rounded-xl border p-3 text-left transition-all ${
                    config.transcribeModel === engine.id
                      ? "border-accent bg-accent-dim text-accent"
                      : "border-border bg-background text-text-secondary hover:border-border-active"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Icon icon={engine.icon} width={18} />
                    <span className="text-[13px] font-semibold">{engine.name}</span>
                  </div>
                  <p className="mt-1.5 text-[10px] leading-4 text-text-muted">{engine.desc}</p>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="模型与对齐" icon="solar:layers-bold-duotone">
            {config.transcribeModel === "whisper_api" ? (
              <EmptyState icon="solar:cloud-bold-duotone" title="云端模型在设置页配置" />
            ) : (
              <div className="space-y-4">
                <div>
                  <FieldLabel label="默认转录模型" value={selectedModel?.downloaded ? "可用" : "未下载"} />
                  <div className="mt-2 flex flex-wrap gap-2">
                    {currentModels.map((model) => (
                      <ModelChip
                        key={model.id}
                        model={model}
                        active={config.whisperModelSize === model.id}
                        downloading={downloadingModel === model.id}
                        progress={downloadProgress[model.id]}
                        onSelect={() => void saveConfig("whisper_model_size", model.id)}
                        onDownload={() => void downloadModel(model.id)}
                      />
                    ))}
                  </div>
                </div>

                {config.transcribeModel === "whisperx" && (
                  <div>
                    <FieldLabel
                      label="Forced alignment"
                      value={selectedAlignModel?.downloaded ? "词级时间轴可用" : "需要模型"}
                    />
                    <div className="mt-2 space-y-2">
                      {alignmentModels.map((model) => (
                        <ModelRow
                          key={model.id}
                          model={model}
                          active={
                            config.whisperxAlignModel === model.align_model ||
                            config.whisperxAlignModel === model.id
                          }
                          downloading={downloadingModel === model.id}
                          progress={downloadProgress[model.id]}
                          onSelect={() =>
                            void saveConfig("whisperx_align_model", model.align_model || model.id)
                          }
                          onDownload={() => void downloadModel(model.id)}
                        />
                      ))}
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
                  checked={config.enableAudioEnhancement}
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

          <TaskActionCard
            title="开始转录"
            description={videoFile ? videoFile.split("/").pop() || videoFile : "先导入视频或音频文件"}
            primaryLabel={taskStatus === "running" ? "转录中" : "开始转录"}
            disabled={!videoFile || isProcessing}
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
  const promptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const quality = useMemo(() => analyzeSubtitleQuality(subtitles), [subtitles]);
  const translatedCount = subtitles.filter((sub) => sub.translated.trim()).length;
  const completion = subtitles.length ? Math.round((translatedCount / subtitles.length) * 100) : 0;

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
          <SubtitlePanel showPrompt={false} showTranslateActions={false} />
        </section>

        <aside className="flex min-h-0 flex-col gap-5 overflow-auto pr-1">
          <Panel title="处理模式" icon="solar:magic-stick-3-bold-duotone">
            <div className="grid grid-cols-2 gap-2">
              <ToggleCard
                title="智能断句"
                desc="语义重组与字幕长度控制"
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
              <ToggleLine
                label="反思模式"
                checked={config.needReflect}
                onChange={(value) => void saveConfig("need_reflect", value)}
              />
            </div>
          </Panel>

          <Panel title="自定义 Prompt" icon="solar:pen-new-square-bold-duotone">
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
              placeholder="例如：保留汽车品牌、车型、技术名词；中文要自然，不要吞掉限定词。"
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
              <QualitySummary quality={quality} compact />
            </div>
          </Panel>

          <TaskActionCard
            title="开始处理字幕"
            description={subtitleFile ? subtitleFile.split("/").pop() || subtitleFile : "等待字幕文件"}
            primaryLabel={taskStatus === "running" ? "处理中" : config.needTranslate ? "开始翻译" : "开始断句"}
            disabled={!subtitleFile || isProcessing}
            progress={taskProgress}
            message={taskMessage}
            running={isProcessing}
            stages={["读取字幕", "智能断句", "优化原文", "生成上下文", "翻译", "重排保存"]}
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
      <div className="mb-3 flex shrink-0 items-center gap-2">
        <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent-dim text-accent">
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
      <span className="text-[11px] font-medium text-text-muted">{label}</span>
      <span className="rounded-full bg-background px-2 py-0.5 text-[10px] text-text-muted">{value}</span>
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
  return (
    <button
      onClick={model.downloaded ? onSelect : onDownload}
      className={`group flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] transition ${
        active
          ? "border-accent bg-accent-dim text-accent"
          : "border-border bg-background text-text-secondary hover:border-border-active"
      }`}
    >
      <span className="font-medium">{model.name || model.id.split("/").pop() || model.id}</span>
      <span className="text-text-muted">{model.size}</span>
      {model.downloaded ? (
        <Icon icon="solar:check-circle-bold" width={14} className="text-emerald-600" />
      ) : downloading ? (
        <span className="font-mono text-accent">{progress ?? 0}%</span>
      ) : (
        <span className="text-accent opacity-80 group-hover:opacity-100">下载</span>
      )}
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
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between rounded-xl border border-border bg-background p-3 text-left transition hover:border-border-active"
    >
      <span className="text-[13px] font-medium text-text-primary">{label}</span>
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
  stages: string[];
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

function LiveSubtitleList({ subtitles }: { subtitles: SubtitleSegment[] }) {
  const visible = subtitles.slice(-8);
  if (!visible.length) {
    return <EmptyState icon="solar:playlist-bold-duotone" title="转录开始后会显示实时字幕" />;
  }
  return (
    <div className="h-full min-h-0 overflow-auto pr-1">
      <div className="space-y-2">
        {visible.map((sub) => (
          <div key={sub.id} className="rounded-xl border border-border bg-background p-3">
            <div className="mb-1 flex items-center gap-2 font-mono text-[10px] text-text-muted">
              <span>{String(sub.id).padStart(3, "0")}</span>
              <span>{sub.start}</span>
              <span>→</span>
              <span>{sub.end}</span>
            </div>
            <p className="text-[12px] leading-5 text-text-primary">{sub.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function QualitySummary({ compact, quality }: { compact?: boolean; quality: SubtitleQuality }) {
  const items = [
    { label: "重叠", value: quality.overlaps.length, tone: quality.overlaps.length ? "bad" : "good" },
    { label: "过长", value: quality.longDurations.length, tone: quality.longDurations.length ? "warn" : "good" },
    { label: "紧贴", value: quality.tightGaps.length, tone: quality.tightGaps.length ? "warn" : "good" },
    { label: "空译文", value: quality.emptyTranslations, tone: quality.emptyTranslations ? "warn" : "good" },
  ];
  return (
    <div className="space-y-3">
      <div className={`grid ${compact ? "grid-cols-2" : "grid-cols-2"} gap-2`}>
        {items.map((item) => (
          <div key={item.label} className="rounded-xl border border-border bg-background p-3">
            <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-text-muted">{item.label}</p>
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
          </div>
        ))}
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

interface SubtitleQuality {
  overlaps: number[];
  longDurations: number[];
  tightGaps: number[];
  emptyTranslations: number;
}

function analyzeSubtitleQuality(subtitles: SubtitleSegment[]): SubtitleQuality {
  const overlaps: number[] = [];
  const longDurations: number[] = [];
  const tightGaps: number[] = [];
  let emptyTranslations = 0;

  subtitles.forEach((subtitle, index) => {
    const start = parseSrtTime(subtitle.start);
    const end = parseSrtTime(subtitle.end);
    const next = subtitles[index + 1];
    const duration = end - start;
    if (duration > 7.5) longDurations.push(subtitle.id);
    if (!subtitle.translated.trim()) emptyTranslations += 1;
    if (next) {
      const nextStart = parseSrtTime(next.start);
      const gap = nextStart - end;
      if (gap < -0.01) overlaps.push(subtitle.id);
      if (gap >= 0 && gap < 0.08) tightGaps.push(subtitle.id);
    }
  });

  return { overlaps, longDurations, tightGaps, emptyTranslations };
}
