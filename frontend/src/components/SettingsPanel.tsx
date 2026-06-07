"use client";

import { useState, useEffect, useRef } from "react";
import { useAppStore } from "@/store/appStore";
import { configApi, transcribeApi, tasksApi } from "@/lib/api";

const LLM_PROVIDERS = [
  { id: "openai", name: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  { id: "deepseek", name: "DeepSeek", baseUrl: "https://api.deepseek.com" },
  { id: "mimo", name: "小米 MiMo", baseUrl: "https://token-plan-cn.xiaomimimo.com/v1" },
  { id: "qwen", name: "通义千问", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
  { id: "zhipu", name: "智谱 GLM", baseUrl: "https://open.bigmodel.cn/api/paas/v4" },
  { id: "moonshot", name: "月之暗面", baseUrl: "https://api.moonshot.cn/v1" },
  { id: "baichuan", name: "百川智能", baseUrl: "https://api.baichuan-ai.com/v1" },
  { id: "yi", name: "零一万物", baseUrl: "https://api.lingyiwanwu.com/v1" },
  { id: "minimax", name: "MiniMax", baseUrl: "https://api.minimax.chat/v1" },
  { id: "siliconflow", name: "SiliconFlow", baseUrl: "https://api.siliconflow.cn/v1" },
  { id: "openrouter", name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1" },
  { id: "custom", name: "自定义", baseUrl: "" },
];

const WHISPER_CPP_MODELS = [
  { id: "tiny", name: "Tiny", size: "75MB", desc: "39M 参数，速度最快，适合快速预览，多语言能力弱" },
  { id: "base", name: "Base", size: "142MB", desc: "74M 参数，速度很快，英文表现尚可，其他语言一般" },
  { id: "small", name: "Small", size: "466MB", desc: "244M 参数，速度与质量较平衡，多语言能力明显提升" },
  { id: "medium", name: "Medium", size: "1.5GB", desc: "769M 参数，高准确率，中日韩等非英语语言推荐起步" },
  { id: "large-v1", name: "Large V1", size: "3.1GB", desc: "1550M 参数，初代旗舰，多语言表现优秀" },
  { id: "large-v2", name: "Large V2", size: "3.1GB", desc: "1550M 参数，训练数据更多，比 V1 更稳定可靠" },
  { id: "large-v3", name: "Large V3", size: "3.1GB", desc: "1550M 参数，最新架构，幻觉更少，推荐高质量转录" },
];

const FASTER_WHISPER_MODELS = [
  { id: "tiny", name: "Tiny", size: "75MB", desc: "CTranslate2 加速，速度极快，适合实时或低配设备" },
  { id: "base", name: "Base", size: "148MB", desc: "CTranslate2 加速，比原版快 4 倍，日常英文够用" },
  { id: "small", name: "Small", size: "496MB", desc: "CTranslate2 加速，性价比最高，多语言可用" },
  { id: "medium", name: "Medium", size: "1.5GB", desc: "CTranslate2 加速，非英语语言推荐，质量接近 Large" },
  { id: "large-v1", name: "Large V1", size: "3.1GB", desc: "CTranslate2 加速，初代旗舰量化版" },
  { id: "large-v2", name: "Large V2", size: "3.1GB", desc: "CTranslate2 加速，比 V1 训练更充分，更少出错" },
  { id: "large-v3", name: "Large V3", size: "3.1GB", desc: "CTranslate2 加速，最佳质量，专业字幕制作首选" },
  { id: "large-v3-turbo", name: "Large V3 Turbo", size: "1.7GB", desc: "V3 蒸馏版，速度提升 8 倍，质量略低于 V3" },
];

const MLX_WHISPER_MODELS = [
  { id: "/Users/guwenhan/Desktop/YouTube/model/whisper-large-v3-fp16", name: "MLX Large V3 FP16", size: "3.1GB", desc: "Apple Silicon 专用 MLX 模型，WhisperX 默认使用", localOnly: true },
  ...FASTER_WHISPER_MODELS,
];

const WHISPERX_ALIGNMENT_MODELS = [
  {
    id: "whisperx-align-en-large",
    name: "English Large LV60K",
    size: "1.18GB",
    desc: "英文 forced alignment 模型，用于 WhisperX 词级时间轴",
    alignModel: "WAV2VEC2_ASR_LARGE_LV60K_960H",
  },
];

export function SettingsPanel() {
  const { setActiveView } = useAppStore();
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [hwInfo, setHwInfo] = useState<{ chip: string; device: string; n_threads: number; compute_type: string; gpu: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("deepseek");
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadedModels, setDownloadedModels] = useState<Set<string>>(new Set());
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const downloadPollsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  useEffect(() => () => {
    downloadPollsRef.current.forEach((id) => clearInterval(id));
    downloadPollsRef.current.clear();
  }, []);
  const [detectedModels, setDetectedModels] = useState<string[] | null>(null);
  const [detectingModels, setDetectingModels] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [testingLlm, setTestingLlm] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<{ ok: boolean; model?: string; error?: string } | null>(null);
  const [testingWhisper, setTestingWhisper] = useState(false);
  const [whisperTestResult, setWhisperTestResult] = useState<{ ok: boolean; error?: string } | null>(null);
  const [whisperModels, setWhisperModels] = useState<string[] | null>(null);
  const [detectingWhisperModels, setDetectingWhisperModels] = useState(false);
  const [whisperModelError, setWhisperModelError] = useState<string | null>(null);
  const [customUrls, setCustomUrls] = useState<Record<string, string>>({});

  useEffect(() => {
    configApi.get().then((data) => {
      setSettings(data);
      // Sync settings used by workflow pages.
      const storeUpdates: Record<string, unknown> = {};
      if (data.need_reflect !== undefined) storeUpdates.needReflect = !!data.need_reflect;
      if (data.custom_prompt !== undefined) storeUpdates.customPrompt = data.custom_prompt as string;
      if (data.transcribe_model !== undefined) storeUpdates.transcribeModel = data.transcribe_model as string;
      if (data.source_language !== undefined) storeUpdates.sourceLanguage = data.source_language as string;
      if (data.target_language !== undefined) storeUpdates.targetLanguage = data.target_language as string;
      if (data.translator !== undefined) storeUpdates.translator = data.translator as string;
      if (data.llm_model !== undefined) storeUpdates.llmModel = data.llm_model as string;
      if (data.whisper_model_size !== undefined) storeUpdates.whisperModelSize = data.whisper_model_size as string;
      if (data.whisperx_align_model !== undefined) storeUpdates.whisperxAlignModel = data.whisperx_align_model as string;
      if (data.whisperx_batch_size !== undefined) storeUpdates.whisperxBatchSize = Number(data.whisperx_batch_size || 4);
      if (data.enable_audio_enhancement !== undefined) storeUpdates.enableAudioEnhancement = !!data.enable_audio_enhancement;
      if (Object.keys(storeUpdates).length > 0) useAppStore.getState().setConfig(storeUpdates);
      // Detect current provider from base URL
      const url = (data.llm_base_url as string) || "";
      const provider = LLM_PROVIDERS.find((p) => p.baseUrl && url.startsWith(p.baseUrl));
      if (provider) setSelectedProvider(provider.id);
      setLoading(false);
    }).catch(() => setLoading(false));

    // Detect hardware
    transcribeApi.hardware().then((hw) => setHwInfo(hw)).catch(() => {});

    // Check which whisper models are already downloaded
    transcribeApi.listModels().then((models) => {
      const downloaded = new Set(models.filter((m) => m.downloaded).map((m) => m.id));
      setDownloadedModels(downloaded);
    }).catch(() => {});
  }, []);

  const handleSave = async (key: string, value: unknown) => {
    setSaving(true);
    try {
      await configApi.update(key, value);
      setSettings((prev) => ({ ...prev, [key]: value }));
      const configKeyMap: Record<string, string> = {
        transcribe_model: "transcribeModel",
        source_language: "sourceLanguage",
        target_language: "targetLanguage",
        translator: "translator",
        llm_model: "llmModel",
        need_reflect: "needReflect",
        custom_prompt: "customPrompt",
        whisper_model_size: "whisperModelSize",
        whisperx_align_model: "whisperxAlignModel",
        whisperx_batch_size: "whisperxBatchSize",
        enable_audio_enhancement: "enableAudioEnhancement",
      };
      const mappedKey = configKeyMap[key];
      if (mappedKey) {
        useAppStore.getState().setConfig({ [mappedKey]: value });
      }
    }
    catch (err) { useAppStore.getState().setError(err instanceof Error ? err.message : "Save failed"); }
    finally { setSaving(false); }
  };

  const handleProviderChange = (providerId: string) => {
    const provider = LLM_PROVIDERS.find((p) => p.id === providerId);
    if (!provider) return;

    // Save current URL for the outgoing provider
    const currentUrl = (settings.llm_base_url as string) || "";
    if (selectedProvider && currentUrl) {
      setCustomUrls((prev) => ({ ...prev, [selectedProvider]: currentUrl }));
    }

    setSelectedProvider(providerId);

    // Restore saved custom URL for this provider, or use its default
    const savedUrl = customUrls[providerId];
    if (savedUrl) {
      handleSave("llm_base_url", savedUrl);
    } else if (provider.baseUrl) {
      handleSave("llm_base_url", provider.baseUrl);
    }

    setDetectedModels(null);
    setDetectError(null);
  };

  const handleDetectModels = async () => {
    setDetectingModels(true);
    setDetectError(null);
    try {
      const result = await configApi.fetchModels();
      if (result.error) {
        setDetectError(result.error);
        setDetectedModels(null);
      } else {
        setDetectedModels(result.models);
      }
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : "检测失败");
      setDetectedModels(null);
    } finally {
      setDetectingModels(false);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    setDownloadingModel(modelId);
    setDownloadProgress((prev) => ({ ...prev, [modelId]: 0 }));
    try {
      const result = await transcribeApi.downloadModel(modelId);
      if (result.status === "already_exists") {
        setDownloadedModels((prev) => new Set([...prev, modelId]));
        setDownloadingModel(null);
        return;
      }
      if (result.task_id) {
        // Poll for progress
        const pollId = setInterval(async () => {
          try {
            const task = await tasksApi.get(result.task_id!);
            setDownloadProgress((prev) => ({ ...prev, [modelId]: task.progress }));
            if (task.status === "completed") {
              clearInterval(downloadPollsRef.current.get(modelId)!);
              downloadPollsRef.current.delete(modelId);
              setDownloadedModels((prev) => new Set([...prev, modelId]));
              setDownloadingModel(null);
              setDownloadProgress((prev) => { const n = { ...prev }; delete n[modelId]; return n; });
            } else if (task.status === "failed") {
              clearInterval(downloadPollsRef.current.get(modelId)!);
              downloadPollsRef.current.delete(modelId);
              useAppStore.getState().setError(task.error || "下载失败");
              setDownloadingModel(null);
              setDownloadProgress((prev) => { const n = { ...prev }; delete n[modelId]; return n; });
            }
          } catch {
            clearInterval(downloadPollsRef.current.get(modelId)!);
            downloadPollsRef.current.delete(modelId);
            setDownloadingModel(null);
          }
        }, 1000);
        downloadPollsRef.current.set(modelId, pollId);
      }
    } catch (err) {
      useAppStore.getState().setError(err instanceof Error ? err.message : "下载失败");
      setDownloadingModel(null);
    }
  };

  if (loading) return <div className="flex items-center justify-center h-full"><div className="w-6 h-6 rounded-full border-2 border-accent/20 border-t-accent animate-spin" /></div>;

  const currentProvider = LLM_PROVIDERS.find((p) => p.id === selectedProvider);

  return (
    <div className="flex flex-col h-full bg-background overflow-auto">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border sticky top-0 bg-surface z-10">
        <h2 className="text-[13px] font-medium text-text-primary">设置</h2>
        <button onClick={() => setActiveView("workflow")} className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>
      <div className="p-5 space-y-6 max-w-2xl pb-20">

        {/* LLM Configuration */}
        <SettingsSection title="LLM 配置" description="用于字幕优化和智能翻译">
          <SettingsField label="服务商">
            <div className="grid grid-cols-4 gap-2">
              {LLM_PROVIDERS.map((p) => (
                <button key={p.id} onClick={() => handleProviderChange(p.id)}
                  className={`py-2 px-2 rounded-lg border text-[12px] transition-all text-center btn-press ${
                    selectedProvider === p.id
                      ? "border-accent bg-accent-dim text-accent font-medium"
                      : "border-border text-text-secondary hover:border-[rgba(0,0,0,0.12)]"
                  }`}>
                  {p.name}
                </button>
              ))}
            </div>
          </SettingsField>
          <SettingsField label="Base URL">
            <input type="text" value={(settings.llm_base_url as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, llm_base_url: e.target.value }))}
              onBlur={(e) => handleSave("llm_base_url", e.target.value)}
              placeholder={currentProvider?.baseUrl || "https://api.example.com/v1"} className="input-field" />
          </SettingsField>
          <SettingsField label="API Key">
            <input type="password" value={(settings.llm_api_key as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, llm_api_key: e.target.value }))}
              onBlur={(e) => handleSave("llm_api_key", e.target.value)}
              placeholder="sk-..." className="input-field" />
          </SettingsField>
          <SettingsField label="模型" description="点击「检测模型」从服务商获取可用模型列表">
            <div className="flex items-center gap-2 mb-2">
              <button onClick={handleDetectModels} disabled={detectingModels || !settings.llm_base_url}
                className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] rounded-md bg-accent-dim text-accent hover:bg-accent/15 transition-all font-medium disabled:opacity-40 disabled:cursor-not-allowed btn-press">
                {detectingModels ? (
                  <>
                    <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>
                    检测中...
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
                    检测模型
                  </>
                )}
              </button>
              {(detectedModels !== null || detectError) && (
                <button onClick={() => { setDetectedModels(null); setDetectError(null); }}
                  className="text-[10px] text-text-muted hover:text-text-secondary transition-colors">
                  清除
                </button>
              )}
            </div>

            {detectError && (
              <div className="mb-2 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-[11px] text-red-600">
                {detectError}
              </div>
            )}

            {detectedModels !== null && (
              detectedModels.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 max-h-48 overflow-auto">
                  {detectedModels.map((m) => (
                    <button key={m} onClick={() => handleSave("llm_model", m)}
                      className={`px-2.5 py-1 rounded-lg border text-[11px] transition-all ${
                        (settings.llm_model as string) === m
                          ? "border-accent bg-accent-dim text-accent font-medium"
                          : "border-border text-text-secondary hover:border-[rgba(0,0,0,0.12)]"
                      }`}>
                      {m}
                    </button>
                  ))}
                </div>
              ) : (
                !detectError && <p className="text-[11px] text-text-muted">未发现可用模型</p>
              )
            )}

            <input type="text" value={(settings.llm_model as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, llm_model: e.target.value }))}
              onBlur={(e) => handleSave("llm_model", e.target.value)}
              placeholder="输入或选择模型名称" className="input-field mt-2" />

            <div className="flex items-center gap-2 mt-2">
              <button onClick={async () => {
                setTestingLlm(true); setLlmTestResult(null);
                try { const r = await configApi.testLlm(); setLlmTestResult(r); } catch (err) { setLlmTestResult({ ok: false, error: err instanceof Error ? err.message : "测试失败" }); } finally { setTestingLlm(false); }
              }} disabled={testingLlm || !settings.llm_api_key || !settings.llm_model}
                className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] rounded-md bg-accent-dim text-accent hover:bg-accent/15 transition-all font-medium disabled:opacity-40 disabled:cursor-not-allowed btn-press">
                {testingLlm ? (<><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>测试中...</>) : "测试连接"}
              </button>
              {llmTestResult && (
                <span className={`text-[11px] ${llmTestResult.ok ? "text-emerald-600" : "text-red-500"}`}>
                  {llmTestResult.ok ? `连接成功 (${llmTestResult.model})` : `失败: ${llmTestResult.error}`}
                </span>
              )}
            </div>
          </SettingsField>
        </SettingsSection>

        {/* ASR Configuration */}
        <SettingsSection title="语音识别" description="选择 ASR 引擎和模型">
          <SettingsField label="引擎">
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: "whisperx", name: "WhisperX", desc: "Apple Silicon 推荐 · MLX 加速 + forced alignment" },
                { id: "whisper_cpp", name: "Whisper.cpp", desc: "macOS 推荐 · Metal GPU 加速" },
                { id: "faster_whisper", name: "FasterWhisper", desc: "NVIDIA GPU 推荐 · CUDA 加速" },
                { id: "whisper_api", name: "Whisper API", desc: "云端接口，无需本地算力" },
              ].map((e) => (
                <button key={e.id} onClick={async () => { await handleSave("transcribe_model", e.id); useAppStore.getState().setConfig({ transcribeModel: e.id }); }}
                  className={`p-3 rounded-xl border-2 text-left transition-all ${
                    (settings.transcribe_model as string) === e.id
                      ? "border-accent bg-accent-dim"
                      : "border-border hover:border-[rgba(0,0,0,0.12)]"
                  }`}>
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-medium text-text-primary">{e.name}</span>
                  </div>
                  <span className="text-[10px] text-text-muted">{e.desc}</span>
                </button>
              ))}
            </div>
            {(settings.transcribe_model as string) === "faster_whisper" && (
              <p className="text-[10px] text-amber-600 mt-1.5">需要 NVIDIA GPU 和 faster-whisper-xxl，macOS 上仅支持 CPU 模式（无 GPU 加速）</p>
            )}
            {(settings.transcribe_model as string) === "whisperx" && (
              <p className="text-[10px] text-emerald-600 mt-1.5">WhisperX 使用 Apple Silicon 专门优化的 MLX Whisper 转录，并通过 forced alignment 生成词级时间轴</p>
            )}
            {(settings.transcribe_model as string) === "whisper_cpp" && (
              <p className="text-[10px] text-emerald-600 mt-1.5">Apple Silicon 原生 Metal 加速，NVIDIA GPU 上使用 CPU + int8 量化</p>
            )}
          </SettingsField>

          {/* Hardware detection info */}
          {hwInfo && hwInfo.chip !== "Unknown" && (
            <div className="rounded-xl border border-accent/20 bg-gradient-to-br from-accent-dim to-[rgba(37,99,235,0.04)] p-4">
              <div className="flex items-center gap-2 mb-2.5">
                <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                <span className="text-[12px] font-medium text-text-primary">硬件加速</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600 font-medium">已优化</span>
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="text-text-muted">芯片</span>
                  <span className="text-text-primary font-medium">{hwInfo.chip}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-text-muted">加速</span>
                  <span className="text-text-primary font-medium">{hwInfo.gpu}</span>
                </div>
              </div>
              <p className="text-[10px] text-text-muted mt-2">系统已自动检测硬件并应用最优配置，无需手动调整。</p>
            </div>
          )}

          {/* Whisper API config */}
          {(settings.transcribe_model as string) === "whisper_api" && (
            <>
            <SettingsField label="API Base URL">
              <input type="text" value={(settings.whisper_base_url as string) || ""}
                onChange={(e) => setSettings((prev) => ({ ...prev, whisper_base_url: e.target.value }))}
                onBlur={(e) => handleSave("whisper_base_url", e.target.value)}
                placeholder="https://api.openai.com/v1" className="input-field" />
            </SettingsField>
            <SettingsField label="API Key">
              <input type="password" value={(settings.whisper_api_key as string) || ""}
                onChange={(e) => setSettings((prev) => ({ ...prev, whisper_api_key: e.target.value }))}
                onBlur={(e) => handleSave("whisper_api_key", e.target.value)}
                placeholder="sk-..." className="input-field" />
            </SettingsField>
            <SettingsField label="模型名称">
              <div className="flex gap-2">
                <input type="text" value={(settings.whisper_api_model as string) || "whisper-1"}
                  onChange={(e) => setSettings((prev) => ({ ...prev, whisper_api_model: e.target.value }))}
                  onBlur={(e) => handleSave("whisper_api_model", e.target.value)}
                  placeholder="whisper-1" className="input-field flex-1" />
                <button onClick={async () => {
                  setDetectingWhisperModels(true); setWhisperModelError(null); setWhisperModels(null);
                  try {
                    const r = await configApi.fetchWhisperModels();
                    if (r.models && r.models.length > 0) {
                      setWhisperModels(r.models);
                    } else {
                      setWhisperModelError(r.error || "未发现可用模型");
                    }
                  } catch (err) {
                    setWhisperModelError(err instanceof Error ? err.message : "检测失败");
                  } finally {
                    setDetectingWhisperModels(false);
                  }
                }} disabled={detectingWhisperModels || !settings.whisper_base_url}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] rounded-md bg-accent-dim text-accent hover:bg-accent/15 transition-all font-medium disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap btn-press">
                  {detectingWhisperModels ? (<><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>检测中...</>) : "检测模型"}
                </button>
              </div>
              {whisperModels && whisperModels.length > 0 && (
                <div className="mt-2 p-2 rounded-lg border border-border bg-[rgba(0,0,0,0.01)] max-h-40 overflow-y-auto">
                  <p className="text-[10px] text-text-muted mb-1.5">点击选择模型：</p>
                  <div className="flex flex-wrap gap-1.5">
                    {whisperModels.map((m) => (
                      <button key={m} onClick={() => {
                        setSettings((prev) => ({ ...prev, whisper_api_model: m }));
                        handleSave("whisper_api_model", m);
                        setWhisperModels(null);
                      }}
                        className={`px-2 py-1 text-[11px] rounded-md border transition-all ${
                          (settings.whisper_api_model as string) === m
                            ? "border-accent bg-accent-dim text-accent"
                            : "border-border text-text-secondary hover:border-accent hover:text-accent"
                        }`}>
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {whisperModelError && (
                <p className="text-[11px] text-red-500 mt-1.5">{whisperModelError}</p>
              )}
            </SettingsField>
            <div className="flex items-center gap-2">
              <button onClick={async () => {
                setTestingWhisper(true); setWhisperTestResult(null);
                try { const r = await configApi.testWhisper(); setWhisperTestResult(r); } catch (err) { setWhisperTestResult({ ok: false, error: err instanceof Error ? err.message : "测试失败" }); } finally { setTestingWhisper(false); }
              }} disabled={testingWhisper}
                className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] rounded-md bg-accent-dim text-accent hover:bg-accent/15 transition-all font-medium disabled:opacity-40 disabled:cursor-not-allowed btn-press">
                {testingWhisper ? (<><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>测试中...</>) : "测试连接"}
              </button>
              {whisperTestResult && (
                <span className={`text-[11px] ${whisperTestResult.ok ? "text-emerald-600" : "text-red-500"}`}>
                  {whisperTestResult.ok ? "连接成功" : `失败: ${whisperTestResult.error}`}
                </span>
              )}
            </div>
            </>
          )}

          {/* Whisper model download */}
          {((settings.transcribe_model as string) === "whisper_cpp" || (settings.transcribe_model as string) === "faster_whisper" || (settings.transcribe_model as string) === "whisperx") && (
            <>
            {(settings.transcribe_model as string) === "whisper_cpp" && (
              <SettingsField label="Whisper.cpp 程序路径" description="填写 whisper-cli 可执行文件路径；模型文件和程序文件必须同时存在">
                <input type="text" value={(settings.whisper_cpp_path as string) || ""}
                  onChange={(e) => setSettings((prev) => ({ ...prev, whisper_cpp_path: e.target.value }))}
                  onBlur={(e) => handleSave("whisper_cpp_path", e.target.value)}
                  placeholder="/opt/homebrew/bin/whisper-cli" className="input-field" />
              </SettingsField>
            )}
            <SettingsField label="模型存储目录" description="留空则使用默认目录（~/SubForge/models）">
              <input type="text" value={(settings.whisper_model_dir as string) || ""}
                onChange={(e) => setSettings((prev) => ({ ...prev, whisper_model_dir: e.target.value }))}
                onBlur={(e) => handleSave("whisper_model_dir", e.target.value)}
                placeholder="~/SubForge/models" className="input-field" />
            </SettingsField>
            {(settings.transcribe_model as string) === "whisperx" && (
              <SettingsField label="Forced alignment 模型" description="英文推荐 WAV2VEC2_ASR_LARGE_LV60K_960H；留空则按语言自动选择">
                <input type="text" value={(settings.whisperx_align_model as string) || "WAV2VEC2_ASR_LARGE_LV60K_960H"}
                  onChange={(e) => setSettings((prev) => ({ ...prev, whisperx_align_model: e.target.value }))}
                  onBlur={(e) => handleSave("whisperx_align_model", e.target.value)}
                  placeholder="WAV2VEC2_ASR_LARGE_LV60K_960H" className="input-field" />
              </SettingsField>
            )}
            <SettingsField label="模型下载" description={settings.transcribe_model === "whisper_cpp" ? "Whisper.cpp GGML 模型" : settings.transcribe_model === "whisperx" ? "WhisperX 的 MLX 转录模型和 forced alignment 模型" : "FasterWhisper 模型（HuggingFace）"}>
              <div className="space-y-2">
                {(settings.transcribe_model === "whisper_cpp" ? WHISPER_CPP_MODELS : settings.transcribe_model === "whisperx" ? MLX_WHISPER_MODELS : FASTER_WHISPER_MODELS).map((m) => (
                  <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border border-border bg-[rgba(0,0,0,0.01)]">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[12px] font-medium text-text-primary">{m.name}</span>
                        <span className="text-[10px] text-text-muted font-mono">{m.size}</span>
                        {Boolean((m as { localOnly?: boolean }).localOnly) && <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600">本地</span>}
                      </div>
                      <span className="text-[10px] text-text-muted">{m.desc}</span>
                    </div>
                    <button onClick={() => handleDownloadModel(m.id)} disabled={downloadingModel === m.id || downloadedModels.has(m.id) || Boolean((m as { localOnly?: boolean }).localOnly)}
                      className={`px-3 py-1.5 text-[11px] rounded-md transition-all font-medium disabled:opacity-50 ${
                        downloadedModels.has(m.id)
                          ? "bg-emerald-50 text-emerald-600 cursor-default"
                          : Boolean((m as { localOnly?: boolean }).localOnly)
                          ? "bg-emerald-50 text-emerald-600 cursor-default"
                          : "bg-accent-dim text-accent hover:bg-accent/15"
                      }`}>
                      {downloadedModels.has(m.id) || Boolean((m as { localOnly?: boolean }).localOnly) ? (
                        <span className="flex items-center gap-1.5">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>
                          {Boolean((m as { localOnly?: boolean }).localOnly) && !downloadedModels.has(m.id) ? "手动配置" : "已下载"}
                        </span>
                      ) : downloadingModel === m.id ? (
                        <span className="flex items-center gap-1.5">
                          <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>
                          {downloadProgress[m.id] != null ? `${downloadProgress[m.id]}%` : "下载中"}
                        </span>
                      ) : "下载"}
                    </button>
                  </div>
                ))}
                {(settings.transcribe_model as string) === "whisperx" && (
                  <div className="pt-2 mt-2 border-t border-border space-y-2">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider">Forced alignment</p>
                    {WHISPERX_ALIGNMENT_MODELS.map((m) => (
                      <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border border-border bg-[rgba(0,0,0,0.01)]">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-[12px] font-medium text-text-primary">{m.name}</span>
                            <span className="text-[10px] text-text-muted font-mono">{m.size}</span>
                          </div>
                          <span className="text-[10px] text-text-muted">{m.desc}</span>
                        </div>
                        <button onClick={async () => { await handleSave("whisperx_align_model", m.alignModel); await handleDownloadModel(m.id); }}
                          disabled={downloadingModel === m.id || downloadedModels.has(m.id)}
                          className={`px-3 py-1.5 text-[11px] rounded-md transition-all font-medium disabled:opacity-50 ${
                            downloadedModels.has(m.id)
                              ? "bg-emerald-50 text-emerald-600 cursor-default"
                              : "bg-accent-dim text-accent hover:bg-accent/15"
                          }`}>
                          {downloadedModels.has(m.id) ? (
                            <span className="flex items-center gap-1.5">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>
                              已下载
                            </span>
                          ) : downloadingModel === m.id ? (
                            <span className="flex items-center gap-1.5">
                              <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>
                              {downloadProgress[m.id] != null ? `${downloadProgress[m.id]}%` : "下载中"}
                            </span>
                          ) : "下载"}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </SettingsField>
            </>
          )}

          {/* Whisper model selection */}
          {((settings.transcribe_model as string) === "whisper_cpp" || (settings.transcribe_model as string) === "faster_whisper" || (settings.transcribe_model as string) === "whisperx") && (
            <SettingsField label="默认模型" description="选择转录时使用的模型大小">
              <div className="flex flex-wrap gap-1.5">
                {(settings.transcribe_model === "whisper_cpp" ? WHISPER_CPP_MODELS : settings.transcribe_model === "whisperx" ? MLX_WHISPER_MODELS : FASTER_WHISPER_MODELS).map((m) => (
                  <button key={m.id} onClick={() => handleSave("whisper_model_size", m.id)}
                    className={`px-2.5 py-1.5 rounded-lg border text-[11px] transition-all flex items-center gap-1.5 ${
                      ((settings.whisper_model_size as string) || "base") === m.id
                        ? "border-accent bg-accent-dim text-accent font-medium"
                        : "border-border text-text-secondary hover:border-[rgba(0,0,0,0.12)]"
                    }`}>
                    {m.name}
                    <span className="text-[9px] text-text-muted">{m.size}</span>
                    {downloadedModels.has(m.id) && (
                      <svg className="w-3 h-3 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>
                    )}
                  </button>
                ))}
              </div>
            </SettingsField>
          )}

          {(settings.transcribe_model as string) === "faster_whisper" && (
            <SettingsField label="人声分离" description="转录前使用 MDX-Net 分离人声和背景音乐，提升嘈杂环境识别准确率">
              <div className="flex items-center gap-3">
                <button onClick={() => handleSave("ff_mdx_kim2", !(settings.ff_mdx_kim2))}
                  className={`relative w-10 h-[22px] rounded-full transition-all duration-200 ${
                    settings.ff_mdx_kim2 ? "bg-accent" : "bg-[rgba(0,0,0,0.1)]"
                  }`}>
                  <div className={`absolute top-[3px] w-4 h-4 rounded-full bg-white shadow-sm transition-all duration-200 ${
                    settings.ff_mdx_kim2 ? "left-[22px]" : "left-[3px]"
                  }`} />
                </button>
                <span className="text-[11px] text-text-muted">{settings.ff_mdx_kim2 ? "已开启" : "关闭"}</span>
              </div>
            </SettingsField>
          )}

          <SettingsField label="默认源语言">
            <select value={(settings.source_language as string) || "auto"} onChange={(e) => handleSave("source_language", e.target.value)} className="input-field">
              <option value="auto">自动检测</option>
              <option value="zh">中文</option><option value="en">英文</option><option value="ja">日文</option><option value="ko">韩文</option>
              <option value="fr">法语</option><option value="de">德语</option><option value="es">西班牙语</option><option value="pt">葡萄牙语</option>
              <option value="ru">俄语</option><option value="ar">阿拉伯语</option><option value="hi">印地语</option><option value="th">泰语</option>
              <option value="vi">越南语</option><option value="id">印尼语</option><option value="ms">马来语</option><option value="tr">土耳其语</option>
              <option value="pl">波兰语</option><option value="nl">荷兰语</option><option value="sv">瑞典语</option><option value="da">丹麦语</option>
              <option value="fi">芬兰语</option><option value="nb">挪威语</option><option value="cs">捷克语</option><option value="el">希腊语</option>
              <option value="he">希伯来语</option><option value="ro">罗马尼亚语</option><option value="hu">匈牙利语</option><option value="uk">乌克兰语</option>
              <option value="bn">孟加拉语</option><option value="tl">菲律宾语</option><option value="ta">泰米尔语</option><option value="ur">乌尔都语</option>
            </select>
          </SettingsField>
        </SettingsSection>

        {/* Translation */}
        <SettingsSection title="翻译服务">
          <SettingsField label="默认翻译服务">
            <select value={(settings.translator as string) || "bing"} onChange={(e) => handleSave("translator", e.target.value)} className="input-field">
              <option value="bing">Bing 翻译 (免费)</option><option value="google">Google 翻译</option><option value="deeplx">DeepLX</option><option value="llm">LLM 翻译</option>
            </select>
          </SettingsField>
          <SettingsField label="默认目标语言">
            <select value={(settings.target_language as string) || "english"} onChange={(e) => handleSave("target_language", e.target.value)} className="input-field">
              <option value="chinese">中文</option><option value="english">英文</option><option value="japanese">日文</option><option value="korean">韩文</option>
              <option value="french">法语</option><option value="german">德语</option><option value="spanish">西班牙语</option><option value="portuguese">葡萄牙语</option>
              <option value="russian">俄语</option><option value="arabic">阿拉伯语</option><option value="hindi">印地语</option><option value="thai">泰语</option>
              <option value="vietnamese">越南语</option><option value="indonesian">印尼语</option><option value="turkish">土耳其语</option><option value="polish">波兰语</option>
              <option value="dutch">荷兰语</option><option value="swedish">瑞典语</option><option value="ukrainian">乌克兰语</option>
            </select>
          </SettingsField>
          <SettingsField label="吴恩达反思模式" description="翻译→批评→重写，提升自然度；仅 LLM 翻译有效，会增加 API 调用">
            <button onClick={() => {
              const newVal = !settings.need_reflect;
              handleSave("need_reflect", newVal);
              useAppStore.getState().setConfig({ needReflect: newVal });
            }}
              className={`flex items-center gap-2 group ${false ? "opacity-40 cursor-not-allowed" : ""}`}>
              <div className={`w-7 h-[15px] rounded-full transition-all duration-200 relative flex items-center ${
                settings.need_reflect ? "bg-accent/20" : "bg-[rgba(0,0,0,0.06)]"
              }`}>
                <div className={`absolute w-[11px] h-[11px] rounded-full transition-all duration-200 shadow-sm ${
                  settings.need_reflect ? "left-[13px] bg-accent" : "left-[2px] bg-text-muted"
                }`} />
              </div>
              <span className="text-[12px] text-text-muted group-hover:text-text-secondary transition-colors">
                {settings.need_reflect ? "已开启" : "已关闭"}
              </span>
            </button>
          </SettingsField>
        </SettingsSection>

        {/* LLM Translation Fine-tuning */}
        <SettingsSection title="字幕处理调节" description="控制智能断句和翻译的细节参数">
          <SettingsField label="中文字幕每行最多字数" description="每段字幕的长度上限，断句优先按语义拆分">
            <div className="flex items-center gap-3">
              <input type="range" min={10} max={40} value={(settings.max_word_count_cjk as number) || 25}
                onChange={(e) => handleSave("max_word_count_cjk", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="text-[12px] text-text-primary font-mono w-8 text-right">{(settings.max_word_count_cjk as number) || 25}</span>
            </div>
          </SettingsField>
          <SettingsField label="英文字幕每行最多字数" description="每段字幕的单词数上限">
            <div className="flex items-center gap-3">
              <input type="range" min={10} max={60} value={(settings.max_word_count_english as number) || 18}
                onChange={(e) => handleSave("max_word_count_english", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="text-[12px] text-text-primary font-mono w-8 text-right">{(settings.max_word_count_english as number) || 18}</span>
            </div>
          </SettingsField>
          <SettingsField label="并发线程数" description="同时处理的LLM请求数量（过高可能触发限流）">
            <div className="flex items-center gap-3">
              <input type="range" min={1} max={20} value={(settings.thread_num as number) || 3}
                onChange={(e) => handleSave("thread_num", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="text-[12px] text-text-primary font-mono w-8 text-right">{(settings.thread_num as number) || 3}</span>
            </div>
          </SettingsField>
          <SettingsField label="翻译批处理大小" description="每次翻译的当前字幕条数；系统会自动附带前后文，过大可能降低一致性">
            <div className="flex items-center gap-3">
              <input type="range" min={1} max={50} value={(settings.batch_size as number) || 10}
                onChange={(e) => handleSave("batch_size", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="text-[12px] text-text-primary font-mono w-8 text-right">{(settings.batch_size as number) || 10}</span>
            </div>
          </SettingsField>
          {/* Custom prompt moved to subtitle page */}
        </SettingsSection>

        {/* Work directory */}
        <SettingsSection title="工作目录">
          <SettingsField label="路径" description="视频和字幕文件的默认保存位置">
            <input type="text" value={(settings.work_dir as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, work_dir: e.target.value }))}
              onBlur={(e) => handleSave("work_dir", e.target.value)}
              placeholder="默认: ~/SubForge/work-dir" className="input-field" />
          </SettingsField>
        </SettingsSection>

        {saving && <div className="text-[11px] text-accent flex items-center gap-2"><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>保存中...</div>}
      </div>
    </div>
  );
}

function SettingsSection({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-[12px] text-text-muted uppercase tracking-wider font-medium">{title}</h3>
        {description && <p className="text-[11px] text-text-muted mt-0.5">{description}</p>}
      </div>
      <div className="space-y-4 bg-surface rounded-xl border border-border p-5 shadow-sm">{children}</div>
    </div>
  );
}

function SettingsField({ label, description, children }: { label: string; description?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-[13px] text-text-secondary font-medium">{label}</label>
      {description && <p className="text-[11px] text-text-muted">{description}</p>}
      {children}
    </div>
  );
}
