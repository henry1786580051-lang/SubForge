"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import { Icon } from "@iconify/react";
import { useAppStore } from "@/store/appStore";
import { configApi, openNativeLogsFolder, transcribeApi, tasksApi } from "@/lib/api";
import type { AsrModelInfo, AsrModelStatus, AsrModelTestResult } from "@/lib/api";
import { groupNvidiaModels } from "@/lib/llmModels";
import {
  ENGLISH_LENGTH_PRESETS,
  FASTER_WHISPER_MODELS,
  LLM_PROVIDERS,
  MLX_WHISPER_MODELS,
  SETTINGS_VIEWS,
  WHISPER_CPP_MODELS,
  type SettingsView,
} from "@/features/settings/catalog";

export function SettingsPanel() {
  const { setActiveView } = useAppStore();
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [hwInfo, setHwInfo] = useState<{ chip: string; device: string; n_threads: number; compute_type: string; gpu: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("deepseek");
  const [activeSettingsView, setActiveSettingsView] = useState<SettingsView>("llm");
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [asrModels, setAsrModels] = useState<AsrModelInfo[]>([]);
  const [modelStatus, setModelStatus] = useState<AsrModelStatus | null>(null);
  const [testingAsrModel, setTestingAsrModel] = useState(false);
  const [asrTestResult, setAsrTestResult] = useState<AsrModelTestResult | null>(null);
  const [alignmentSearch, setAlignmentSearch] = useState("");
  const [alignmentFilter, setAlignmentFilter] = useState<"recommended" | "installed" | "all">("recommended");
  const downloadPollsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  useEffect(() => () => {
    downloadPollsRef.current.forEach((id) => clearInterval(id));
    downloadPollsRef.current.clear();
  }, []);
  const [detectedModels, setDetectedModels] = useState<string[] | null>(null);
  const [detectingModels, setDetectingModels] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [modelSearch, setModelSearch] = useState("");
  const [expandedModelCompany, setExpandedModelCompany] = useState<string | null>(null);
  const [testingLlm, setTestingLlm] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<{ ok: boolean; model?: string; error?: string } | null>(null);
  const [testingAzureTranslator, setTestingAzureTranslator] = useState(false);
  const [azureTranslatorTestResult, setAzureTranslatorTestResult] = useState<{ ok: boolean; translated?: string; error?: string } | null>(null);
  const [testingWhisper, setTestingWhisper] = useState(false);
  const [whisperTestResult, setWhisperTestResult] = useState<{ ok: boolean; error?: string } | null>(null);
  const [whisperModels, setWhisperModels] = useState<string[] | null>(null);
  const [detectingWhisperModels, setDetectingWhisperModels] = useState(false);
  const [whisperModelError, setWhisperModelError] = useState<string | null>(null);

  const refreshAsrState = async () => {
    const [models, status] = await Promise.all([
      transcribeApi.listModels(),
      transcribeApi.modelStatus(),
    ]);
    setAsrModels(models);
    setModelStatus(status);
  };

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
      if (data.llm_provider !== undefined) storeUpdates.llmProvider = data.llm_provider as string;
      if (data.llm_model !== undefined) storeUpdates.llmModel = data.llm_model as string;
      if (data.whisper_model_size !== undefined) storeUpdates.whisperModelSize = data.whisper_model_size as string;
      if (data.whisperx_alignment_strategy !== undefined) {
        storeUpdates.whisperxAlignmentStrategy = data.whisperx_alignment_strategy as "auto" | "manual";
      }
      if (data.whisperx_align_model !== undefined) storeUpdates.whisperxAlignModel = data.whisperx_align_model as string;
      if (data.whisperx_batch_size !== undefined) storeUpdates.whisperxBatchSize = Number(data.whisperx_batch_size || 4);
      if (data.whisperx_supported !== undefined) storeUpdates.whisperxSupported = !!data.whisperx_supported;
      if (data.enable_audio_enhancement !== undefined) storeUpdates.enableAudioEnhancement = !!data.enable_audio_enhancement;
      if (data.speaker_diarization !== undefined) storeUpdates.speakerDiarization = data.speaker_diarization as "off" | "two" | "auto" | "fixed";
      if (data.speaker_count !== undefined) storeUpdates.speakerCount = Number(data.speaker_count || 2);
      if (Object.keys(storeUpdates).length > 0) useAppStore.getState().setConfig(storeUpdates);
      // Prefer the persisted provider; detect legacy configurations by URL.
      const url = (data.llm_base_url as string) || "";
      const provider = LLM_PROVIDERS.find((p) => p.baseUrl && url.startsWith(p.baseUrl));
      const providerId = data.llm_provider as string;
      if (LLM_PROVIDERS.some((item) => item.id === providerId)) {
        setSelectedProvider(providerId);
      } else if (provider) {
        setSelectedProvider(provider.id);
      } else {
        setSelectedProvider("custom");
      }
      setLoading(false);
    }).catch(() => setLoading(false));

    // Detect hardware
    transcribeApi.hardware().then((hw) => setHwInfo(hw)).catch(() => {});

    // Check which whisper models are already downloaded
    const asrRefreshTimer = window.setTimeout(() => {
      refreshAsrState().catch(() => {});
    }, 0);
    return () => window.clearTimeout(asrRefreshTimer);
  }, []);

  const handleSave = async (key: string, value: unknown) => {
    setSaving(true);
    try {
      await configApi.update(key, value);
      const secretKey = ["llm_api_key", "whisper_api_key", "huggingface_token", "azure_translator_key"].includes(key);
      if (["max_word_count_cjk", "max_word_count_english"].includes(key)) {
        setSettings(await configApi.get());
      } else {
        setSettings((prev) => ({
          ...prev,
          [key]: secretKey ? "" : value,
          ...(secretKey ? { [`${key}_configured`]: true } : {}),
        }));
      }
      const configKeyMap: Record<string, string> = {
        transcribe_model: "transcribeModel",
        source_language: "sourceLanguage",
        target_language: "targetLanguage",
        translator: "translator",
        llm_model: "llmModel",
        need_reflect: "needReflect",
        custom_prompt: "customPrompt",
        whisper_model_size: "whisperModelSize",
        whisperx_alignment_strategy: "whisperxAlignmentStrategy",
        whisperx_align_model: "whisperxAlignModel",
        whisperx_batch_size: "whisperxBatchSize",
        enable_audio_enhancement: "enableAudioEnhancement",
        speaker_diarization: "speakerDiarization",
        speaker_count: "speakerCount",
      };
      const mappedKey = configKeyMap[key];
      if (mappedKey) {
        useAppStore.getState().setConfig({ [mappedKey]: value });
      }
      if (["transcribe_model", "whisper_model_size", "whisperx_alignment_strategy", "whisperx_align_model", "source_language", "whisper_model_dir", "whisper_cpp_path"].includes(key)) {
        setAsrTestResult(null);
        await refreshAsrState();
      }
      if (["azure_translator_key", "azure_translator_region", "azure_translator_endpoint"].includes(key)) {
        setAzureTranslatorTestResult(null);
      }
    }
    catch (err) { useAppStore.getState().setError(err instanceof Error ? err.message : "Save failed"); }
    finally { setSaving(false); }
  };

  const handleTestAzureTranslator = async () => {
    setTestingAzureTranslator(true);
    setAzureTranslatorTestResult(null);
    try {
      const endpoint = ((settings.azure_translator_endpoint as string) || "https://api.cognitive.microsofttranslator.com").trim();
      const region = ((settings.azure_translator_region as string) || "").trim();
      const key = ((settings.azure_translator_key as string) || "").trim();
      await configApi.update("azure_translator_endpoint", endpoint);
      await configApi.update("azure_translator_region", region);
      if (key) await configApi.update("azure_translator_key", key);
      setSettings((prev) => ({
        ...prev,
        azure_translator_endpoint: endpoint,
        azure_translator_region: region,
        azure_translator_key: "",
        azure_translator_key_configured: key ? true : prev.azure_translator_key_configured,
      }));
      setAzureTranslatorTestResult(await configApi.testAzureTranslator());
    } catch (err) {
      setAzureTranslatorTestResult({
        ok: false,
        error: err instanceof Error ? err.message : "测试失败",
      });
    } finally {
      setTestingAzureTranslator(false);
    }
  };

  const handleProviderChange = async (providerId: string) => {
    const provider = LLM_PROVIDERS.find((p) => p.id === providerId);
    if (!provider || providerId === selectedProvider || saving) return;

    setSaving(true);
    try {
      const result = await configApi.switchLlmProvider({
        provider: providerId,
        current_base_url: (settings.llm_base_url as string) || "",
        current_api_key: (settings.llm_api_key as string) || "",
        current_model: (settings.llm_model as string) || "",
      });
      setSelectedProvider(result.provider);
      setSettings((prev) => ({
        ...prev,
        llm_provider: result.provider,
        llm_base_url: result.base_url,
        llm_api_key: "",
        llm_api_key_configured: result.api_key_configured,
        llm_model: result.model,
      }));
      useAppStore.getState().setConfig({
        llmProvider: result.provider,
        llmModel: result.model,
      });
      setLlmTestResult(null);
    } catch (err) {
      useAppStore.getState().setError(
        err instanceof Error ? err.message : "切换 LLM 服务失败"
      );
    } finally {
      setSaving(false);
    }

    setDetectedModels(null);
    setDetectError(null);
    setModelSearch("");
    setExpandedModelCompany(null);
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
        setDownloadingModel(null);
        await refreshAsrState();
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
              setDownloadingModel(null);
              setDownloadProgress((prev) => { const n = { ...prev }; delete n[modelId]; return n; });
              await refreshAsrState();
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

  const alignmentModels = useMemo(
    () => asrModels.filter((model) => model.type === "alignment"),
    [asrModels]
  );
  const installedAlignmentCount = alignmentModels.filter((model) => model.downloaded).length;
  const filteredAlignmentModels = useMemo(() => {
    const query = alignmentSearch.trim().toLowerCase();
    return alignmentModels.filter((model) => {
      if (alignmentFilter === "installed" && !model.downloaded) return false;
      if (alignmentFilter === "recommended") {
        const configuredLanguage = String(settings.source_language || "auto");
        const sourceLanguage = configuredLanguage === "nb" ? "no" : configuredLanguage;
        if (sourceLanguage !== "auto" && model.language !== sourceLanguage) return false;
        if (
          sourceLanguage === "auto"
          && !model.downloaded
          && !["en", "zh", "ja", "ko"].includes(model.language || "")
        ) return false;
      }
      if (!query) return true;
      return [model.language_name, model.language, model.name, model.align_model]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query));
    });
  }, [alignmentFilter, alignmentModels, alignmentSearch, settings.source_language]);

  if (loading) return <div className="flex items-center justify-center h-full"><div className="w-6 h-6 rounded-full border-2 border-accent/20 border-t-accent animate-spin" /></div>;

  const currentProvider = LLM_PROVIDERS.find((p) => p.id === selectedProvider);
  const nvidiaModelGroups = groupNvidiaModels(detectedModels || [], modelSearch);
  const effectiveWhisperModel = modelStatus?.model_value || (settings.whisper_model_size as string) || "large-v3";
  const activeViewMeta = SETTINGS_VIEWS.find((view) => view.id === activeSettingsView) || SETTINGS_VIEWS[0];

  return (
    <div className="flex h-full flex-col overflow-hidden bg-background">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-surface px-5">
        <div className="flex items-center gap-2.5">
          <Icon icon="solar:settings-linear" className="h-4 w-4 text-text-muted" />
          <h2 className="text-[13px] font-semibold text-text-primary">设置</h2>
          {saving && (
            <span className="flex items-center gap-1.5 text-[10px] text-accent">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
              正在保存
            </span>
          )}
        </div>
        <button onClick={() => setActiveView("workflow")} className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[210px_minmax(0,1fr)]">
        <nav className="shrink-0 overflow-x-auto border-b border-border bg-surface px-3 py-2 md:overflow-y-auto md:border-b-0 md:border-r md:px-3 md:py-4" aria-label="设置分类">
          <div className="flex min-w-max gap-1 md:min-w-0 md:flex-col">
            {SETTINGS_VIEWS.map((view) => (
              <button
                key={view.id}
                onClick={() => setActiveSettingsView(view.id)}
                className={`flex min-h-11 items-center gap-3 rounded-md px-3 py-2 text-left transition-colors md:w-full ${
                  activeSettingsView === view.id
                    ? "bg-accent-dim text-accent"
                    : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                }`}
              >
                <Icon icon={view.icon} className="h-[18px] w-[18px] shrink-0" />
                <span className="min-w-0">
                  <span className="block text-[12px] font-medium">{view.label}</span>
                  <span className="hidden truncate text-[9px] text-text-muted md:block">{view.description}</span>
                </span>
              </button>
            ))}
          </div>
        </nav>

        <main className="min-h-0 overflow-y-auto">
          <div className="mx-auto w-full max-w-[880px] px-5 pb-20 pt-6 lg:px-8">
            <div className="mb-6 border-b border-border pb-4">
              <div className="flex items-center gap-2">
                <Icon icon={activeViewMeta.icon} className="h-5 w-5 text-accent" />
                <h1 className="text-[18px] font-semibold text-text-primary">{activeViewMeta.label}</h1>
              </div>
              <p className="mt-1 text-[11px] text-text-muted">{activeViewMeta.description}</p>
            </div>

            <div className="space-y-6">

        {/* LLM Configuration */}
        {activeSettingsView === "llm" && (
          <>
        <SettingsSection title="LLM 配置" description="用于字幕优化和智能翻译">
          <SettingsField label="服务商">
            <div className="flex items-center gap-3">
              <select
                value={selectedProvider}
                onChange={(event) => handleProviderChange(event.target.value)}
                disabled={saving}
                className="input-field flex-1"
              >
                {LLM_PROVIDERS.map((provider) => (
                  <option key={provider.id} value={provider.id}>{provider.name}</option>
                ))}
              </select>
              <span className={`flex shrink-0 items-center gap-1.5 text-[10px] ${settings.llm_api_key || settings.llm_api_key_configured ? "text-emerald-600" : "text-amber-600"}`}>
                <span className={`h-1.5 w-1.5 rounded-full ${settings.llm_api_key || settings.llm_api_key_configured ? "bg-emerald-500" : "bg-amber-500"}`} />
                {settings.llm_api_key || settings.llm_api_key_configured ? "已配置" : "待配置"}
              </span>
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
              onBlur={(e) => {
                if (e.target.value) void handleSave("llm_api_key", e.target.value);
              }}
              placeholder={settings.llm_api_key_configured ? "已安全保存，输入新值覆盖" : "sk-..."} className="input-field" />
          </SettingsField>
          <SettingsField
            label="模型"
            description={currentProvider?.groupedModels ? "检测后按模型公司分类管理" : "点击「检测模型」从服务商获取可用模型列表"}
          >
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

            {detectedModels !== null && detectedModels.length > 0 && currentProvider?.groupedModels && (
              <div className="mb-2 overflow-hidden rounded-lg border border-border bg-[rgba(0,0,0,0.01)]">
                <div className="flex items-center gap-2 border-b border-border px-3 py-2">
                  <Icon icon="solar:magnifer-linear" className="h-3.5 w-3.5 shrink-0 text-text-muted" />
                  <input
                    type="search"
                    value={modelSearch}
                    onChange={(event) => setModelSearch(event.target.value)}
                    placeholder={`搜索 ${detectedModels.length} 个模型`}
                    className="min-w-0 flex-1 bg-transparent text-[11px] text-text-primary outline-none placeholder:text-text-muted"
                  />
                  <span className="shrink-0 text-[10px] text-text-muted">{nvidiaModelGroups.length} 家公司</span>
                </div>
                <div className="max-h-80 overflow-y-auto p-1.5">
                  {nvidiaModelGroups.length > 0 ? nvidiaModelGroups.map((group) => {
                    const expanded = expandedModelCompany === group.id || modelSearch.trim().length > 0;
                    const selectedInGroup = group.models.includes((settings.llm_model as string) || "");
                    return (
                      <div key={group.id} className="border-b border-border last:border-b-0">
                        <button
                          type="button"
                          aria-expanded={expanded}
                          onClick={() => setExpandedModelCompany((current) => current === group.id ? null : group.id)}
                          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-surface-hover"
                        >
                          <Icon
                            icon="solar:alt-arrow-right-linear"
                            className={`h-3.5 w-3.5 shrink-0 text-text-muted transition-transform ${expanded ? "rotate-90" : ""}`}
                          />
                          <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-text-primary">{group.name}</span>
                          {selectedInGroup && <span className="text-[9px] font-medium text-accent">当前</span>}
                          <span className="min-w-6 text-right font-mono text-[9px] text-text-muted">{group.models.length}</span>
                        </button>
                        {expanded && (
                          <div className="space-y-1 pb-2 pl-8 pr-2">
                            {group.models.map((model) => {
                              const selected = (settings.llm_model as string) === model;
                              return (
                                <button
                                  key={model}
                                  type="button"
                                  onClick={() => void handleSave("llm_model", model)}
                                  className={`flex w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-[10px] transition-colors ${
                                    selected
                                      ? "border-accent bg-accent-dim font-medium text-accent"
                                      : "border-transparent text-text-secondary hover:border-border hover:bg-surface"
                                  }`}
                                >
                                  <span className="min-w-0 flex-1 break-all">{model}</span>
                                  {selected && <Icon icon="solar:check-circle-bold" className="h-3.5 w-3.5 shrink-0" />}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  }) : (
                    <p className="px-3 py-5 text-center text-[11px] text-text-muted">没有匹配的模型</p>
                  )}
                </div>
              </div>
            )}

            {detectedModels !== null && detectedModels.length > 0 && !currentProvider?.groupedModels && (
              <div className="flex max-h-48 flex-wrap gap-1.5 overflow-auto">
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
            )}

            {detectedModels !== null && detectedModels.length === 0 && !detectError && (
              <p className="text-[11px] text-text-muted">未发现可用模型</p>
            )}

            <input type="text" value={(settings.llm_model as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, llm_model: e.target.value }))}
              onBlur={(e) => handleSave("llm_model", e.target.value)}
              placeholder="输入或选择模型名称" className="input-field mt-2" />

            <div className="flex items-center gap-2 mt-2">
              <button onClick={async () => {
                setTestingLlm(true); setLlmTestResult(null);
                try { const r = await configApi.testLlm(); setLlmTestResult(r); } catch (err) { setLlmTestResult({ ok: false, error: err instanceof Error ? err.message : "测试失败" }); } finally { setTestingLlm(false); }
              }} disabled={testingLlm || (!settings.llm_api_key && !settings.llm_api_key_configured) || !settings.llm_model}
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

        <SettingsSection title="请求性能" description="控制 LLM 并发和单次提交规模">
          <SettingsField label="并发线程数" description="同时处理的 LLM 请求数量，过高可能触发服务商限流">
            <div className="flex items-center gap-3">
              <input type="range" min={1} max={20} value={(settings.thread_num as number) || 3}
                onChange={(e) => handleSave("thread_num", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="w-8 text-right font-mono text-[12px] text-text-primary">{(settings.thread_num as number) || 3}</span>
            </div>
          </SettingsField>
          <SettingsField label="批处理大小" description="每次提交的字幕条数，系统仍会自动附带前后文">
            <div className="flex items-center gap-3">
              <input type="range" min={1} max={50} value={(settings.batch_size as number) || 10}
                onChange={(e) => handleSave("batch_size", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="w-8 text-right font-mono text-[12px] text-text-primary">{(settings.batch_size as number) || 10}</span>
            </div>
          </SettingsField>
          <SettingsField label="日志详细度" description="摘要仅记录任务指标；标准保留截断内容；调试记录完整请求与响应">
            <select
              value={(settings.llm_log_level as string) || "summary"}
              onChange={(e) => handleSave("llm_log_level", e.target.value)}
              className="input-field"
            >
              <option value="summary">摘要（推荐）</option>
              <option value="standard">标准</option>
              <option value="debug">调试</option>
            </select>
          </SettingsField>
        </SettingsSection>
          </>
        )}

        {/* ASR Configuration */}
        {activeSettingsView === "asr" && (
          <>
        <SettingsSection title="语音识别方案" description="顶部确认下一次转录使用的完整方案，下方分别调整引擎和模型">
          {modelStatus && (
            <div className="rounded-lg border border-accent/20 bg-accent-dim/45 p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[10px] font-semibold text-accent">
                    <span className={`h-1.5 w-1.5 rounded-full ${modelStatus.testable ? "bg-emerald-500" : "bg-amber-500"}`} />
                    当前方案
                  </div>
                  <p className="mt-1.5 text-[14px] font-semibold text-text-primary">
                    {modelStatus.engine_name} · {modelStatus.model_name}
                  </p>
                  <p className="mt-1 text-[10px] leading-4 text-text-muted">
                    {modelStatus.model_message}
                    {hwInfo && hwInfo.chip !== "Unknown" ? ` · ${hwInfo.chip} · ${hwInfo.gpu}` : ""}
                  </p>
                  {modelStatus.engine === "whisperx" && (
                    <p className="mt-1 text-[10px] leading-4 text-text-secondary">
                      词级对齐：{modelStatus.alignment_strategy === "auto" ? "按源语言自动匹配" : "手动指定"}
                      {modelStatus.alignment_language_name ? ` · ${modelStatus.alignment_language_name}` : ""}
                      {modelStatus.alignment_language === "auto" ? " · 识别语言后加载" : ""}
                    </p>
                  )}
                </div>
                <button
                  onClick={async () => {
                    setTestingAsrModel(true);
                    setAsrTestResult(null);
                    try {
                      const result = await transcribeApi.testModel();
                      setAsrTestResult(result);
                      await refreshAsrState();
                    } catch (err) {
                      setAsrTestResult({
                        ...modelStatus,
                        ok: false,
                        error: err instanceof Error ? err.message : "模型测试失败",
                      });
                    } finally {
                      setTestingAsrModel(false);
                    }
                  }}
                  disabled={testingAsrModel || !modelStatus.testable}
                  className="shrink-0 rounded-md bg-accent px-3 py-1.5 text-[11px] font-semibold text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {testingAsrModel ? "正在测试..." : "测试当前方案"}
                </button>
              </div>
              {asrTestResult && (
                <div className={`mt-3 rounded-md px-3 py-2 text-[11px] ${asrTestResult.ok ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-600"}`}>
                  {asrTestResult.ok
                    ? `测试通过 · ${asrTestResult.elapsed_seconds}s · ${asrTestResult.segment_count} 个片段`
                    : `测试失败：${asrTestResult.error || "未知错误"}`}
                </div>
              )}
            </div>
          )}

          <SettingsField label="识别引擎">
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: "whisperx", name: "WhisperX", desc: "Apple 使用 MLX；Windows 使用 CTranslate2 + forced alignment" },
                { id: "whisper_cpp", name: "Whisper.cpp", desc: "macOS 推荐 · Metal GPU 加速" },
                { id: "faster_whisper", name: "FasterWhisper", desc: "NVIDIA GPU 推荐 · CUDA 加速" },
                { id: "whisper_api", name: "Whisper API", desc: "云端接口，无需本地算力" },
              ].map((e) => {
                const unsupported = e.id === "whisperx" && settings.whisperx_supported === false;
                return (
                <button key={e.id} disabled={unsupported} onClick={async () => { await handleSave("transcribe_model", e.id); useAppStore.getState().setConfig({ transcribeModel: e.id }); }}
                  className={`rounded-lg border p-3 text-left transition-all ${
                    (settings.transcribe_model as string) === e.id
                      ? "border-accent bg-accent-dim"
                      : unsupported
                      ? "border-border opacity-45 cursor-not-allowed"
                      : "border-border hover:border-[rgba(0,0,0,0.12)]"
                  }`}>
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-medium text-text-primary">{e.name}</span>
                    {(settings.transcribe_model as string) === e.id && (
                      <span className="ml-auto inline-flex items-center gap-1 text-[9px] font-semibold text-accent">
                        <span className="h-1.5 w-1.5 rounded-full bg-accent" />当前
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-text-muted">{unsupported ? "当前平台不支持" : e.desc}</span>
                </button>
              )})}
            </div>
            {(settings.transcribe_model as string) === "faster_whisper" && (
              <p className="text-[10px] text-emerald-600 mt-1.5">内置 CTranslate2 运行时；Windows 自动使用可用的 NVIDIA CUDA，否则回退 CPU</p>
            )}
            {(settings.transcribe_model as string) === "whisperx" && (
              <p className="text-[10px] text-emerald-600 mt-1.5">
                {settings.whisperx_backend === "mlx"
                  ? "当前使用 Apple Silicon 优化的 MLX Whisper，并通过 forced alignment 生成词级时间轴"
                  : "当前使用 Windows CTranslate2/CUDA 或 CPU 路径，并通过 forced alignment 生成词级时间轴"}
              </p>
            )}
            {(settings.transcribe_model as string) === "whisper_cpp" && (
              <p className="text-[10px] text-emerald-600 mt-1.5">Apple Silicon 原生 Metal 加速，NVIDIA GPU 上使用 CPU + int8 量化</p>
            )}
          </SettingsField>

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
                onBlur={(e) => {
                  if (e.target.value) void handleSave("whisper_api_key", e.target.value);
                }}
                placeholder={settings.whisper_api_key_configured ? "已安全保存，输入新值覆盖" : "sk-..."} className="input-field" />
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
            <details className="rounded-lg border border-border bg-background/60">
              <summary className="cursor-pointer px-4 py-3 text-[12px] font-medium text-text-secondary transition-colors hover:text-text-primary">
                高级配置
              </summary>
              <div className="space-y-4 border-t border-border px-4 py-4">
            {(settings.transcribe_model as string) === "whisper_cpp" && (
              <SettingsField label="Whisper.cpp 程序路径" description="填写 whisper-cli 可执行文件路径；模型文件和程序文件必须同时存在">
                <input type="text" value={(settings.whisper_cpp_path as string) || ""}
                  onChange={(e) => setSettings((prev) => ({ ...prev, whisper_cpp_path: e.target.value }))}
                  onBlur={(e) => handleSave("whisper_cpp_path", e.target.value)}
                  placeholder="/opt/homebrew/bin/whisper-cli" className="input-field" />
              </SettingsField>
            )}
            <SettingsField label="模型存储目录" description={(settings.transcribe_model as string) === "whisperx" ? "保存 WhisperX 转录与 forced alignment 模型；首次使用可自动下载" : "留空则使用默认目录（~/SubForge/models）"}>
              <input type="text" value={(settings.whisper_model_dir as string) || ""}
                onChange={(e) => setSettings((prev) => ({ ...prev, whisper_model_dir: e.target.value }))}
                onBlur={(e) => handleSave("whisper_model_dir", e.target.value)}
                placeholder="~/SubForge/models" className="input-field" />
            </SettingsField>
              </div>
            </details>

            <SettingsField label="语音转录模型" description="选择负责将语音识别为文字的 Whisper 模型；当前使用项会同步到下一次转录任务">
              <details className="rounded-lg border border-border bg-background/60">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3">
                  <span className="min-w-0">
                    <span className="block truncate text-[12px] font-semibold text-text-primary">
                      {modelStatus?.model_name || "选择模型"}
                    </span>
                    <span className="mt-0.5 block text-[10px] text-text-muted">
                      {modelStatus?.model_ready ? "本地已就绪" : modelStatus?.model_state === "on_demand" ? "首次使用自动下载" : "检查模型状态"}
                    </span>
                  </span>
                  <span className="shrink-0 text-[10px] font-medium text-accent">更换模型</span>
                </summary>
                <div className="space-y-2 border-t border-border p-3">
                {(settings.transcribe_model === "whisper_cpp" ? WHISPER_CPP_MODELS : settings.transcribe_model === "whisperx" ? MLX_WHISPER_MODELS : FASTER_WHISPER_MODELS).map((m) => {
                  const apiModel = asrModels.find((item) =>
                    item.category === settings.transcribe_model && (item.value || item.id) === m.id
                  );
                  const isSelected = effectiveWhisperModel === m.id || Boolean(apiModel?.selected);
                  const isReady = Boolean(apiModel?.downloaded);
                  const isOnDemand = apiModel?.state === "on_demand" || Boolean((m as { onDemand?: boolean }).onDemand);
                  const downloadKey = apiModel?.id || m.id;
                  return (
                  <div key={m.id} className={`flex items-center justify-between gap-3 rounded-lg border p-3 ${isSelected ? "border-accent/40 bg-accent-dim/40" : "border-border bg-[rgba(0,0,0,0.01)]"}`}>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[12px] font-medium text-text-primary">{m.name}</span>
                        <span className="text-[10px] text-text-muted font-mono">{m.size}</span>
                        {isSelected && <span className="text-[9px] px-1.5 py-0.5 rounded bg-accent text-white">当前使用</span>}
                        {isReady && <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-600">本地已就绪</span>}
                        {!isReady && isOnDemand && <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700">按需下载</span>}
                      </div>
                      <span className="text-[10px] text-text-muted">{m.desc}</span>
                      {apiModel?.path && <p className="mt-1 max-w-[360px] truncate font-mono text-[9px] text-text-muted" title={apiModel.path}>{apiModel.path}</p>}
                    </div>
                    <button
                      onClick={() => {
                        if (isSelected) return;
                        if (settings.transcribe_model === "whisperx" || isReady) {
                          void handleSave("whisper_model_size", m.id);
                        } else {
                          void handleDownloadModel(downloadKey);
                        }
                      }}
                      disabled={downloadingModel === downloadKey || isSelected}
                      className={`px-3 py-1.5 text-[11px] rounded-md transition-all font-medium disabled:opacity-50 ${
                        isSelected
                          ? "bg-accent text-white cursor-default"
                          : isReady
                          ? "bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                          : "bg-accent-dim text-accent hover:bg-accent/15"
                      }`}>
                      {isSelected ? "当前使用" : isReady ? "使用" : downloadingModel === downloadKey ? (
                        <span className="flex items-center gap-1.5">
                          <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="42 21" strokeLinecap="round" /></svg>
                          {downloadProgress[downloadKey] != null ? `${downloadProgress[downloadKey]}%` : "下载中"}
                        </span>
                      ) : settings.transcribe_model === "whisperx" ? "选择" : "下载"}
                    </button>
                  </div>
                  );
                })}
                </div>
              </details>
            </SettingsField>
            {(settings.transcribe_model as string) === "whisperx" && (
              <SettingsField
                label="词级时间轴模型"
                description="默认按源语言自动匹配；只需下载经常使用的语言"
              >
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-1 rounded-lg bg-background p-1 ring-1 ring-border">
                    {([
                      ["auto", "自动匹配", "推荐"],
                      ["manual", "手动指定", "高级"],
                    ] as const).map(([value, label, hint]) => {
                      const active = (settings.whisperx_alignment_strategy || "auto") === value;
                      return (
                        <button
                          key={value}
                          onClick={() => void handleSave("whisperx_alignment_strategy", value)}
                          className={`flex items-center justify-center gap-2 rounded-md px-3 py-2 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 ${
                            active
                              ? "bg-surface text-text-primary shadow-sm"
                              : "text-text-muted hover:text-text-secondary"
                          }`}
                        >
                          {label}
                          <span className={`text-[9px] font-medium ${active ? "text-accent" : "text-text-muted"}`}>
                            {hint}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  <details className="group overflow-hidden rounded-lg border border-border bg-background/60">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/30">
                      <span className="flex min-w-0 items-center gap-3">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent-dim text-accent">
                          <Icon icon="solar:align-bottom-linear" className="h-4 w-4" />
                        </span>
                        <span className="min-w-0">
                          <span className="block text-[12px] font-semibold text-text-primary">
                            {modelStatus?.alignment_strategy === "manual"
                              ? modelStatus.alignment_language_name || "自定义对齐模型"
                              : modelStatus?.alignment_language_name
                                ? `${modelStatus.alignment_language_name}自动匹配`
                                : "识别语言后自动匹配"}
                          </span>
                          <span className="mt-0.5 block text-[10px] text-text-muted">
                            已安装 {installedAlignmentCount} / {alignmentModels.length} 种语言
                          </span>
                        </span>
                      </span>
                      <span className="flex shrink-0 items-center gap-2 text-[10px] font-medium text-accent">
                        管理模型
                        <Icon icon="solar:alt-arrow-down-linear" className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
                      </span>
                    </summary>

                    <div className="border-t border-border p-3">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex rounded-md bg-surface-hover p-0.5">
                          {([
                            ["recommended", "推荐"],
                            ["installed", "已安装"],
                            ["all", "全部"],
                          ] as const).map(([value, label]) => (
                            <button
                              key={value}
                              onClick={() => setAlignmentFilter(value)}
                              className={`rounded px-2.5 py-1 text-[10px] font-medium transition-colors ${
                                alignmentFilter === value
                                  ? "bg-surface text-text-primary shadow-sm"
                                  : "text-text-muted hover:text-text-secondary"
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                        <label className="flex min-w-0 items-center gap-2 rounded-md border border-border bg-surface px-2.5 py-1.5 focus-within:border-accent/50 sm:w-56">
                          <Icon icon="solar:magnifer-linear" className="h-3.5 w-3.5 shrink-0 text-text-muted" />
                          <input
                            value={alignmentSearch}
                            onChange={(event) => setAlignmentSearch(event.target.value)}
                            placeholder="搜索语言或模型"
                            className="min-w-0 flex-1 bg-transparent text-[10px] text-text-primary outline-none placeholder:text-text-muted"
                          />
                        </label>
                      </div>

                      <div className="mt-3 max-h-72 space-y-1.5 overflow-y-auto pr-1">
                        {filteredAlignmentModels.map((model) => {
                          const manual = (settings.whisperx_alignment_strategy || "auto") === "manual";
                          const selected = manual && (settings.whisperx_align_model as string) === model.align_model;
                          const downloading = downloadingModel === model.id;
                          return (
                            <div
                              key={model.id}
                              className={`flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors ${
                                selected ? "bg-accent-dim/60 ring-1 ring-accent/25" : "bg-surface-hover/65 hover:bg-surface-hover"
                              }`}
                            >
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                                  <span className="text-[11px] font-semibold text-text-primary">
                                    {model.language_name || model.name}
                                  </span>
                                  <span className="font-mono text-[9px] uppercase text-text-muted">{model.language}</span>
                                  {model.downloaded && (
                                    <span className="text-[9px] font-medium text-emerald-600">本地可用</span>
                                  )}
                                </div>
                                <p className="mt-0.5 max-w-xl truncate text-[9px] text-text-muted" title={model.align_model}>
                                  {model.align_model} · {model.source === "torchaudio" ? "TorchAudio" : "Hugging Face"}
                                  {model.size ? ` · ${model.size}` : ""}
                                </p>
                              </div>
                              <button
                                onClick={async () => {
                                  if (selected || downloading) return;
                                  if (manual) await handleSave("whisperx_align_model", model.align_model || "");
                                  if (!model.downloaded) await handleDownloadModel(model.id);
                                }}
                                disabled={selected || downloading}
                                className={`shrink-0 rounded-md px-2.5 py-1.5 text-[10px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:cursor-default ${
                                  selected
                                    ? "bg-accent text-white"
                                    : model.downloaded && manual
                                      ? "bg-surface text-text-secondary ring-1 ring-border hover:text-accent"
                                      : model.downloaded
                                        ? "text-emerald-700"
                                        : "bg-accent text-white hover:bg-accent-hover"
                                }`}
                              >
                                {selected
                                  ? "当前使用"
                                  : downloading
                                    ? `${downloadProgress[model.id] ?? 0}%`
                                    : model.downloaded
                                      ? manual ? "使用" : "已安装"
                                      : "下载"}
                              </button>
                            </div>
                          );
                        })}
                        {filteredAlignmentModels.length === 0 && (
                          <div className="px-3 py-8 text-center text-[10px] text-text-muted">
                            {alignmentFilter === "installed" ? "尚未安装对齐模型" : "没有匹配的语言模型"}
                          </div>
                        )}
                      </div>

                      {(settings.whisperx_alignment_strategy || "auto") === "manual" && (
                        <div className="mt-3 border-t border-border pt-3">
                          <label className="text-[10px] font-medium text-text-secondary">自定义模型 ID</label>
                          <input
                            type="text"
                            value={(settings.whisperx_align_model as string) || ""}
                            onChange={(event) => setSettings((prev) => ({ ...prev, whisperx_align_model: event.target.value }))}
                            onBlur={(event) => void handleSave("whisperx_align_model", event.target.value.trim())}
                            placeholder="organization/wav2vec2-model"
                            className="input-field mt-1.5"
                          />
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              </SettingsField>
            )}
            </>
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
          </>
        )}

        {/* Translation */}
        {activeSettingsView === "subtitle" && (
          <>
        <SettingsSection title="翻译服务">
          <SettingsField label="默认翻译服务">
            <select value={(settings.translator as string) || "bing"} onChange={(e) => handleSave("translator", e.target.value)} className="input-field">
              <option value="bing">Microsoft Azure Translator</option><option value="google">Google 翻译</option><option value="deeplx">DeepLX</option><option value="llm">LLM 翻译</option>
            </select>
          </SettingsField>
          {(settings.translator as string) === "bing" && (
            <>
              <SettingsField label="Azure 服务终结点" description="默认使用微软全球 Translator v3 终结点，也支持 Azure 资源的自定义域名">
                <input
                  type="url"
                  value={(settings.azure_translator_endpoint as string) || "https://api.cognitive.microsofttranslator.com"}
                  onChange={(event) => setSettings((prev) => ({ ...prev, azure_translator_endpoint: event.target.value }))}
                  onBlur={(event) => void handleSave("azure_translator_endpoint", event.target.value.trim())}
                  placeholder="https://api.cognitive.microsofttranslator.com"
                  className="input-field"
                />
              </SettingsField>
              <SettingsField label="Azure API Key" description="密钥仅保存在本机设置中，界面和日志不会回显">
                <input
                  type="password"
                  value={(settings.azure_translator_key as string) || ""}
                  onChange={(event) => setSettings((prev) => ({ ...prev, azure_translator_key: event.target.value }))}
                  onBlur={(event) => {
                    const value = event.target.value.trim();
                    if (value) void handleSave("azure_translator_key", value);
                  }}
                  placeholder={settings.azure_translator_key_configured ? "已配置，输入新密钥可替换" : "输入 Azure Translator 密钥"}
                  autoComplete="off"
                  className="input-field"
                />
              </SettingsField>
              <SettingsField label="资源区域" description="多服务或区域资源需要填写，例如 eastasia；全球单服务资源可留空">
                <input
                  type="text"
                  value={(settings.azure_translator_region as string) || ""}
                  onChange={(event) => setSettings((prev) => ({ ...prev, azure_translator_region: event.target.value }))}
                  onBlur={(event) => void handleSave("azure_translator_region", event.target.value.trim())}
                  placeholder="可选，例如 eastasia"
                  className="input-field"
                />
                <div className="mt-2 flex min-h-7 items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleTestAzureTranslator()}
                    disabled={testingAzureTranslator || (!settings.azure_translator_key && !settings.azure_translator_key_configured)}
                    className="flex items-center gap-1.5 rounded-md bg-accent-dim px-3 py-1.5 text-[12px] font-medium text-accent transition-colors hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-40 btn-press"
                  >
                    {testingAzureTranslator ? (
                      <><Icon icon="solar:refresh-circle-linear" className="h-3.5 w-3.5 animate-spin" />测试中...</>
                    ) : (
                      <><Icon icon="solar:check-circle-linear" className="h-3.5 w-3.5" />测试连接</>
                    )}
                  </button>
                  {azureTranslatorTestResult && (
                    <span className={`text-[11px] ${azureTranslatorTestResult.ok ? "text-emerald-600" : "text-red-500"}`}>
                      {azureTranslatorTestResult.ok ? `连接成功：${azureTranslatorTestResult.translated}` : `失败：${azureTranslatorTestResult.error}`}
                    </span>
                  )}
                </div>
              </SettingsField>
            </>
          )}
          <SettingsField label="默认目标语言">
            <select value={(settings.target_language as string) || "english"} onChange={(e) => handleSave("target_language", e.target.value)} className="input-field">
              <option value="chinese">中文</option><option value="english">英文</option><option value="japanese">日文</option><option value="korean">韩文</option>
              <option value="french">法语</option><option value="german">德语</option><option value="spanish">西班牙语</option><option value="portuguese">葡萄牙语</option>
              <option value="russian">俄语</option><option value="arabic">阿拉伯语</option><option value="thai">泰语</option>
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
          <SettingsField label="中文字幕硬上限" description="中文、日语、韩语字幕不可超过的字符数">
            <div className="flex items-center gap-3">
              <input type="range" min={10} max={40} value={(settings.max_word_count_cjk as number) || 25}
                onChange={(e) => handleSave("max_word_count_cjk", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="text-[12px] text-text-primary font-mono w-8 text-right">{(settings.max_word_count_cjk as number) || 25}</span>
            </div>
          </SettingsField>
          <SettingsField
            label="英文字幕目标长度"
            description={`优先不超过 ${Number((settings.subtitle_length_policy as Record<string, unknown> | undefined)?.english_soft_limit || settings.max_word_count_english || 18)} 词；为保持语义完整，最多允许 ${Number((settings.subtitle_length_policy as Record<string, unknown> | undefined)?.english_hard_limit || Number(settings.max_word_count_english || 18) + 4)} 词`}
          >
            <div className="grid grid-cols-3 gap-2">
              {ENGLISH_LENGTH_PRESETS.map((preset) => {
                const selected = Number(settings.max_word_count_english || 18) === preset.value;
                return (
                  <button
                    key={preset.value}
                    type="button"
                    onClick={() => handleSave("max_word_count_english", preset.value)}
                    className={`min-h-[58px] border px-3 py-2 text-left transition-colors ${selected ? "border-accent bg-accent/5" : "border-border bg-surface hover:border-text-muted"}`}
                  >
                    <span className={`block text-[12px] font-medium ${selected ? "text-accent" : "text-text-primary"}`}>{preset.label}</span>
                    <span className="mt-1 block text-[10px] text-text-muted">{preset.description}</span>
                  </button>
                );
              })}
            </div>
            <div className="mt-3 flex items-center gap-3">
              <span className="w-12 text-[11px] text-text-muted">自定义</span>
              <input type="range" min={10} max={40} value={Number(settings.max_word_count_english || 18)}
                onChange={(e) => handleSave("max_word_count_english", parseInt(e.target.value))}
                className="flex-1 accent-accent" />
              <span className="w-14 text-right font-mono text-[12px] text-text-primary">{Number(settings.max_word_count_english || 18)} 词</span>
            </div>
          </SettingsField>
          <SettingsField label="中文标点美化" description="完成中文翻译与断句后，将译文中的中文逗号、句号替换为空格；不调用 LLM，不影响英文原文">
            <button
              onClick={() => handleSave("replace_chinese_punctuation", !settings.replace_chinese_punctuation)}
              className="flex items-center gap-2 group"
            >
              <div className={`w-7 h-[15px] rounded-full transition-all duration-200 relative flex items-center ${
                settings.replace_chinese_punctuation !== false ? "bg-accent/20" : "bg-[rgba(0,0,0,0.06)]"
              }`}>
                <div className={`absolute w-[11px] h-[11px] rounded-full transition-all duration-200 shadow-sm ${
                  settings.replace_chinese_punctuation !== false ? "left-[13px] bg-accent" : "left-[2px] bg-text-muted"
                }`} />
              </div>
              <span className="text-[12px] text-text-muted group-hover:text-text-secondary transition-colors">
                {settings.replace_chinese_punctuation !== false ? "已开启" : "已关闭"}
              </span>
            </button>
          </SettingsField>
          {/* Custom prompt moved to subtitle page */}
        </SettingsSection>
          </>
        )}

        {/* Work directory */}
        {activeSettingsView === "files" && (
        <SettingsSection title="工作目录">
          <SettingsField label="路径" description="视频和字幕文件的默认保存位置">
            <input type="text" value={(settings.work_dir as string) || ""}
              onChange={(e) => setSettings((prev) => ({ ...prev, work_dir: e.target.value }))}
              onBlur={(e) => handleSave("work_dir", e.target.value)}
              placeholder="默认: ~/SubForge/work-dir" className="input-field" />
          </SettingsField>
          <SettingsField label="诊断日志" description="转录或启动失败时可在此目录找到完整错误记录">
            <button
              type="button"
              onClick={async () => {
                try {
                  const result = await openNativeLogsFolder();
                  if (!result.available) {
                    useAppStore.getState().addToast("诊断目录仅在桌面应用中可打开", "info");
                  }
                } catch (error) {
                  useAppStore.getState().setError(
                    error instanceof Error ? error.message : "无法打开诊断目录"
                  );
                }
              }}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-border bg-background px-3 text-[11px] font-semibold text-text-secondary transition-colors hover:border-accent/40 hover:text-accent"
            >
              <Icon icon="solar:folder-open-linear" width={16} />
              打开诊断目录
            </button>
          </SettingsField>
        </SettingsSection>
        )}

            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function SettingsSection({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2.5">
      <div>
        <h3 className="text-[13px] font-semibold text-text-primary">{title}</h3>
        {description && <p className="text-[11px] text-text-muted mt-0.5">{description}</p>}
      </div>
      <div className="space-y-4 rounded-lg border border-border bg-surface p-5">{children}</div>
    </section>
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
