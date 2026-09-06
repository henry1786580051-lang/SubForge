"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "@/components/Icon";

import {
  freeModelsApi,
  tasksApi,
  type FreeModelProbeResult,
  type FreeModelProbeStatus,
  type FreeModelProviderStatus,
  type FreeModelScanResult,
  type TaskInfo,
} from "@/lib/api";
import { groupNvidiaModels } from "@/lib/llmModels";
import { useAppStore } from "@/store/appStore";

type ResultFilter = "available" | "busy" | "all";

const FILTERS: { id: ResultFilter; label: string }[] = [
  { id: "available", label: "可用" },
  { id: "busy", label: "繁忙" },
  { id: "all", label: "全部" },
];

const STATUS_META: Record<
  FreeModelProbeStatus,
  { label: string; className: string; dot: string }
> = {
  available: {
    label: "可用",
    className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    dot: "bg-emerald-500",
  },
  busy: {
    label: "繁忙",
    className: "border-amber-200 bg-amber-50 text-amber-700",
    dot: "bg-amber-500",
  },
  restricted: {
    label: "受限",
    className: "border-orange-200 bg-orange-50 text-orange-700",
    dot: "bg-orange-500",
  },
  incompatible: {
    label: "不兼容",
    className: "border-border bg-background text-text-muted",
    dot: "bg-text-muted",
  },
  unavailable: {
    label: "不可用",
    className: "border-red-200 bg-red-50 text-red-700",
    dot: "bg-red-400",
  },
};

function isScanResult(value: unknown): value is FreeModelScanResult {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<FreeModelScanResult>;
  return candidate.provider === "nvidia" && Array.isArray(candidate.results);
}

function scanDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function durationLabel(durationMs: number): string {
  const seconds = Math.max(0, Math.round(durationMs / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
}

export function FreeModelsPanel() {
  const [provider, setProvider] = useState<FreeModelProviderStatus | null>(null);
  const [result, setResult] = useState<FreeModelScanResult | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [task, setTask] = useState<TaskInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<ResultFilter>("available");
  const [search, setSearch] = useState("");

  const refreshStatus = useCallback(async () => {
    try {
      const status = await freeModelsApi.nvidiaStatus();
      setProvider(status);
      if (status.last_scan) setResult(status.last_scan);
      if (status.active_task_id) setTaskId(status.active_task_id);
    } catch (error) {
      useAppStore
        .getState()
        .setError(error instanceof Error ? error.message : "无法读取免费模型状态");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshStatus(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshStatus]);

  useEffect(() => {
    if (!taskId) return;
    let disposed = false;
    const poll = async () => {
      try {
        const next = await tasksApi.get(taskId);
        if (disposed) return;
        setTask(next);
        if (next.status === "completed") {
          if (isScanResult(next.result)) setResult(next.result);
          setTaskId(null);
          await refreshStatus();
          useAppStore.getState().addToast("NVIDIA 模型扫描完成", "success");
        } else if (next.status === "failed") {
          setTaskId(null);
          useAppStore.getState().setError(next.error || "模型扫描失败");
        } else if (next.status === "cancelled") {
          setTaskId(null);
          useAppStore.getState().addToast("模型扫描已取消", "info");
        }
      } catch (error) {
        if (!disposed) {
          setTaskId(null);
          useAppStore
            .getState()
            .setError(error instanceof Error ? error.message : "无法读取扫描进度");
        }
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 900);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [refreshStatus, taskId]);

  const startScan = useCallback(async () => {
    try {
      const started = await freeModelsApi.scanNvidia();
      setTask({
        id: started.task_id,
        type: "free-model-scan",
        status: "pending",
        progress: 0,
        message: "正在准备扫描",
      });
      setTaskId(started.task_id);
    } catch (error) {
      useAppStore
        .getState()
        .setError(error instanceof Error ? error.message : "无法开始模型扫描");
      await refreshStatus();
    }
  }, [refreshStatus]);

  const cancelScan = useCallback(async () => {
    if (!taskId) return;
    try {
      await tasksApi.cancel(taskId);
    } catch (error) {
      useAppStore
        .getState()
        .setError(error instanceof Error ? error.message : "无法取消模型扫描");
    }
  }, [taskId]);

  const visibleResults = useMemo(() => {
    if (!result) return [];
    const query = search.trim().toLowerCase();
    return result.results.filter((item) => {
      const statusMatches =
        filter === "all" ||
        item.status === filter ||
        (filter === "busy" && item.status === "restricted");
      return statusMatches && (!query || item.id.toLowerCase().includes(query));
    });
  }, [filter, result, search]);

  const groups = useMemo(() => {
    const byId = new Map(visibleResults.map((item) => [item.id, item]));
    return groupNvidiaModels(visibleResults.map((item) => item.id)).map((group) => ({
      ...group,
      results: group.models
        .map((model) => byId.get(model))
        .filter((item): item is FreeModelProbeResult => Boolean(item)),
    }));
  }, [visibleResults]);

  const isScanning = Boolean(taskId);
  const configured = provider?.api_key_configured ?? false;
  const availableCount = result?.counts.available ?? 0;
  const busyCount = result?.counts.busy ?? 0;
  const blockedCount = result
    ? result.counts.restricted + result.counts.incompatible + result.counts.unavailable
    : 0;

  return (
    <main className="h-full overflow-y-auto bg-background">
      <div className="mx-auto max-w-[1320px] px-7 py-7">
        <div className="flex items-end justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
              <Icon icon="solar:radar-2-bold-duotone" width={16} />
              Free API Radar
            </div>
            <h1 className="mt-2 text-[30px] font-semibold leading-tight text-text-primary">免费模型</h1>
            <p className="mt-2 max-w-[680px] text-[13px] leading-6 text-text-secondary">
              检测免费 API 中当前可调用的聊天模型。扫描只发送一个最短探针，不会改变现有翻译配置。
            </p>
          </div>
          {result && (
            <div className="hidden text-right xl:block">
              <p className="text-[11px] text-text-muted">最近扫描</p>
              <p className="mt-1 text-[13px] font-medium text-text-secondary">
                {scanDate(result.scanned_at)} · {durationLabel(result.duration_ms)}
              </p>
            </div>
          )}
        </div>

        <section className="mt-6 overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-5 px-5 py-5">
            <div className="flex min-w-0 items-center gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-700">
                <Icon icon="solar:server-square-cloud-bold-duotone" width={25} />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-[17px] font-semibold text-text-primary">NVIDIA API Catalog</h2>
                  <span className="rounded-full border border-border bg-background px-2.5 py-1 text-[10px] font-medium text-text-muted">
                    首发支持
                  </span>
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-medium ${
                      configured
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-amber-200 bg-amber-50 text-amber-700"
                    }`}
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${configured ? "bg-emerald-500" : "bg-amber-500"}`} />
                    {configured ? "凭据已就绪" : "尚未配置"}
                  </span>
                </div>
                <p className="mt-1.5 truncate text-[12px] text-text-muted">
                  {provider?.base_url || "https://integrate.api.nvidia.com/v1"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2.5">
              {!configured && (
                <button
                  onClick={() => useAppStore.getState().setActiveView("settings")}
                  className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-4 py-2 text-[12px] font-medium text-text-secondary transition hover:border-border-active hover:bg-surface-hover"
                >
                  <Icon icon="solar:settings-linear" width={16} />
                  前往设置
                </button>
              )}
              {isScanning ? (
                <button
                  onClick={() => void cancelScan()}
                  className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-4 py-2 text-[12px] font-medium text-red-700 transition hover:bg-red-100"
                >
                  <Icon icon="solar:stop-circle-linear" width={16} />
                  停止扫描
                </button>
              ) : (
                <button
                  onClick={() => void startScan()}
                  disabled={!configured || loading}
                  className="inline-flex items-center gap-2 rounded-full bg-accent px-4 py-2 text-[12px] font-medium text-white shadow-sm transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Icon icon="solar:radar-2-bold" width={16} />
                  一键测试
                </button>
              )}
            </div>
          </div>

          {isScanning && (
            <div className="border-t border-border bg-background/60 px-5 py-4">
              <div className="flex items-center justify-between gap-4 text-[11px]">
                <span className="font-medium text-text-secondary">{task?.message || "正在扫描模型"}</span>
                <span className="tabular-nums text-text-muted">{task?.progress ?? 0}%</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-border">
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-500"
                  style={{ width: `${task?.progress ?? 0}%` }}
                />
              </div>
            </div>
          )}
        </section>

        {result ? (
          <>
            <section className="mt-5 grid grid-cols-4 divide-x divide-border overflow-hidden rounded-2xl border border-border bg-surface max-md:grid-cols-2 max-md:divide-x-0">
              <Metric label="可用模型" value={availableCount} tone="success" />
              <Metric label="暂时繁忙" value={busyCount} tone="warning" />
              <Metric label="不可调用" value={blockedCount} />
              <Metric label="扫描耗时" value={durationLabel(result.duration_ms)} compact />
            </section>

            <p className="mt-3 text-[11px] leading-5 text-text-muted">
              已实际测试 {result.tested_count} 个模型。目录中的旧模型可能已下线；“繁忙”可稍后重试，“受限”或“不可用”通常无需重复测试。
            </p>

            <section className="mt-5">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
                <div className="flex items-center gap-1 rounded-full border border-border bg-surface p-1">
                  {FILTERS.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => setFilter(item.id)}
                      className={`rounded-full px-3 py-1.5 text-[11px] font-medium transition ${
                        filter === item.id
                          ? "bg-text-primary text-white"
                          : "text-text-muted hover:text-text-secondary"
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
                <label className="flex min-w-[240px] items-center gap-2 rounded-full border border-border bg-surface px-3.5 py-2 text-text-muted focus-within:border-accent">
                  <Icon icon="solar:magnifier-linear" width={15} />
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="搜索模型"
                    className="min-w-0 flex-1 bg-transparent text-[12px] text-text-primary outline-none placeholder:text-text-muted"
                  />
                </label>
              </div>

              {groups.length > 0 ? (
                <div className="divide-y divide-border">
                  {groups.map((group) => (
                    <div key={group.id} className="grid grid-cols-[180px_minmax(0,1fr)] gap-5 py-5 max-md:grid-cols-1">
                      <div>
                        <h3 className="text-[13px] font-semibold text-text-primary">{group.name}</h3>
                        <p className="mt-1 text-[11px] text-text-muted">{group.results.length} 个模型</p>
                      </div>
                      <div className="grid grid-cols-2 gap-2.5 max-lg:grid-cols-1">
                        {group.results.map((model) => (
                          <ModelRow key={model.id} model={model} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex min-h-[260px] flex-col items-center justify-center text-center">
                  <Icon icon="solar:radar-2-linear" width={34} className="text-text-muted" />
                  <p className="mt-3 text-[13px] font-medium text-text-secondary">没有符合条件的模型</p>
                  <p className="mt-1 text-[11px] text-text-muted">切换筛选条件或清除搜索词后再查看</p>
                </div>
              )}
            </section>
          </>
        ) : (
          <section className="mt-5 flex min-h-[330px] flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-surface/55 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-dim text-accent">
              <Icon icon="solar:radar-2-bold-duotone" width={29} />
            </div>
            <h2 className="mt-4 text-[16px] font-semibold text-text-primary">等待首次扫描</h2>
            <p className="mt-2 max-w-[420px] text-[12px] leading-5 text-text-muted">
              模型目录会实时变化。一键测试将验证标准聊天接口，只保留真正返回响应的模型。
            </p>
          </section>
        )}
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
  tone = "default",
  compact = false,
}: {
  label: string;
  value: string | number;
  tone?: "default" | "success" | "warning";
  compact?: boolean;
}) {
  const color =
    tone === "success"
      ? "text-emerald-600"
      : tone === "warning"
        ? "text-amber-600"
        : "text-text-primary";
  return (
    <div className="min-w-0 px-5 py-4 max-md:border-b max-md:border-border">
      <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-text-muted">{label}</p>
      <p className={`mt-2 truncate font-semibold tabular-nums ${compact ? "text-[15px]" : "text-[24px]"} ${color}`}>
        {value}
      </p>
    </div>
  );
}

function ModelRow({ model }: { model: FreeModelProbeResult }) {
  const status = STATUS_META[model.status];
  const copyModel = async () => {
    try {
      await navigator.clipboard.writeText(model.id);
      useAppStore.getState().addToast("模型名称已复制", "success");
    } catch {
      useAppStore.getState().setError("无法复制模型名称");
    }
  };
  return (
    <div className="flex min-w-0 items-center gap-3 rounded-xl border border-border bg-surface px-3.5 py-3 transition hover:border-border-active">
      <span className={`h-2 w-2 shrink-0 rounded-full ${status.dot}`} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12px] font-medium text-text-primary" title={model.id}>{model.id}</p>
        <p className="mt-1 text-[10px] text-text-muted">
          {model.message}{model.http_status ? ` · HTTP ${model.http_status}` : ""}
          {model.status === "available" ? ` · ${model.latency_ms} ms` : ""}
          {model.retryable && model.status !== "available" ? " · 可重试" : ""}
        </p>
      </div>
      <span className={`shrink-0 rounded-full border px-2 py-1 text-[9px] font-medium ${status.className}`}>
        {status.label}
      </span>
      <button
        onClick={() => void copyModel()}
        title="复制模型名称"
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-text-muted transition hover:bg-surface-hover hover:text-text-secondary"
      >
        <Icon icon="solar:copy-linear" width={15} />
      </button>
    </div>
  );
}
