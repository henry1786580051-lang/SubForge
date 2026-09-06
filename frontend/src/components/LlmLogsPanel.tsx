"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { llmLogsApi, openNativeLogsFolder, type LlmLogEntry, type LlmLogGroup } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import { Icon } from "@/components/Icon";

const fmtDuration = (ms?: number) => ms == null ? "—" : ms >= 60_000 ? `${(ms / 60_000).toFixed(1)} 分钟` : ms >= 1000 ? `${(ms / 1000).toFixed(1)} 秒` : `${ms} ms`;
const fmtNumber = (value?: number) => value == null ? "—" : value.toLocaleString();
const fmtDate = (value?: string) => value && !Number.isNaN(Date.parse(value)) ? new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "时间未记录";
const logError = (error: unknown, fallback: string) => error instanceof TypeError ? "无法连接本地服务，请重新打开应用后重试。" : error instanceof Error ? error.message : fallback;

const stageLabel = (stage?: string) => ({ translate: "翻译", optimize: "精校", context: "上下文准备", llm: "模型请求", reflect: "翻译复核" }[stage || ""] || stage || "模型请求");

function RequestDetail({ entry }: { entry: LlmLogEntry }) {
  const [open, setOpen] = useState(false);
  return <details className="request-detail" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>
      <Icon icon="solar:alt-arrow-right-linear" width={15} className="request-chevron" />
      <span className={`request-status ${entry.error ? "has-error" : ""}`}><Icon icon={entry.error ? "solar:danger-circle-linear" : "solar:check-circle-bold"} width={17} /></span>
      <span className="request-title"><strong>{stageLabel(entry.stage)}</strong><small>{entry.batch || "单次请求"} · {entry.model || "模型未记录"}</small></span>
      <span className="request-duration">{fmtDuration(entry.duration_ms)}</span>
    </summary>
    {open && <div className="request-body">
      <dl className="request-metrics"><div><dt>输入</dt><dd>{fmtNumber(entry.prompt_tokens)}</dd></div><div><dt>输出</dt><dd>{fmtNumber(entry.completion_tokens)}</dd></div><div><dt>推理</dt><dd>{fmtNumber(entry.reasoning_tokens)}</dd></div><div><dt>状态</dt><dd>{entry.error ? "失败" : entry.status ?? "未记录"}</dd></div></dl>
      {entry.error && <p className="utility-error">{entry.error}</p>}
      {!!entry.request && <LogPayload title="请求参数" value={entry.request} />}
      {!!entry.response && <LogPayload title="响应内容" value={entry.response} />}
      {!entry.request && !entry.response && <p className="settings-footnote">这条记录未包含请求与响应正文。</p>}
    </div>}
  </details>;
}

function LogPayload({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  return <details className="log-payload" onToggle={(event) => setOpen(event.currentTarget.open)}><summary>{title}</summary>{open && <pre>{JSON.stringify(value, null, 2)}</pre>}</details>;
}

export function LlmLogsPanel() {
  const { setActiveView, addToast } = useAppStore();
  const [groups, setGroups] = useState<LlmLogGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<LlmLogGroup | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const listRequest = useRef(0);
  const detailRequest = useRef(0);
  const invalidateRequests = useCallback(() => { ++listRequest.current; ++detailRequest.current; }, []);

  const fetchLogs = useCallback(async (nextPage: number, query: string) => {
    const request = ++listRequest.current;
    ++detailRequest.current;
    setLoading(true); setError(null); setSelected(null); setDetailLoading(false);
    try {
      const data = await llmLogsApi.list(nextPage, query);
      if (request !== listRequest.current) return;
      setGroups(data.groups); setTotal(data.total); setPages(data.pages); setPage(data.page);
    } catch (err) {
      if (request !== listRequest.current) return;
      setGroups([]); setTotal(0); setPages(1);
      setError(logError(err, "无法读取日志，请稍后重试。"));
    } finally {
      if (request === listRequest.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => { void fetchLogs(1, search); }, 250);
    return () => { clearTimeout(timer); invalidateRequests(); };
  }, [fetchLogs, search, invalidateRequests]);

  const handleClear = async () => {
    setClearing(true);
    try {
      await llmLogsApi.clear();
      setConfirmClear(false);
      await fetchLogs(1, search);
      addToast("日志已清除", "success");
    } catch (err) { addToast(err instanceof Error ? err.message : "清除失败，请重试", "error"); }
    finally { setClearing(false); }
  };

  const handleSelect = async (group: LlmLogGroup) => {
    const request = ++detailRequest.current;
    setSelected(group); setDetailLoading(true); setDetailError(null);
    try {
      const detail = await llmLogsApi.detail(group.id);
      if (request === detailRequest.current) setSelected(detail);
    } catch (err) {
      if (request === detailRequest.current) setDetailError(logError(err, "无法读取请求明细"));
    } finally { if (request === detailRequest.current) setDetailLoading(false); }
  };

  const closeDetail = () => { ++detailRequest.current; setSelected(null); setDetailLoading(false); };
  const openFolder = async () => {
    try {
      const result = await openNativeLogsFolder();
      if (!result.available) addToast("诊断目录仅在桌面应用中可打开", "info");
    } catch (err) { addToast(err instanceof Error ? err.message : "无法打开诊断目录", "error"); }
  };

  return <div className="utility-page diagnostics-page">
    <header className="utility-heading">
      <div><h1>诊断日志</h1><p>查看 LLM 任务的执行记录，定位请求与错误。</p></div>
      <div className="utility-heading-actions">
        <button className="subtle-button" onClick={() => void openFolder()}><Icon icon="solar:folder-open-linear" width={17} />打开日志目录</button>
        <button aria-label="关闭诊断日志" onClick={() => setActiveView("workflow")} className="utility-icon-button"><Icon icon="solar:close-circle-linear" width={21} /></button>
      </div>
    </header>
    <div className="diagnostics-tools glass-surface">
      <label className="diagnostics-search"><Icon icon="solar:magnifier-linear" width={19} /><input type="search" aria-label="搜索日志" value={search} onChange={(event) => { ++listRequest.current; setSearch(event.target.value); }} placeholder="搜索文件、任务、模型或阶段" /></label>
      <span className="diagnostics-count" role="status">{loading ? "正在读取…" : error ? "读取失败" : `${total} 个任务`}</span>
      <button className="utility-icon-button" aria-label="刷新日志" disabled={loading || clearing} onClick={() => void fetchLogs(page, search)}><Icon icon="solar:refresh-linear" width={18} /></button>
      <button className="subtle-button" disabled={clearing || loading} aria-expanded={confirmClear} onClick={() => setConfirmClear(!confirmClear)}>清除日志</button>
    </div>
    {confirmClear && <div className="diagnostics-confirm" role="group" aria-label="确认清除日志"><span>清除全部任务日志？此操作无法撤销。</span><button className="subtle-button" disabled={clearing} onClick={() => setConfirmClear(false)}>取消</button><button className="subtle-button text-red-600" disabled={clearing} onClick={() => void handleClear()}>{clearing ? "正在清除…" : "确认清除"}</button></div>}
    <div className={`diagnostics-layout ${selected ? "has-selection" : ""}`}>
      <section className="diagnostics-list" aria-label="任务日志列表" aria-busy={loading}>
        <div className="diagnostics-list-caption"><span>任务记录</span><span>按最近时间排列</span></div>
        <div className="diagnostics-list-scroll">
          {loading ? <div className="utility-skeleton" role="status" aria-label="正在读取日志">{[1, 2, 3, 4].map((item) => <div key={item}><i /><i /><i /></div>)}</div> : error ? <div className="utility-empty"><Icon icon="solar:danger-circle-linear" width={32} /><h2>日志暂时不可用</h2><p>{error}</p><button className="toolbar-button" onClick={() => void fetchLogs(1, search)}>重新读取</button></div> : groups.length === 0 ? <div className="utility-empty"><Icon icon="solar:document-text-linear" width={34} /><h2>{search ? "没有匹配的任务" : "还没有任务记录"}</h2><p>{search ? "尝试文件名、模型名，或更短的关键词。" : "执行 LLM 翻译或精校后，可以在这里查看请求记录。"}</p>{search && <button className="toolbar-button" onClick={() => setSearch("")}>清空搜索</button>}</div> : groups.map((group) => <button key={group.id} className={`diagnostic-task ${selected?.id === group.id ? "is-selected" : ""}`} aria-pressed={selected?.id === group.id} onClick={() => void handleSelect(group)}>
            <span className="diagnostic-task-top"><time>{fmtDate(group.started_at)}</time><span className={group.error_count ? "log-status has-error" : "log-status"}>{group.error_count ? `${group.error_count} 条错误` : "无请求错误"}</span></span>
            <strong title={group.file_name}>{group.file_name || "未关联文件的任务"}</strong>
            <span className="diagnostic-task-stage">{group.stages.map(stageLabel).join(" · ") || "模型请求"}</span>
            <span className="diagnostic-task-bottom"><span>{group.request_count} 次请求</span><span>{fmtDuration(group.duration_ms)}</span><span>{fmtNumber(group.tokens)} tokens</span><Icon icon="solar:alt-arrow-right-linear" width={15} /></span>
          </button>)}
        </div>
        <footer className="diagnostics-pagination"><span>{total > 0 ? `第 ${page} / ${pages} 页` : "任务记录保存在本机"}</span><div><button aria-label="上一页" disabled={loading || page <= 1} className="utility-icon-button" onClick={() => void fetchLogs(page - 1, search)}><Icon icon="solar:alt-arrow-left-linear" width={17} /></button><button aria-label="下一页" disabled={loading || page >= pages} className="utility-icon-button" onClick={() => void fetchLogs(page + 1, search)}><Icon icon="solar:alt-arrow-right-linear" width={17} /></button></div></footer>
      </section>
      <aside className="diagnostics-detail" aria-label="任务详情">
        {selected ? <div key={selected.id} className="diagnostics-detail-content">
          <header className="diagnostic-detail-heading"><div><span className="detail-eyebrow">任务详情</span><h2>{selected.file_name || "未关联文件的任务"}</h2><p>{fmtDate(selected.started_at)} · {selected.models.join(" · ") || "模型未记录"}</p></div><button className="utility-icon-button" aria-label="关闭任务详情" onClick={closeDetail}><Icon icon="solar:close-circle-linear" width={20} /></button></header>
          <dl className="diagnostic-metrics">{[["请求", `${selected.request_count} 次`], ["总耗时", fmtDuration(selected.duration_ms)], ["Tokens", fmtNumber(selected.tokens)], ["错误", selected.error_count]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          <div className="diagnostic-token-breakdown"><span>输入 {fmtNumber(selected.prompt_tokens)}</span><span>输出 {fmtNumber(selected.completion_tokens)}</span><span>推理 {fmtNumber(selected.reasoning_tokens)}</span></div>
          <div className="diagnostic-requests-heading"><h3>请求明细</h3><span>展开查看参数与响应</span></div>
          {detailLoading ? <p className="detail-loading" role="status">正在读取请求明细…</p> : detailError ? <div className="utility-error" role="alert"><p>{detailError}</p><button className="subtle-button" onClick={() => void handleSelect(selected)}>重试</button></div> : selected.entries.length ? selected.entries.map((entry, index) => <RequestDetail key={`${entry.timestamp}-${index}`} entry={entry} />) : <div className="utility-empty compact"><h3>没有请求明细</h3><p>这项任务仅保留了汇总信息。</p></div>}
          <details className="diagnostic-identifiers"><summary>任务标识</summary><code>{selected.task_id || selected.id}</code></details>
        </div> : <div className="utility-empty diagnostic-intro"><span className="diagnostic-intro-symbol"><Icon icon="solar:document-text-linear" width={36} /></span><h2>每一次处理，都有迹可循</h2><p>选择左侧任务，查看请求明细、用量和错误信息。</p><div className="diagnostic-intro-guide"><span>任务概览</span><Icon icon="solar:alt-arrow-right-linear" width={15} /><span>请求明细</span><Icon icon="solar:alt-arrow-right-linear" width={15} /><span>参数与响应</span></div></div>}
      </aside>
    </div>
  </div>;
}
