"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { llmLogsApi, type LlmLogEntry, type LlmLogGroup } from "@/lib/api";
import { useAppStore } from "@/store/appStore";

const fmtDuration = (ms?: number) => {
  if (!ms) return "-";
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}min`;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
};

const fmtNumber = (value?: number) => value ? value.toLocaleString() : "-";

function RequestDetail({ entry }: { entry: LlmLogEntry }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full grid grid-cols-[70px_1fr_90px_80px_24px] items-center gap-2 px-3 py-2 text-left hover:bg-surface-hover transition-colors">
        <span className="text-[10px] font-mono text-text-muted">{entry.batch || "单次"}</span>
        <span className="text-[11px] truncate">{entry.stage || "LLM 请求"}</span>
        <span className="text-[10px] font-mono text-text-muted truncate">{entry.model || "-"}</span>
        <span className="text-[10px] font-mono text-right">{fmtDuration(entry.duration_ms)}</span>
        <span className={`text-[11px] text-center ${entry.error ? "text-red-500" : "text-text-muted"}`}>{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="border-t border-border p-3 space-y-3 bg-background">
          <div className="grid grid-cols-4 gap-2 text-[11px]">
            <span>输入 {fmtNumber(entry.prompt_tokens)}</span>
            <span>输出 {fmtNumber(entry.completion_tokens)}</span>
            <span>推理 {fmtNumber(entry.reasoning_tokens)}</span>
            <span>状态 {entry.error ? "失败" : entry.status || "-"}</span>
          </div>
          {entry.error && <div className="text-[11px] text-red-600 bg-red-50 p-2 rounded-md">{entry.error}</div>}
          {!!entry.request && <LogPayload title="请求参数" value={entry.request} />}
          {!!entry.response && <LogPayload title="响应内容" value={entry.response} />}
        </div>
      )}
    </div>
  );
}

function LogPayload({ title, value }: { title: string; value: unknown }) {
  return (
    <details>
      <summary className="text-[11px] text-text-secondary cursor-pointer">{title}</summary>
      <pre className="mt-2 text-[10px] bg-[rgba(0,0,0,0.03)] rounded-md p-3 overflow-auto max-h-72 font-mono whitespace-pre-wrap break-all">{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function LlmLogsPanel() {
  const { setActiveView } = useAppStore();
  const [groups, setGroups] = useState<LlmLogGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<LlmLogGroup | null>(null);

  const fetchLogs = useCallback(async (nextPage: number, query: string) => {
    setLoading(true);
    try {
      const data = await llmLogsApi.list(nextPage, query);
      setGroups(data.groups);
      setTotal(data.total);
      setPages(data.pages);
      setPage(data.page);
      setSelected((current) => data.groups.some((group) => group.id === current?.id) ? current : null);
    } catch {
      setGroups([]);
      setTotal(0);
      setPages(1);
      setSelected(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => { void fetchLogs(1, search); }, 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchLogs, search]);

  const handleClear = async () => {
    await llmLogsApi.clear();
    setSelected(null);
    await fetchLogs(1, "");
  };

  const handleSelect = async (group: LlmLogGroup) => {
    setSelected({ ...group, entries: [] });
    try {
      const detail = await llmLogsApi.detail(group.id);
      setSelected((current) => current?.id === group.id ? detail : current);
    } catch {
      setSelected((current) => current?.id === group.id ? null : current);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-surface">
        <div>
          <h2 className="text-[13px] font-medium text-text-primary">LLM 任务日志</h2>
          <p className="text-[11px] text-text-muted mt-0.5">每个任务聚合显示，展开后查看批次请求</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleClear} className="px-2.5 py-1 text-[12px] rounded-md text-text-muted hover:text-red-500 hover:bg-red-50 border border-border btn-press">清除日志</button>
          <button onClick={() => setActiveView("workflow")} title="关闭" className="p-1.5 rounded-md text-text-muted hover:bg-surface-hover btn-press">×</button>
        </div>
      </div>

      <div className="px-5 py-3 border-b border-border">
        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索任务、文件、模型或阶段" className="input-field" />
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className={`${selected ? "w-[48%]" : "flex-1"} overflow-auto transition-all`}>
          {loading ? <div className="h-40 grid place-items-center text-[12px] text-text-muted">加载中...</div> : groups.length === 0 ? (
            <div className="h-40 grid place-items-center text-[13px] text-text-muted">暂无日志记录</div>
          ) : (
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-surface z-10"><tr className="border-b border-border text-text-muted">
                <th className="px-3 py-2 text-left font-medium">任务时间</th><th className="px-3 py-2 text-left font-medium">文件</th>
                <th className="px-3 py-2 text-left font-medium">阶段</th><th className="px-3 py-2 text-right font-medium">请求</th>
                <th className="px-3 py-2 text-right font-medium">耗时</th><th className="px-3 py-2 text-right font-medium">Tokens</th>
              </tr></thead>
              <tbody>{groups.map((group) => (
                <tr key={group.id} onClick={() => void handleSelect(group)} className={`border-b border-border cursor-pointer hover:bg-surface-hover ${selected?.id === group.id ? "bg-accent-dim" : ""}`}>
                  <td className="px-3 py-2 font-mono text-[10px] text-text-muted">{group.started_at ? new Date(group.started_at).toLocaleString() : "-"}</td>
                  <td className="px-3 py-2 truncate max-w-[180px]">{group.file_name || "未关联文件"}</td>
                  <td className="px-3 py-2 text-[10px]">{group.stages.join(" · ") || "-"}</td>
                  <td className="px-3 py-2 text-right font-mono">{group.request_count}{group.error_count ? <span className="text-red-500 ml-1">/{group.error_count}错</span> : null}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmtDuration(group.duration_ms)}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmtNumber(group.tokens)}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
          {pages > 1 && <div className="flex items-center justify-center gap-2 py-3 border-t border-border">
            <button onClick={() => void fetchLogs(page - 1, search)} disabled={page <= 1} className="px-2 py-1 text-[12px] rounded border border-border disabled:opacity-30">上一页</button>
            <span className="text-[12px] text-text-muted">{page} / {pages}（{total} 个任务）</span>
            <button onClick={() => void fetchLogs(page + 1, search)} disabled={page >= pages} className="px-2 py-1 text-[12px] rounded border border-border disabled:opacity-30">下一页</button>
          </div>}
        </div>

        {selected && <aside className="w-[52%] border-l border-border overflow-auto p-4 bg-surface">
          <div className="flex items-start justify-between mb-4">
            <div><h3 className="text-[13px] font-medium">{selected.file_name || "LLM 任务详情"}</h3><p className="text-[10px] font-mono text-text-muted mt-1">{selected.task_id || selected.id}</p></div>
            <button onClick={() => setSelected(null)} title="关闭详情" className="text-text-muted p-1">×</button>
          </div>
          <div className="grid grid-cols-4 gap-2 mb-4">
            {[['请求', selected.request_count], ['总耗时', fmtDuration(selected.duration_ms)], ['总 Tokens', fmtNumber(selected.tokens)], ['错误', selected.error_count]].map(([label, value]) => (
              <div key={String(label)} className="border border-border rounded-lg p-2"><div className="text-[10px] text-text-muted">{label}</div><div className="text-[13px] font-mono mt-1">{value}</div></div>
            ))}
          </div>
          <div className="grid grid-cols-3 gap-2 mb-4 text-[11px] text-text-secondary">
            <span>输入 {fmtNumber(selected.prompt_tokens)}</span><span>输出 {fmtNumber(selected.completion_tokens)}</span><span>推理 {fmtNumber(selected.reasoning_tokens)}</span>
          </div>
          <div className="space-y-2">
            {selected.entries.length ? selected.entries.map((entry, index) => <RequestDetail key={`${entry.timestamp || index}-${index}`} entry={entry} />) : <div className="py-8 text-center text-[11px] text-text-muted">正在加载请求明细...</div>}
          </div>
        </aside>}
      </div>
    </div>
  );
}
