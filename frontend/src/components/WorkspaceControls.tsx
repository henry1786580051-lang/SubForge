"use client";

import { useId, useState } from "react";
import { Icon } from "@/components/Icon";

export function Panel({
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
    <section className={`workspace-panel ${fill ? "flex min-h-0 flex-1 flex-col overflow-hidden" : ""}`}>
      <div className="mb-3.5 flex min-h-8 shrink-0 items-center gap-2.5">
        <span className="flex shrink-0 items-center justify-center text-text-muted">
          <Icon icon={icon} width={18} />
        </span>
        <h2 className="text-[14px] font-semibold text-text-primary">{title}</h2>
      </div>
      <div className={fill ? "min-h-0 flex-1 overflow-hidden" : ""}>{children}</div>
    </section>
  );
}


export function ToggleLine({
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
  const descriptionId = useId();
  return (
    <button
      type="button"
      aria-describedby={description ? descriptionId : undefined}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="toggle-line"
    >
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-text-primary">{label}</span>
        {description && (
          <span id={descriptionId} className="mt-0.5 block text-[12px] leading-5 text-text-muted">{description}</span>
        )}
      </span>
      <span className={`relative h-[22px] w-10 shrink-0 rounded-full transition ${checked ? "bg-accent" : "bg-black/10"}`}>
        <span
          className="toggle-thumb absolute left-[3px] top-[3px] h-4 w-4 rounded-full bg-white shadow-sm"
          style={{ transform: `translateX(${checked ? 18 : 0}px)` }}
        />
      </span>
    </button>
  );
}


export function TaskActionCard({
  currentStage,
  description,
  disabled,
  message,
  onCancel,
  onPrimary,
  primaryLabel,
  progress,
  running,
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
  title: string;
}) {
  const [cancelling, setCancelling] = useState(false);
  const percent = Number.isFinite(progress) ? Math.max(0, Math.min(100, Math.round(progress))) : 0;
  return (
    <section className="task-action" aria-label={title}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold">{title}</h2>
          <p className="mt-1 max-w-[280px] truncate text-[12px] text-text-muted">{description}</p>
        </div>
        {running && <span className="rounded-full bg-background px-2 py-1 font-mono text-[11px]">{percent}%</span>}
      </div>
      {running && (
        <div className="mt-4">
          <div role="progressbar" aria-label={title} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} className="h-1.5 overflow-hidden rounded-full bg-background">
            <div className="task-progress-fill h-full rounded-full bg-accent" style={{ transform: `scaleX(${percent / 100})` }} />
          </div>
          <p role="status" className="mt-2 text-[12px] leading-5 text-text-secondary">{message || currentStage}</p>

        </div>
      )}
      <div className="mt-4 flex items-center gap-2">
        {running ? (
          <button
            disabled={cancelling}
            onClick={async () => { setCancelling(true); try { await onCancel(); } finally { setCancelling(false); } }}
            className="w-full rounded-full border border-border px-4 py-2 text-[13px] font-medium text-text-primary transition hover:bg-surface-hover"
          >
            {cancelling ? "正在取消…" : "取消任务"}
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

