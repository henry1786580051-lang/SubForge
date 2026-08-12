export type PreviewTaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface PreviewTaskState {
  id: string;
  status: PreviewTaskStatus;
  subtitle_file?: string | null;
  preview_segments?: unknown[] | null;
  preview_delta?: unknown | null;
}

export function shouldLoadLegacyPartial(task: PreviewTaskState): boolean {
  return Boolean(
    task.status === "running" &&
      task.subtitle_file &&
      !task.preview_segments &&
      !task.preview_delta
  );
}

export function shouldApplyLegacyPartial(
  taskId: string,
  currentTaskId: string | null,
  currentTaskStatus: string,
  terminalTaskId: string | null
): boolean {
  return (
    taskId === currentTaskId &&
    currentTaskStatus === "running" &&
    terminalTaskId !== taskId
  );
}

export function isStaleAfterTerminal(
  taskId: string,
  status: PreviewTaskStatus,
  terminalTaskId: string | null
): boolean {
  return (
    terminalTaskId === taskId &&
    (status === "pending" || status === "running")
  );
}
