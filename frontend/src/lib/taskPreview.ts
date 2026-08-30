import type { SubtitleSegment, TaskInfo } from "@/lib/api";

export interface TaskPreviewUpdate {
  segments: SubtitleSegment[];
  revision: number;
}

/** Merge one task snapshot/delta without depending on React or transport state. */
export function mergeTaskPreview(
  current: SubtitleSegment[],
  task: TaskInfo,
  previousRevision: number
): TaskPreviewUpdate | null {
  if (task.status !== "running" && task.status !== "failed") return null;

  const revision = task.preview_revision || 0;
  if (revision <= previousRevision) return null;

  if (task.preview_segments) {
    return { segments: task.preview_segments, revision };
  }

  const delta = task.preview_delta;
  if (!delta) return null;
  if (delta.mode === "replace") {
    return { segments: delta.segments, revision };
  }
  // Append/patch messages are relative to the immediately preceding snapshot.
  // Let the regular full-snapshot poll recover a gap without advancing revision.
  if (revision !== previousRevision + 1) return null;
  const currentIds = new Set(current.map((segment) => segment.id));
  const deltaIds = new Set(delta.segments.map((segment) => segment.id));
  if (currentIds.size !== current.length || deltaIds.size !== delta.segments.length) return null;
  if (delta.mode === "append") {
    if (current.length + delta.segments.length !== delta.total
      || delta.segments.some((segment) => currentIds.has(segment.id))) return null;
    return { segments: [...current, ...delta.segments], revision };
  }

  if (current.length !== delta.total
    || delta.segments.some((segment) => !currentIds.has(segment.id))) return null;
  const changed = new Map(delta.segments.map((segment) => [segment.id, segment]));
  return {
    segments: current.map((segment) => {
      const patch = changed.get(segment.id);
      return patch ? { ...segment, ...patch } : segment;
    }),
    revision,
  };
}
