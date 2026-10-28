type Review = Record<string, unknown>;

function record(value: unknown): Review {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Review : {};
}

function indices(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((x): x is number => Number.isInteger(x) && x > 0) : [];
}

export function taskReviewSummary(result: Record<string, unknown> | null | undefined): string {
  const review = record(result?.quality_review);
  const recovery = record(review.recovery);
  const structure = record(review.structure);
  const messages: string[] = [];
  if (recovery.status === "partial") {
    const missing = indices(recovery.missing_indices);
    const checked = indices(recovery.reviewed_indices);
    const unchecked = indices(recovery.unchecked_indices);
    messages.push(`恢复稿：${missing.length} 条缺译${missing.length ? `（${missing.slice(0, 12).join("、")}${missing.length > 12 ? "…" : ""}）` : ""}，已复核 ${checked.length} 条，未复核 ${unchecked.length} 条`);
  }
  if (structure.status === "needs_review") {
    const unresolved = indices(structure.unresolved_boundaries);
    messages.push(`字幕结构仍需复核：第 ${unresolved.slice(0, 12).join("、")}${unresolved.length > 12 ? "…" : ""} 条附近`);
  }
  return messages.join("；");
}
