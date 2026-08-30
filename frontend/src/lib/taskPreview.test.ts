import { describe, expect, it } from "vitest";

import type { SubtitleSegment, TaskInfo } from "@/lib/api";
import { mergeTaskPreview } from "./taskPreview";

const segment = (id: number, translated = ""): SubtitleSegment => ({
  id,
  start: "00:00:00.000",
  end: "00:00:01.000",
  text: `source ${id}`,
  translated,
});

const task = (update: Partial<TaskInfo>): TaskInfo => ({
  id: "task-1",
  type: "subtitle",
  status: "running",
  progress: 50,
  message: "working",
  ...update,
});

describe("mergeTaskPreview", () => {
  it("restores the full preview from a failed task snapshot", () => {
    const recovered = [segment(1, "恢复译文")];
    const result = mergeTaskPreview(
      [],
      task({ status: "failed", preview_revision: 4, preview_segments: recovered }),
      0
    );

    expect(result).toEqual({ segments: recovered, revision: 4 });
  });

  it("applies a translation patch without replacing unchanged cues", () => {
    const current = [segment(1), segment(2)];
    const result = mergeTaskPreview(
      current,
      task({
        preview_revision: 2,
        preview_delta: {
          mode: "patch",
          total: 2,
          segments: [{ id: 2, translated: "第二条" }],
        },
      }),
      1
    );

    expect(result?.segments).toEqual([segment(1), segment(2, "第二条")]);
  });

  it("ignores stale revisions and terminal success snapshots", () => {
    expect(
      mergeTaskPreview([], task({ preview_revision: 1, preview_segments: [segment(1)] }), 1)
    ).toBeNull();
    expect(
      mergeTaskPreview(
        [],
        task({ status: "completed", preview_revision: 2, preview_segments: [segment(1)] }),
        1
      )
    ).toBeNull();
  });

  it("waits for a full snapshot after a missing delta instead of making loss permanent", () => {
    const current = [segment(1), segment(2)];
    const delta = task({ preview_revision: 4, preview_delta: {
      mode: "patch", total: 2, segments: [{ id: 2, translated: "second" }],
    } });
    expect(mergeTaskPreview(current, delta, 2)).toBeNull();
    const snapshot = [segment(1, "first"), segment(2, "second")];
    expect(mergeTaskPreview(current, task({ preview_revision: 4, preview_segments: snapshot }), 2))
      .toEqual({ segments: snapshot, revision: 4 });
  });

  it("rejects an append whose base length or IDs do not match", () => {
    expect(mergeTaskPreview([segment(1)], task({ preview_revision: 2, preview_delta: {
      mode: "append", total: 3, segments: [segment(3)],
    } }), 1)).toBeNull();
    expect(mergeTaskPreview([segment(1)], task({ preview_revision: 2, preview_delta: {
      mode: "append", total: 2, segments: [segment(1)],
    } }), 1)).toBeNull();
  });

  it("rejects a patch for a cue that does not exist in the current snapshot", () => {
    expect(mergeTaskPreview([segment(1)], task({ preview_revision: 2, preview_delta: {
      mode: "patch", total: 1, segments: [{ id: 2, translated: "wrong base" }],
    } }), 1)).toBeNull();
  });
});
