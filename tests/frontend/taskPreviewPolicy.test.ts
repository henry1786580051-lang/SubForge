import assert from "node:assert/strict";
import test from "node:test";

import {
  isStaleAfterTerminal,
  shouldApplyLegacyPartial,
  shouldLoadLegacyPartial,
} from "../../frontend/src/lib/taskPreviewPolicy.ts";

test("structured preview delta disables bilingual partial-file fallback", () => {
  assert.equal(
    shouldLoadLegacyPartial({
      id: "subtitle-1",
      status: "running",
      subtitle_file: "partial_bilingual.srt",
      preview_delta: { mode: "patch", segments: [] },
    }),
    false
  );
});

test("legacy backend without structured preview may still load its partial file", () => {
  assert.equal(
    shouldLoadLegacyPartial({
      id: "subtitle-1",
      status: "running",
      subtitle_file: "partial.srt",
    }),
    true
  );
});

test("late partial response cannot overwrite a completed task result", () => {
  assert.equal(
    shouldApplyLegacyPartial(
      "subtitle-1",
      "subtitle-1",
      "completed",
      "subtitle-1"
    ),
    false
  );
});

test("completed bilingual fields survive a late mixed-column partial response", () => {
  const completed = [
    {
      id: 1,
      text: "The Chinese name for America is 美国.",
      translated: "美国的中文名字叫美国。",
    },
  ];
  const stalePartial = [
    {
      id: 1,
      text: "美国的中文名字叫美国 The Chinese name for America is 美国.",
      translated: "",
    },
  ];
  let editorSegments = completed;

  if (
    shouldApplyLegacyPartial(
      "subtitle-1",
      "subtitle-1",
      "completed",
      "subtitle-1"
    )
  ) {
    editorSegments = stalePartial;
  }

  assert.deepEqual(editorSegments, completed);
});

test("late running update is ignored after the same task completed", () => {
  assert.equal(isStaleAfterTerminal("subtitle-1", "running", "subtitle-1"), true);
  assert.equal(isStaleAfterTerminal("subtitle-1", "completed", "subtitle-1"), false);
});
