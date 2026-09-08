import { hasUnsavedSubtitles } from "../lib/subtitleEdits";
import { beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./appStore";
const first = { id: 1, start: "00:00:00,000", end: "00:00:02,000", text: "Hello", translated: "你好" };
beforeEach(() => { useAppStore.getState().setIsProcessing(false); useAppStore.getState().setSubtitles([{ ...first }]); });
describe("subtitle undo", () => {
  it("restores text and structural edits in order", () => {
    const store = useAppStore.getState();
    store.updateSubtitle(1, "translated", "您好");
    store.commitSubtitles([]);
    store.undoSubtitleEdit();
    expect(useAppStore.getState().subtitles[0].translated).toBe("您好");
    store.undoSubtitleEdit();
    expect(useAppStore.getState().subtitles).toEqual([first]);
  });
  it("does not restore an old document after new results are loaded", () => {
    const store = useAppStore.getState();
    store.commitSubtitles([]);
    store.setSubtitles([{ ...first, text: "New file" }]);
    store.undoSubtitleEdit();
    expect(useAppStore.getState().subtitles[0].text).toBe("New file");
  });
  it("ignores unchanged edits and caps history", () => {
    const store = useAppStore.getState();
    store.updateSubtitle(1, "text", "Hello");
    expect(useAppStore.getState().subtitleHistory).toHaveLength(0);
    for (let i = 0; i < 40; i++) store.updateSubtitle(1, "text", `${i}`);
    expect(useAppStore.getState().subtitleHistory).toHaveLength(30);
  });
});


describe("subtitle saved baseline", () => {
  it("tracks changes and returns to clean after undo", () => {
    const store = useAppStore.getState();
    expect(hasUnsavedSubtitles(store)).toBe(false);
    store.updateSubtitle(1, "text", "Edited");
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(true);
    store.undoSubtitleEdit();
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(false);
  });
  it("does not mark edits made during an asynchronous save as saved", () => {
    useAppStore.getState().updateSubtitle(1, "text", "First edit");
    const saving = useAppStore.getState();
    saving.updateSubtitle(1, "text", "Later edit");
    saving.markSubtitlesSaved(saving.subtitles, saving.subtitleDocumentRevision);
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(true);
    saving.undoSubtitleEdit();
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(false);
  });
  it("ignores a late save after a document reload", () => {
    const saving = useAppStore.getState();
    saving.setSubtitles([{ ...first, text: "Reloaded" }]);
    saving.markSubtitlesSaved(saving.subtitles, saving.subtitleDocumentRevision);
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(false);
    expect(useAppStore.getState().savedSubtitles[0].text).toBe("Reloaded");
  });
  it("recognizes a saved empty document", () => {
    useAppStore.getState().commitSubtitles([]);
    const saving = useAppStore.getState();
    expect(hasUnsavedSubtitles(saving)).toBe(true);
    saving.markSubtitlesSaved(saving.subtitles, saving.subtitleDocumentRevision);
    expect(hasUnsavedSubtitles(useAppStore.getState())).toBe(false);
  });
});


it("protects live task results from structural and text edits", () => {
  const store = useAppStore.getState();
  store.updateSubtitle(1, "text", "Before processing");
  store.setIsProcessing(true);
  store.updateSubtitle(1, "text", "Blocked");
  store.commitSubtitles([]);
  store.undoSubtitleEdit();
  expect(useAppStore.getState().subtitles[0].text).toBe("Before processing");
  store.setIsProcessing(false);
});
