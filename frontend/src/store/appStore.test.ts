import { beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./appStore";
const first = { id: 1, start: "00:00:00,000", end: "00:00:02,000", text: "Hello", translated: "你好" };
beforeEach(() => useAppStore.getState().setSubtitles([{ ...first }]));
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
