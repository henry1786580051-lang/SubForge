import type { SubtitleSegment } from "./api";

/** Undo may recreate the saved contents without sharing the same array. */
export function hasUnsavedSubtitles(state: { subtitles: SubtitleSegment[]; savedSubtitles: SubtitleSegment[] }): boolean {
  return state.subtitles !== state.savedSubtitles && JSON.stringify(state.subtitles) !== JSON.stringify(state.savedSubtitles);
}
