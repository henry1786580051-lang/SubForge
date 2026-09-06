"use client";
import { create } from "zustand";
export const useUiStore = create<{
  nativeToolbar: boolean;
  setNativeToolbar: (value: boolean) => void;
  exportRequested: boolean;
  requestExport: () => void;
  consumeExport: () => void;
  appearance: "system" | "light" | "dark";
  setAppearance: (value: "system" | "light" | "dark") => void;
  inspectorOpen: boolean;
  toggleInspector: () => void;
  openRequested: boolean;
  requestOpen: () => void;
  consumeOpen: () => void;
}>((set) => ({
  nativeToolbar: false,
  setNativeToolbar: (nativeToolbar) => set((state) => state.nativeToolbar === nativeToolbar ? state : { nativeToolbar }),
  exportRequested: false,
  requestExport: () => set({ exportRequested: true }),
  consumeExport: () => set({ exportRequested: false }),
  appearance: "system",
  setAppearance: (appearance) => set({ appearance }),
  inspectorOpen: true,
  toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),
  openRequested: false,
  requestOpen: () => set({ openRequested: true }),
  consumeOpen: () => set({ openRequested: false }),
}));
