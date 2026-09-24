import type { StateCreator } from "zustand";

export interface PrefsSlice {
  fallback: boolean;
  rounds: number;
  setFallback: (fallback: boolean) => void;
  setRounds: (rounds: number) => void;
}

/** Device preferences for the one-shot form. */
export const createPrefsSlice: StateCreator<PrefsSlice> = (set) => ({
  fallback: false,
  rounds: 3,
  setFallback: (fallback) => set({ fallback }),
  setRounds: (rounds) => set({ rounds }),
});
