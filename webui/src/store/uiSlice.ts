import type { StateCreator } from "zustand";

export interface PrefsSlice {
  fallback: boolean;
  rounds: number;
  showPool: boolean;
  setFallback: (fallback: boolean) => void;
  setRounds: (rounds: number) => void;
  setShowPool: (showPool: boolean) => void;
}

/** Device preferences for the one-shot form and the live candidate list. */
export const createPrefsSlice: StateCreator<PrefsSlice> = (set) => ({
  fallback: false,
  rounds: 3,
  showPool: true,
  setFallback: (fallback) => set({ fallback }),
  setRounds: (rounds) => set({ rounds }),
  setShowPool: (showPool) => set({ showPool }),
});
