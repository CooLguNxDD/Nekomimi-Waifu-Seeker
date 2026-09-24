import type { StateCreator } from "zustand";

export interface SessionSlice {
  sessionId: string | null;
  setSessionId: (sessionId: string | null) => void;
  clearSession: () => void;
}

/** Remember the live Nekomimi round so a refresh can rejoin it. */
export const createSessionSlice: StateCreator<SessionSlice> = (set) => ({
  sessionId: null,
  setSessionId: (sessionId) => set({ sessionId }),
  clearSession: () => set({ sessionId: null }),
});
