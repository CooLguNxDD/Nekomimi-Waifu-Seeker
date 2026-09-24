import { createJSONStorage, persist } from "zustand/middleware";
import { createStore } from "zustand/vanilla";
import { createPrefsSlice, type PrefsSlice } from "@/store/uiSlice";
import { createSessionSlice, type SessionSlice } from "@/store/sessionSlice";
import { useStore } from "@/store/solid";

export const sessionStore = createStore<SessionSlice>()(
  persist(createSessionSlice, {
    name: "nekomimi-session",
    storage: createJSONStorage(() => sessionStorage),
  }),
);

export const prefsStore = createStore<PrefsSlice>()(
  persist(createPrefsSlice, {
    name: "waifu-prefs",
    storage: createJSONStorage(() => localStorage),
  }),
);

export const useSessionId = () => useStore(sessionStore, (s) => s.sessionId);
export const useFallback = () => useStore(prefsStore, (s) => s.fallback);
export const useRounds = () => useStore(prefsStore, (s) => s.rounds);
