import { api } from "@/api/client";
import type { NekomimiState } from "@/types/game";

/** Open a guessing round. An empty seed is a cold start. */
export function startRound(seed: string): Promise<NekomimiState> {
  return api.post<NekomimiState>("/api/nekomimi/start", { seed });
}

/** Send a yes/no, choice key, or free-text detail for the current question. */
export function answerQuestion(
  sessionId: string,
  answer: string,
  detail = "",
): Promise<NekomimiState> {
  return api.post<NekomimiState>("/api/nekomimi/answer", {
    session_id: sessionId,
    answer,
    detail,
  });
}

/** Tell the engine whether the current guess is the character. */
export function resolveGuess(sessionId: string, correct: boolean): Promise<NekomimiState> {
  return api.post<NekomimiState>("/api/nekomimi/guess", {
    session_id: sessionId,
    correct,
  });
}

/** Reload a live round after a refresh. */
export function fetchState(sessionId: string): Promise<NekomimiState> {
  return api.get<NekomimiState>(`/api/nekomimi/state/${encodeURIComponent(sessionId)}`);
}
