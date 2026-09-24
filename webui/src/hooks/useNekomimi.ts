import { useMutation, useQuery, useQueryClient } from "@tanstack/solid-query";
import { answerQuestion, fetchState, resolveGuess, startRound } from "@/api/nekomimi";
import { bus } from "@/events/bus";
import { sessionStore } from "@/store";
import { getErrorMessage } from "@/utils/errors";
import type { NekomimiState } from "@/types/game";

export const nekomimiQueryKey = (id: string) => ["nekomimi", id] as const;

function remember(qc: ReturnType<typeof useQueryClient>, data: NekomimiState) {
  if (!data.session_id) return;
  sessionStore.getState().setSessionId(data.session_id);
  qc.setQueryData(nekomimiQueryKey(data.session_id), data);
}

/** Rejoin a round after refresh. Disabled until a session id is stored. */
export function useNekomimiQuery(sessionId: () => string | null) {
  return useQuery(() => {
    const id = sessionId();
    return {
      queryKey: id ? nekomimiQueryKey(id) : ["nekomimi", "none"],
      queryFn: () => fetchState(id!),
      enabled: Boolean(id),
      staleTime: Infinity,
      retry: false,
    };
  });
}

/** Start, answer, and confirm guesses. Each response replaces the cached round. */
export function useNekomimiMutations() {
  const qc = useQueryClient();

  const start = useMutation(() => ({
    mutationFn: (seed: string) => startRound(seed),
    onSuccess: (data: NekomimiState) => remember(qc, data),
    onError: (err: unknown) => bus.emit("toast", { message: getErrorMessage(err, "Could not start") }),
  }));

  const answer = useMutation(() => ({
    mutationFn: (input: { sessionId: string; answer: string; detail?: string }) =>
      answerQuestion(input.sessionId, input.answer, input.detail ?? ""),
    onSuccess: (data: NekomimiState) => remember(qc, data),
    onError: (err: unknown) => bus.emit("toast", { message: getErrorMessage(err, "Could not answer") }),
  }));

  const guess = useMutation(() => ({
    mutationFn: (input: { sessionId: string; correct: boolean }) =>
      resolveGuess(input.sessionId, input.correct),
    onSuccess: (data: NekomimiState, input) => {
      remember(qc, data);
      bus.emit("guess:resolved", { correct: input.correct });
    },
    onError: (err: unknown) => bus.emit("toast", { message: getErrorMessage(err, "Could not send guess") }),
  }));

  return { start, answer, guess };
}
