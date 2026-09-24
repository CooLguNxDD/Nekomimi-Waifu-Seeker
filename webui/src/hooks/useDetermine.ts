import { useMutation, useQueryClient } from "@tanstack/solid-query";
import { determine, type DetermineInput } from "@/api/determine";
import { bus } from "@/events/bus";
import { getErrorMessage } from "@/utils/errors";
import type { DetermineResult } from "@/types/game";

export const determineQueryKey = ["determine"] as const;

/** One-shot search. The result lives in the query cache, not in Zustand. */
export function useDetermineMutation() {
  const qc = useQueryClient();
  return useMutation(() => ({
    mutationFn: (input: DetermineInput) => determine(input),
    onSuccess: (data: DetermineResult) => {
      qc.setQueryData(determineQueryKey, data);
    },
    onError: (err: unknown) => {
      bus.emit("toast", { message: getErrorMessage(err, "Determine failed") });
    },
  }));
}
