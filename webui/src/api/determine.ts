import { api } from "@/api/client";
import type { DetermineResult } from "@/types/game";

export interface DetermineInput {
  query: string;
  fallback: boolean;
  rounds: number;
}

/** Run the one-shot feature matcher. */
export function determine(input: DetermineInput): Promise<DetermineResult> {
  return api.post<DetermineResult>("/api/determine", {
    query: input.query,
    fallback: input.fallback,
    online: true,
    rounds: input.rounds,
  });
}
