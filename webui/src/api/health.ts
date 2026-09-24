import { api } from "@/api/client";

export interface HealthStatus {
  status: string;
  laya?: { loaded?: boolean; backend?: string };
  query_llm?: { enabled?: boolean };
}

/** Readiness probe used by the shell, not by a player turn. */
export function fetchHealth(): Promise<HealthStatus> {
  return api.get<HealthStatus>("/healthz");
}
