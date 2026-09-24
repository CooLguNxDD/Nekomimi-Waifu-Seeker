import { useQuery } from "@tanstack/solid-query";
import { fetchHealth } from "@/api/health";

export const healthQueryKey = ["health"] as const;

/** Poll readiness so the shell can show whether Laya loaded. */
export function useHealthQuery() {
  return useQuery(() => ({
    queryKey: healthQueryKey,
    queryFn: fetchHealth,
    staleTime: 30_000,
    retry: false,
  }));
}
