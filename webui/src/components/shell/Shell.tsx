import { Link, useRouterState } from "@tanstack/solid-router";
import type { JSX } from "solid-js";
import { Toasts } from "@/components/shell/Toasts";
import { useHealthQuery } from "@/hooks/useHealth";

/** App chrome shared by the one-shot page and the guessing game. */
export function Shell(props: { children: JSX.Element }) {
  const health = useHealthQuery();
  const path = useRouterState({ select: (s) => s.location.pathname });
  const laya = () => (health.data?.laya?.loaded ? "Laya loaded" : "Heuristics");

  const wide = () => path() === "/nekomimi";

  return (
    <div class={`mx-auto px-4 py-8 ${wide() ? "max-w-5xl" : "max-w-[720px]"}`}>
      <header class="mb-6 flex items-end justify-between gap-4">
        <div>
          <h1 class="text-2xl font-semibold">Nekomimi-Waifu-Seeker</h1>
          <p class="mt-1 text-sm text-muted">ACG characters. Laya decides; it does not write.</p>
        </div>
        <p class="font-mono text-xs text-muted">{health.isError ? "API offline" : laya()}</p>
      </header>
      <nav class="mb-4 flex gap-3 text-sm">
        <Link to="/" class={`focus-ring rounded-control ${path() === "/" ? "text-amber" : "text-link"}`}>
          Determine
        </Link>
        <Link to="/nekomimi" class={`focus-ring rounded-control ${path() === "/nekomimi" ? "text-amber" : "text-link"}`}>
          Play Nekomimi
        </Link>
      </nav>
      {props.children}
      <Toasts />
    </div>
  );
}
