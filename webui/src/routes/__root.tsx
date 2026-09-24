import { Outlet, createRootRoute } from "@tanstack/solid-router";
import { Shell } from "@/components/shell/Shell";

export const Route = createRootRoute({
  component: () => (
    <Shell>
      <Outlet />
    </Shell>
  ),
});
