import { createFileRoute } from "@tanstack/solid-router";
import { NekomimiPage } from "@/components/nekomimi/NekomimiPage";

export const Route = createFileRoute("/nekomimi")({
  component: NekomimiPage,
});
