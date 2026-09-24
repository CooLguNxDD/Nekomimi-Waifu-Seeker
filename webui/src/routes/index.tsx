import { createFileRoute } from "@tanstack/solid-router";
import { DeterminePage } from "@/components/determine/DeterminePage";

export const Route = createFileRoute("/")({
  component: DeterminePage,
});
