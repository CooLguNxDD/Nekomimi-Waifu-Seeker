import { createSignal, onCleanup, onMount } from "solid-js";
import { useReducedMotion } from "@/hooks/useReducedMotion";

const TIPS = [
  "Laya is weighing who still fits.",
  "Searching the pool. This beat is the think.",
  "Ears forward. Multi-second turns are normal.",
  "Crossing off the ones that don't match.",
];

/** Cycle a waiting line while a turn is in flight. Reduced motion keeps the first tip still. */
export function useWaitingTip(busy: () => boolean): () => string {
  const reduced = useReducedMotion();
  const [index, setIndex] = createSignal(0);
  onMount(() => {
    const timer = window.setInterval(() => {
      if (!busy() || reduced()) return;
      setIndex((current) => (current + 1) % TIPS.length);
    }, 2800);
    onCleanup(() => window.clearInterval(timer));
  });
  return () => (busy() ? TIPS[index()]! : "");
}
