import { createSignal, onCleanup, onMount } from "solid-js";

/** Whether the player asked the OS to keep motion still. */
export function useReducedMotion(): () => boolean {
  const [reduced, setReduced] = createSignal(false);
  onMount(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    onCleanup(() => media.removeEventListener("change", onChange));
  });
  return reduced;
}
