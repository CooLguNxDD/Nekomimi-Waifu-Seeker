import { createSignal, onCleanup, onMount, Show } from "solid-js";
import { Motion, Presence } from "solid-motionone";

/** Fade a panel in. Reduced motion uses opacity only; otherwise a short slide. */
export function AnimatedCard(props: { show: boolean; children: unknown }) {
  const [reduced, setReduced] = createSignal(false);
  onMount(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    onCleanup(() => media.removeEventListener("change", onChange));
  });

  return (
    <Presence>
      <Show when={props.show}>
        <Motion.div
          initial={reduced() ? { opacity: 0 } : { opacity: 0, y: 8 }}
          animate={reduced() ? { opacity: 1 } : { opacity: 1, y: 0 }}
          exit={reduced() ? { opacity: 0 } : { opacity: 0, y: 8 }}
          transition={{ duration: 0.2, easing: [0.16, 1, 0.3, 1] }}
        >
          {props.children as never}
        </Motion.div>
      </Show>
    </Presence>
  );
}
