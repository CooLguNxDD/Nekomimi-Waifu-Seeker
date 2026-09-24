import { createSignal, onCleanup, onMount, Show } from "solid-js";
import { Motion, Presence } from "solid-motionone";

/** Expand or fade a panel. Reduced motion skips the height tween. */
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
          initial={reduced() ? { opacity: 0 } : { height: 0, opacity: 0 }}
          animate={reduced() ? { opacity: 1 } : { height: "auto", opacity: 1 }}
          exit={reduced() ? { opacity: 0 } : { height: 0, opacity: 0 }}
          transition={{ duration: 0.2, easing: [0.16, 1, 0.3, 1] }}
          class="overflow-hidden"
        >
          {props.children as never}
        </Motion.div>
      </Show>
    </Presence>
  );
}
