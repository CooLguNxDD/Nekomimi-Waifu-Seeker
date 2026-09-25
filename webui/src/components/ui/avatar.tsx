import { createSignal, Show } from "solid-js";

/** Round portrait or initial. A failed image falls back to the letter, never a broken icon. */
export function Avatar(props: { name: string; imageUrl?: string | null; size?: "sm" | "md" }) {
  const [failed, setFailed] = createSignal(false);
  const initial = () => (props.name.trim().charAt(0) || "?").toUpperCase();
  const box = () => (props.size === "md" ? "h-10 w-10 text-sm" : "h-7 w-7 text-xs");
  const show = () => Boolean(props.imageUrl) && !failed();
  return (
    <span
      class={`inline-flex shrink-0 items-center justify-center overflow-hidden rounded-pill bg-accent-soft font-semibold text-accent-strong ${box()}`}
      aria-hidden="true"
    >
      <Show
        when={show()}
        fallback={<span>{initial()}</span>}
      >
        <img
          src={props.imageUrl!}
          alt=""
          loading="lazy"
          referrerPolicy="no-referrer"
          class="h-full w-full object-cover"
          onError={() => setFailed(true)}
        />
      </Show>
    </span>
  );
}
