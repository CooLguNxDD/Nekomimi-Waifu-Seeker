import { createEffect, createSignal, Show } from "solid-js";
import { portraitFailedAfterIdentityChange, portraitShowsImage } from "@/components/ui/portraitState";

/** Hero plate for a guess or win. Missing or broken URLs become a named silhouette. */
export function Portrait(props: {
  name: string;
  series?: string;
  imageUrl?: string | null;
}) {
  const [failed, setFailed] = createSignal(false);
  // One Portrait stays mounted across guesses. A broken URL must not stick.
  createEffect(() => {
    void props.imageUrl;
    void props.name;
    setFailed(portraitFailedAfterIdentityChange());
  });
  const show = () => portraitShowsImage(props.imageUrl, failed());
  return (
    <div class="overflow-hidden rounded-card border border-border bg-portrait shadow-glow">
      <Show
        when={show()}
        fallback={
          <div class="flex min-h-56 flex-col items-center justify-center gap-2 px-6 py-10 text-center">
            <svg viewBox="0 0 120 90" class="h-24 w-32 text-accent" aria-hidden="true">
              <ellipse cx="60" cy="70" rx="28" ry="16" fill="currentColor" opacity="0.25" />
              <path d="M38 42 L30 16 L50 32 Z" fill="currentColor" />
              <path d="M82 42 L90 16 L70 32 Z" fill="currentColor" />
              <circle cx="60" cy="46" r="22" fill="currentColor" opacity="0.85" />
              <circle cx="52" cy="44" r="2.5" fill="var(--bg)" />
              <circle cx="68" cy="44" r="2.5" fill="var(--bg)" />
            </svg>
            <p class="text-lg font-semibold text-ink">{props.name}</p>
            <Show when={props.series}>
              <p class="text-sm text-muted">{props.series}</p>
            </Show>
          </div>
        }
      >
        <img
          src={props.imageUrl!}
          alt={props.name}
          loading="lazy"
          referrerPolicy="no-referrer"
          class="mx-auto block max-h-96 w-full bg-portrait object-contain"
          onError={() => setFailed(true)}
        />
      </Show>
    </div>
  );
}
