import { Show } from "solid-js";
import { Button } from "@/components/ui/button";
import type { CharacterCard } from "@/types/game";

/** Guess confirmation. Names and blurbs are text nodes, never HTML. */
export function GuessCard(props: {
  guess: CharacterCard | null | undefined;
  guessNumber?: number;
  message?: string;
  busy: boolean;
  onResolve: (correct: boolean) => void;
}) {
  return (
    <div>
      <Show
        when={props.guess}
        fallback={<p>{props.message || "No candidates left."}</p>}
      >
        {(guess) => (
          <>
            <h2 class="text-xl font-bold">Is it {guess().name}?</h2>
            <p class="mt-1 text-sm text-muted">
              {[guess().series, guess().medium].filter(Boolean).join(" · ")} —{" "}
              {Math.round((guess().probability || 0) * 100)}% confident (guess {props.guessNumber})
            </p>
            <Show when={guess().image_url}>
              <img
                class="my-3 block max-h-80 max-w-full rounded-[10px] bg-[#0b0d11]"
                src={guess().image_url!}
                alt={guess().name}
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            </Show>
            <Show when={guess().blurb}>
              <p>{guess().blurb}</p>
            </Show>
            <Show when={guess().source_url}>
              <a class="text-[#9db7ff]" href={guess().source_url!} target="_blank" rel="noopener">
                source
              </a>
            </Show>
            <div class="mt-4 flex gap-2">
              <Button disabled={props.busy} onClick={() => props.onResolve(true)}>
                Yes, that's it!
              </Button>
              <Button tone="ghost" disabled={props.busy} onClick={() => props.onResolve(false)}>
                No, keep going
              </Button>
            </div>
          </>
        )}
      </Show>
    </div>
  );
}
