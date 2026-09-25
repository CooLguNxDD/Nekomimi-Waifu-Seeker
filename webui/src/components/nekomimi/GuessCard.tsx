import { Show } from "solid-js";
import { Button } from "@/components/ui/button";
import { Meter } from "@/components/ui/meter";
import { Portrait } from "@/components/ui/portrait";
import type { CharacterCard } from "@/types/game";

/** Guess confirmation. The portrait leads; names and blurbs stay text nodes. */
export function GuessCard(props: {
  guess: CharacterCard | null | undefined;
  guessNumber?: number;
  message?: string;
  busy: boolean;
  onResolve: (correct: boolean) => void;
}) {
  return (
    <div>
      <Show when={props.guess} fallback={<p>{props.message || "No candidates left."}</p>}>
        {(guess) => (
          <>
            <Portrait name={guess().name} series={guess().series} imageUrl={guess().image_url} />
            <h2 class="mt-4 text-xl font-bold">Is it {guess().name}?</h2>
            <p class="mt-1 text-sm text-muted">
              {[guess().series, guess().medium].filter(Boolean).join(" · ")}
              {props.guessNumber ? ` · guess ${props.guessNumber}` : ""}
            </p>
            <div class="mt-3">
              <Meter value={guess().probability || 0} label="Confidence" />
            </div>
            <Show when={guess().blurb}>
              <p class="mt-3">{guess().blurb}</p>
            </Show>
            <Show when={guess().source_url}>
              <a
                class="focus-ring mt-2 inline-block rounded-control text-link"
                href={guess().source_url!}
                target="_blank"
                rel="noopener"
              >
                source
              </a>
            </Show>
            <div class="mt-4 flex flex-wrap gap-2">
              <Button tone="success" disabled={props.busy} onClick={() => props.onResolve(true)}>
                Yes, that's it!
              </Button>
              <Button tone="secondary" disabled={props.busy} onClick={() => props.onResolve(false)}>
                No, keep going
              </Button>
            </div>
          </>
        )}
      </Show>
    </div>
  );
}
