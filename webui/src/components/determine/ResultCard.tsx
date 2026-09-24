import { For, Show } from "solid-js";
import { Card } from "@/components/ui/card";
import { AnimatedCard } from "@/components/motion/AnimatedCard";
import type { CharacterCard, DetermineResult } from "@/types/game";

/** Portrait that never interpolates scraped URLs into HTML strings. */
function Portrait(props: { card: CharacterCard; tall?: boolean }) {
  return (
    <Show
      when={props.card.image_url}
      fallback={
        <div class="flex h-[120px] items-center justify-center text-xs text-[#666]">no image</div>
      }
    >
      <img
        src={props.card.image_url!}
        alt={props.card.name}
        loading="lazy"
        referrerPolicy="no-referrer"
        class={
          props.tall
            ? "mb-3 block max-h-[360px] w-full rounded-[10px] bg-[#0b0d11] object-contain"
            : "block h-[120px] w-full bg-[#0b0d11] object-cover"
        }
      />
    </Show>
  );
}

/** Winner, notes, and runners-up from a determine() payload. */
export function ResultCard(props: { result: DetermineResult | undefined }) {
  const winner = () => props.result?.winner ?? null;
  return (
    <AnimatedCard show={Boolean(winner())}>
      <Show when={winner()}>
        {(card) => (
          <Card>
            <p class="text-sm text-muted">Mode: {props.result?.mode}</p>
            <h2 class="my-1 text-xl font-bold">
              {card().name}{" "}
              <span class="font-medium text-muted">({card().series})</span>
            </h2>
            <p>
              Confidence: <b>{card().confidence}</b>
            </p>
            <Portrait card={card()} tall />
            <p>{card().blurb}</p>
            <Show when={card().source_url}>
              <a class="text-[#9db7ff]" href={card().source_url!} target="_blank" rel="noopener">
                source
              </a>
            </Show>
            <p class="text-sm text-muted">{props.result?.notes}</p>
            <Show when={props.result?.search}>
              <p class="mt-2 font-mono text-xs text-muted">
                DDG online={String(props.result?.search?.online_used)} hits=
                {props.result?.search?.online_count} catalog={props.result?.search?.catalog_size} err=
                {props.result?.search?.online_error}
              </p>
            </Show>
            <Show when={(props.result?.search?.rounds?.length ?? 0) > 0}>
              <ul class="mt-2 list-disc pl-5 text-sm text-muted">
                <For each={props.result?.search?.rounds ?? []}>
                  {(log) => (
                    <li>
                      r{log.round}: {log.query} — raw {log.raw_hits}, new {log.new_candidates}
                      <Show when={log.error}> — {log.error}</Show>
                    </li>
                  )}
                </For>
              </ul>
            </Show>
            <Show when={(props.result?.runners_up?.length ?? 0) > 0}>
              <h3 class="mt-4 font-semibold">Runners-up</h3>
              <div class="mt-2 grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                <For each={props.result?.runners_up ?? []}>
                  {(runner) => (
                    <article class="overflow-hidden rounded-[10px] border border-border bg-[#12151b]">
                      <Portrait card={runner} />
                      <p class="p-2 text-xs">
                        <b>{runner.name}</b>
                        <br />
                        {runner.confidence}
                      </p>
                    </article>
                  )}
                </For>
              </div>
            </Show>
          </Card>
        )}
      </Show>
    </AnimatedCard>
  );
}
