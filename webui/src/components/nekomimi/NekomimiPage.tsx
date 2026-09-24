import { createSignal, Show } from "solid-js";
import { GuessCard } from "@/components/nekomimi/GuessCard";
import { QuestionCard } from "@/components/nekomimi/QuestionCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/field";
import { useNekomimiMutations, useNekomimiQuery } from "@/hooks/useNekomimi";
import { sessionStore, useSessionId } from "@/store";
import type { NekomimiState } from "@/types/game";

/** Interactive round: the engine asks, the player answers, then a guess. */
export function NekomimiPage() {
  const [seed, setSeed] = createSignal("");
  const sessionId = useSessionId();
  const round = useNekomimiQuery(sessionId);
  const mutations = useNekomimiMutations();
  const state = (): NekomimiState | undefined => round.data;
  const busy = () => mutations.start.isPending || mutations.answer.isPending || mutations.guess.isPending;
  const id = () => state()?.session_id ?? sessionId() ?? "";

  const play = () => Boolean(sessionId()) && Boolean(state()?.stage);

  return (
    <>
      <p class="mb-4 text-sm text-muted">
        Think of a character from an anime, manga, comic, game, movie or TV series. Answer each
        question — yes/no or pick an option — or type a detail to help. Laya decides what to ask next.
      </p>
      <Show when={!play()}>
        <Card>
          <div class="flex flex-wrap gap-2">
            <Input
              class="min-w-56 flex-1"
              placeholder="Optional hint (e.g. sci-fi game, silver hair)"
              value={seed()}
              onInput={(event) => setSeed(event.currentTarget.value)}
            />
            <Button
              disabled={busy()}
              onClick={() => mutations.start.mutate(seed().trim())}
            >
              Start
            </Button>
          </div>
          <p class="mt-2 text-sm text-muted">Leave the hint blank for a cold start.</p>
        </Card>
      </Show>
      <Show when={play() && state()?.stage === "asking" && state()?.question}>
        <Card>
          <QuestionCard
            question={state()!.question!}
            candidatesAlive={state()?.candidates_alive ?? 0}
            laya={Boolean(state()?.laya)}
            top={state()?.top ?? []}
            busy={busy()}
            onAnswer={(answer, detail) => mutations.answer.mutate({ sessionId: id(), answer, detail })}
          />
        </Card>
      </Show>
      <Show when={play() && state()?.stage === "guessing"}>
        <Card>
          <GuessCard
            guess={state()?.guess}
            guessNumber={state()?.guess_number}
            message={state()?.message}
            busy={busy()}
            onResolve={(correct) => mutations.guess.mutate({ sessionId: id(), correct })}
          />
        </Card>
      </Show>
      <Show when={play() && state()?.stage === "done"}>
        <Card>
          <h2 class="text-xl font-bold">
            {state()?.correct && state()?.winner
              ? `Got it: ${state()?.winner?.name}`
              : state()?.message || "Round over."}
          </h2>
          <Show when={state()?.winner?.image_url}>
            <img
              class="my-3 block max-h-80 rounded-[10px]"
              src={state()!.winner!.image_url!}
              alt={state()?.winner?.name ?? ""}
              referrerPolicy="no-referrer"
            />
          </Show>
          <p class="text-sm text-muted">Questions asked: {state()?.turns ?? 0}</p>
          <Button
            class="mt-3"
            onClick={() => {
              sessionStore.getState().clearSession();
              setSeed("");
            }}
          >
            Play again
          </Button>
        </Card>
      </Show>
    </>
  );
}
