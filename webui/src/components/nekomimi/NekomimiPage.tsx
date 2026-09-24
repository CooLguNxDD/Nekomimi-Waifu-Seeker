import { createEffect, createSignal, Show } from "solid-js";
import { CatHost } from "@/components/nekomimi/CatHost";
import { hostCaption, hostEmotion } from "@/components/nekomimi/emotion";
import { GuessCard } from "@/components/nekomimi/GuessCard";
import { chipsFromAsked, type AnswerChipData } from "@/components/nekomimi/history";
import { QuestionCard } from "@/components/nekomimi/QuestionCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/field";
import { Portrait } from "@/components/ui/portrait";
import { Skeleton } from "@/components/ui/skeleton";
import { useNekomimiMutations, useNekomimiQuery } from "@/hooks/useNekomimi";
import { useWaitingTip } from "@/hooks/useWaitingTip";
import { prefsStore, sessionStore, useSessionId, useShowPool } from "@/store";
import type { NekomimiState } from "@/types/game";

/** Interactive round: the cat asks, the player answers, then a portrait guess. */
export function NekomimiPage() {
  const [seed, setSeed] = createSignal("");
  const [history, setHistory] = createSignal<AnswerChipData[]>([]);
  const [sulking, setSulking] = createSignal(false);
  const [seenSession, setSeenSession] = createSignal<string | null>(null);
  const sessionId = useSessionId();
  const showPool = useShowPool();
  const round = useNekomimiQuery(sessionId);
  const mutations = useNekomimiMutations();
  const state = (): NekomimiState | undefined => round.data;
  const busy = () => mutations.start.isPending || mutations.answer.isPending || mutations.guess.isPending;
  const tip = useWaitingTip(busy);
  const id = () => state()?.session_id ?? sessionId() ?? "";
  const play = () => Boolean(sessionId()) && Boolean(state()?.stage);
  const confidence = () => {
    const current = state();
    if (current?.stage === "guessing") return current.guess?.probability ?? 0;
    return current?.top?.[0]?.probability ?? 0;
  };
  const emotion = () =>
    hostEmotion({
      playing: play(),
      busy: busy(),
      sulking: sulking(),
      stage: state()?.stage,
      correct: state()?.correct,
      confidence: confidence(),
    });
  const line = () => (busy() ? tip() || hostCaption("thinking") : hostCaption(emotion()));

  createEffect(() => {
    const current = state();
    const sid = current?.session_id;
    if (!sid || sid === seenSession()) return;
    setSeenSession(sid);
    setHistory(chipsFromAsked(current?.seed, current?.asked));
    setSulking(false);
  });

  createEffect(() => {
    if (state()?.stage === "asking" && state()?.question?.qid) setSulking(false);
  });

  const remember = (label: string, question: string) => {
    setHistory((chips) => [...chips, { id: `${chips.length}-${label}`, label, question }]);
  };

  return (
    <div class={play() ? "lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:items-start lg:gap-6" : "mx-auto max-w-[720px]"}>
      <div class={`mb-2 lg:sticky lg:top-6 ${play() ? "" : "mb-4"}`}>
        <CatHost emotion={emotion()} line={line()} />
        <Show when={busy()}>
          <div class="mt-3 space-y-2" aria-hidden="true">
            <Skeleton class="h-2 w-full" />
            <Skeleton class="h-2 w-2/3" />
          </div>
        </Show>
      </div>
      <div>
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
              <Button disabled={busy()} onClick={() => mutations.start.mutate(seed().trim())}>
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
              history={history()}
              showPool={showPool()}
              busy={busy()}
              onTogglePool={() => prefsStore.getState().setShowPool(!showPool())}
              onAnswer={(answer, detail, label) => {
                setSulking(false);
                const question = state()?.question?.text ?? "";
                remember(label || answer, question);
                mutations.answer.mutate({ sessionId: id(), answer, detail });
              }}
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
              onResolve={(correct) => {
                if (!correct) {
                  const name = state()?.guess?.name;
                  if (name) remember(`Not ${name}`, "Rejected guess");
                  setSulking(true);
                }
                mutations.guess.mutate({ sessionId: id(), correct });
              }}
            />
          </Card>
        </Show>
        <Show when={play() && state()?.stage === "done"}>
          <Card>
            <Show when={state()?.winner}>
              <Portrait
                name={state()!.winner!.name}
                series={state()?.winner?.series}
                imageUrl={state()?.winner?.image_url}
              />
            </Show>
            <h2 class="mt-4 text-xl font-bold">
              {state()?.correct && state()?.winner
                ? `Got it: ${state()?.winner?.name}`
                : state()?.message || "Round over."}
            </h2>
            <Show when={state()?.winner?.series}>
              <p class="mt-1 text-sm text-muted">{state()?.winner?.series}</p>
            </Show>
            <p class="mt-2 text-sm text-muted">Questions asked: {state()?.turns ?? 0}</p>
            <Button
              class="mt-3"
              onClick={() => {
                sessionStore.getState().clearSession();
                setSeed("");
                setHistory([]);
                setSulking(false);
                setSeenSession(null);
              }}
            >
              Play again
            </Button>
          </Card>
        </Show>
      </div>
    </div>
  );
}
