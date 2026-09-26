import { createEffect, createSignal, Show } from "solid-js";
import { useQueryClient } from "@tanstack/solid-query";
import { CatHost } from "@/components/nekomimi/CatHost";
import { hostCaption, hostEmotion } from "@/components/nekomimi/emotion";
import { GuessCard } from "@/components/nekomimi/GuessCard";
import { useAnswerTranscript } from "@/components/nekomimi/history";
import { QuestionCard } from "@/components/nekomimi/QuestionCard";
import { restartNeedsConfirm } from "@/components/nekomimi/restart";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/field";
import { Portrait } from "@/components/ui/portrait";
import { Skeleton } from "@/components/ui/skeleton";
import { nekomimiQueryKey, useNekomimiMutations, useNekomimiQuery } from "@/hooks/useNekomimi";
import { useWaitingTip } from "@/hooks/useWaitingTip";
import { prefsStore, sessionStore, useSessionId, useShowPool } from "@/store";
import type { NekomimiState } from "@/types/game";

/** Second tap before a live trail is discarded. A finished round restarts in one tap. */
function RestartControl(props: {
  confirming: boolean;
  busy: boolean;
  onRequest: () => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Show
      when={props.confirming}
      fallback={
        <Button tone="secondary" disabled={props.busy} onClick={() => props.onRequest()}>
          Restart
        </Button>
      }
    >
      <div class="flex flex-wrap items-center justify-end gap-2" role="group" aria-label="Confirm restart">
        <p class="text-sm text-muted">Abandon this round?</p>
        <Button tone="danger" disabled={props.busy} onClick={() => props.onConfirm()}>
          Restart
        </Button>
        <Button tone="ghost" disabled={props.busy} onClick={() => props.onCancel()}>
          Keep playing
        </Button>
      </div>
    </Show>
  );
}

/** Interactive round: the cat asks, the player answers, then a portrait guess. */
export function NekomimiPage() {
  const [seed, setSeed] = createSignal("");
  const [sulking, setSulking] = createSignal(false);
  const [confirming, setConfirming] = createSignal(false);
  const [restarting, setRestarting] = createSignal(false);
  const [leftBehind, setLeftBehind] = createSignal<string | null>(null);
  const sessionId = useSessionId();
  const showPool = useShowPool();
  const round = useNekomimiQuery(sessionId);
  const mutations = useNekomimiMutations();
  const qc = useQueryClient();
  const state = (): NekomimiState | undefined => round.data;
  const transcript = useAnswerTranscript(state, sessionId);
  const busy = () =>
    restarting() || mutations.start.isPending || mutations.answer.isPending || mutations.guess.isPending;
  const tip = useWaitingTip(busy);
  const id = () => state()?.session_id ?? sessionId() ?? "";
  const play = () => Boolean(sessionId()) && Boolean(state()?.stage);
  const live = () => play() || restarting();
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
    if (state()?.stage === "asking" && state()?.question?.qid) setSulking(false);
  });

  // Close the confirm row when the question, guess, or round changes under it.
  let roundStamp = "";
  createEffect(() => {
    const next = `${state()?.session_id ?? ""}|${state()?.stage ?? ""}|${state()?.question?.qid ?? ""}|${state()?.guess?.id ?? ""}`;
    if (roundStamp && next !== roundStamp) setConfirming(false);
    roundStamp = next;
  });

  createEffect(() => {
    if (!restarting()) return;
    const sid = state()?.session_id;
    const previous = leftBehind();
    if (!sid || previous === null || sid === previous) return;
    setRestarting(false);
  });

  /** Open a new empty-seed round. The previous trail is dropped only after the server accepts it. */
  const beginFreshRound = () => {
    if (restarting() || mutations.start.isPending) return;
    const previous = id();
    setConfirming(false);
    setLeftBehind(previous);
    setRestarting(true);
    setSeed("");
    setSulking(false);
    transcript.blank(previous);
    mutations.start.mutate("", {
      onSuccess: (data) => {
        if (!data.session_id || data.session_id === previous) {
          transcript.reopen(previous);
          setRestarting(false);
          setLeftBehind(null);
          return;
        }
        transcript.reset(previous);
        if (previous) qc.removeQueries({ queryKey: nekomimiQueryKey(previous) });
      },
      onError: () => {
        transcript.reopen(previous);
        setRestarting(false);
        setLeftBehind(null);
      },
    });
  };

  /** Ask first while a question or guess is still in play. */
  const requestRestart = () => {
    if (busy()) return;
    if (restartNeedsConfirm(state()?.stage)) {
      setConfirming(true);
      return;
    }
    beginFreshRound();
  };

  return (
    <div class={live() ? "lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:items-start lg:gap-6" : "mx-auto max-w-[720px]"}>
      <div class={`mb-2 lg:sticky lg:top-6 ${live() ? "" : "mb-4"}`}>
        <CatHost emotion={emotion()} line={line()} />
        <Show when={busy()}>
          <div class="mt-3 space-y-2" aria-hidden="true">
            <Skeleton class="h-2 w-full" />
            <Skeleton class="h-2 w-2/3" />
          </div>
        </Show>
      </div>
      <div>
        <div class="mb-4 flex flex-wrap items-start justify-between gap-3">
          <p class="max-w-xl text-sm text-muted">
            Think of a character from an anime, manga, comic, game, movie or TV series. Answer each
            question — yes/no or pick an option — or type a detail to help. Laya decides what to ask next.
          </p>
          <Show when={live()}>
            <RestartControl
              confirming={confirming()}
              busy={busy()}
              onRequest={requestRestart}
              onConfirm={beginFreshRound}
              onCancel={() => setConfirming(false)}
            />
          </Show>
        </div>
        <Show when={restarting()}>
          <Card>
            <p class="text-sm text-muted" role="status">
              Starting a new round…
            </p>
            <Skeleton class="mt-3 h-6 w-2/3" />
            <div class="mt-3 flex gap-2" aria-hidden="true">
              <Skeleton class="h-11 w-24" />
              <Skeleton class="h-11 w-24" />
            </div>
          </Card>
        </Show>
        <Show when={!play() && !restarting()}>
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
        <Show when={!restarting() && play() && state()?.stage === "asking" && state()?.question}>
          <Card>
            <QuestionCard
              question={state()!.question!}
              candidatesAlive={state()?.candidates_alive ?? 0}
              laya={Boolean(state()?.laya)}
              top={state()?.top ?? []}
              history={transcript.chips()}
              showPool={showPool()}
              busy={busy()}
              onTogglePool={() => prefsStore.getState().setShowPool(!showPool())}
              onAnswer={(answer, detail, label) => {
                const question = state()?.question;
                const sid = id();
                mutations.answer.mutate(
                  { sessionId: sid, answer, detail },
                  {
                    onSuccess: () => {
                      setSulking(false);
                      transcript.commitAnswer(sid, question, answer, detail, label);
                    },
                  },
                );
              }}
            />
          </Card>
        </Show>
        <Show when={!restarting() && play() && state()?.stage === "guessing"}>
          <Card>
            <GuessCard
              guess={state()?.guess}
              guessNumber={state()?.guess_number}
              message={state()?.message}
              busy={busy()}
              onResolve={(correct) => {
                const name = state()?.guess?.name;
                const candidateId = state()?.guess?.id;
                const sid = id();
                mutations.guess.mutate(
                  { sessionId: sid, correct },
                  {
                    onSuccess: () => {
                      if (!correct && name) {
                        transcript.commitRejection(sid, name, candidateId);
                        setSulking(true);
                      }
                    },
                  },
                );
              }}
            />
          </Card>
        </Show>
        <Show when={!restarting() && play() && state()?.stage === "done"}>
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
              disabled={busy()}
              onClick={() => {
                const sid = id();
                sessionStore.getState().clearSession();
                transcript.reset(sid);
                setSeed("");
                setSulking(false);
                setConfirming(false);
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
