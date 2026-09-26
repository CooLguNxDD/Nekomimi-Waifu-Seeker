import { For, Show } from "solid-js";
import { AnswerHistory } from "@/components/nekomimi/AnswerHistory";
import type { AnswerChipData } from "@/components/nekomimi/history";
import { questionHeading } from "@/components/nekomimi/questionHeading";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/field";
import { Avatar } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import type { CharacterCard, NekomimiQuestion } from "@/types/game";

/** Yes/no or choice controls. A typed detail does not answer the current trait. */
export function QuestionCard(props: {
  question: NekomimiQuestion;
  candidatesAlive: number;
  laya: boolean;
  top: CharacterCard[];
  history: AnswerChipData[];
  showPool: boolean;
  busy: boolean;
  onTogglePool: () => void;
  onAnswer: (answer: string, detail?: string, label?: string) => void;
}) {
  let detailInput: HTMLInputElement | undefined;
  let detail = "";
  const clearDetail = () => {
    detail = "";
    if (detailInput) detailInput.value = "";
  };
  const answer = (value: string, extra?: string, label?: string) => {
    props.onAnswer(value, extra, label);
    clearDetail();
  };
  const choice = () => props.question.kind === "choice" && props.question.options;
  const turn = () => props.question.turn ?? 0;
  const max = () => props.question.max_turns ?? 1;
  const heading = () => questionHeading(props.question);
  // Key the prompt by question id. Keying the whole object restarted the
  // enter animation on every payload refresh and could leave the heading
  // stuck on the previous sentence.

  return (
    <div class={props.busy ? "opacity-80" : ""}>
      <Progress
        value={turn()}
        max={max()}
        label={`Question ${turn()} of up to ${max()} · ${props.candidatesAlive || 0} candidates in play · ${props.laya ? "Laya" : "heuristics"}`}
      />
      <AnswerHistory chips={props.history} />
      <div aria-live="polite">
        <Show when={props.question.qid || heading()} keyed>
          {(key) => (
            <h2 class="question-prompt my-3 text-xl font-bold" data-qid={key}>
              {heading()}
            </h2>
          )}
        </Show>
      </div>
      <Show when={!props.candidatesAlive}>
        <p id="emptyHint" class="mb-3 rounded-control bg-amber-soft px-3 py-2 text-sm text-amber-ink">
          No candidates yet — search needs something specific. Type a series, franchise or name-like
          detail below (e.g. “Vocaloid”, “Final Fantasy”), or keep answering.
        </p>
      </Show>
      <div class="sticky bottom-0 z-20 -mx-4 mt-3 border-t border-border bg-surface/95 px-4 py-3 backdrop-blur-md md:static md:mx-0 md:border-0 md:bg-transparent md:p-0 md:backdrop-blur-none">
        <div class="flex max-h-[38vh] flex-wrap gap-2 overflow-y-auto">
          <Show
            when={choice()}
            fallback={
              <>
                <Button disabled={props.busy} onClick={() => answer("yes", undefined, "Yes")}>
                  Yes
                </Button>
                <Button tone="secondary" disabled={props.busy} onClick={() => answer("no", undefined, "No")}>
                  No
                </Button>
              </>
            }
          >
            <For each={props.question.options ?? []}>
              {(opt) => (
                <Button disabled={props.busy} onClick={() => answer(opt.key, undefined, opt.label)}>
                  {opt.label}
                </Button>
              )}
            </For>
          </Show>
        </div>
        <div class="mt-3 flex flex-wrap gap-2">
          <Input
            ref={detailInput}
            type="text"
            class="min-w-56 flex-1"
            placeholder="Add a detail instead (e.g. she pilots a mech)"
            onInput={(event) => {
              detail = event.currentTarget.value;
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && detail.trim() && !props.busy) {
                answer("detail", detail.trim(), detail.trim());
              }
            }}
          />
          <Button
            tone="secondary"
            disabled={props.busy}
            onClick={() => {
              const text = detail.trim();
              if (!text) return;
              answer("detail", text, text);
            }}
          >
            Send detail
          </Button>
        </div>
      </div>
      <div class="mt-4">
        <div class="flex items-center justify-between gap-3">
          <p class="text-xs uppercase tracking-wide text-muted">Candidates</p>
          <Button
            tone="ghost"
            size="sm"
            aria-pressed={props.showPool}
            onClick={() => props.onTogglePool()}
          >
            {props.showPool ? "Hide pool" : "Peek pool"}
          </Button>
        </div>
        <Show
          when={props.showPool}
          fallback={
            <p class="mt-2 text-sm text-muted">
              {props.top.length} names hidden. Peek when you want the odds.
            </p>
          }
        >
          <div class="mt-2 text-sm text-ink-soft">
            <For each={props.top}>
              {(candidate) => (
                <div class="flex items-center justify-between gap-3 py-1">
                  <span class="flex min-w-0 items-center gap-2">
                    <Avatar name={candidate.name} imageUrl={candidate.image_url} />
                    <span class="truncate">
                      {candidate.name}
                      {candidate.series ? ` — ${candidate.series}` : ""}
                    </span>
                  </span>
                  <span class="shrink-0 tabular-nums">{Math.round((candidate.probability || 0) * 100)}%</span>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );
}
