import { For, Show } from "solid-js";
import { Button } from "@/components/ui/button";
import type { CharacterCard, NekomimiQuestion } from "@/types/game";

/** Yes/no or choice controls. A typed detail does not answer the current trait. */
export function QuestionCard(props: {
  question: NekomimiQuestion;
  candidatesAlive: number;
  laya: boolean;
  top: CharacterCard[];
  busy: boolean;
  onAnswer: (answer: string, detail?: string) => void;
}) {
  let detailInput: HTMLInputElement | undefined;
  let detail = "";
  const clearDetail = () => {
    detail = "";
    if (detailInput) detailInput.value = "";
  };
  const answer = (value: string, extra?: string) => {
    props.onAnswer(value, extra);
    clearDetail();
  };
  const choice = () => props.question.kind === "choice" && props.question.options;
  const turn = () => props.question.turn ?? 0;
  const max = () => props.question.max_turns ?? 1;

  return (
    <div>
      <p class="text-sm text-muted">
        Question {turn()} of up to {max()} · {props.candidatesAlive || 0} candidates in play ·{" "}
        {props.laya ? "Laya" : "heuristics"}
      </p>
      <div class="mt-2 h-1.5 overflow-hidden rounded-full bg-[#232833]">
        <i
          class="block h-full bg-accent"
          style={{ width: `${Math.min(100, (turn() / max()) * 100)}%` }}
        />
      </div>
      <h2 class="my-3 text-xl font-bold">{props.question.text}</h2>
      <Show when={!props.candidatesAlive}>
        <p id="emptyHint" class="mb-3 rounded-[10px] bg-[#2a2412] px-3 py-2 text-sm text-[#f1d58a]">
          No candidates yet — search needs something specific. Type a series, franchise or name-like
          detail below (e.g. “Vocaloid”, “Final Fantasy”), or keep answering.
        </p>
      </Show>
      <div class="flex flex-wrap gap-2">
        <Show
          when={choice()}
          fallback={
            <>
              <Button disabled={props.busy} onClick={() => answer("yes")}>
                Yes
              </Button>
              <Button tone="ghost" disabled={props.busy} onClick={() => answer("no")}>
                No
              </Button>
            </>
          }
        >
          <For each={props.question.options ?? []}>
            {(opt) => (
              <Button disabled={props.busy} onClick={() => answer(opt.key)}>
                {opt.label}
              </Button>
            )}
          </For>
        </Show>
      </div>
      <div class="mt-3 flex flex-wrap gap-2">
        <input
          ref={detailInput}
          type="text"
          class="min-w-56 flex-1 rounded-[10px] border border-[#333] bg-[#1a1d24] px-3 py-2 outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          placeholder="Add a detail instead (e.g. she pilots a mech)"
          onInput={(event) => {
            detail = event.currentTarget.value;
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && detail.trim() && !props.busy) {
              answer("detail", detail.trim());
            }
          }}
        />
        <Button
          tone="ghost"
          disabled={props.busy}
          onClick={() => {
            const text = detail.trim();
            if (!text) return;
            answer("detail", text);
          }}
        >
          Send detail
        </Button>
      </div>
      <div class="mt-3 text-sm text-[#b9bdc6]">
        <For each={props.top}>
          {(candidate) => (
            <div class="flex justify-between gap-4 py-0.5">
              <span>
                {candidate.name}
                {candidate.series ? ` — ${candidate.series}` : ""}
              </span>
              <span>{Math.round((candidate.probability || 0) * 100)}%</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}
