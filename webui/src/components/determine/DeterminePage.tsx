import { createSignal, Show } from "solid-js";
import { ResultCard } from "@/components/determine/ResultCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/field";
import { useDetermineMutation } from "@/hooks/useDetermine";
import { prefsStore, useFallback, useRounds } from "@/store";

/** One-shot feature form. Preferences stay on this device. */
export function DeterminePage() {
  const [query, setQuery] = createSignal("");
  const fallback = useFallback();
  const rounds = useRounds();
  const mutation = useDetermineMutation();

  const submit = (event: SubmitEvent) => {
    event.preventDefault();
    const text = query().trim();
    if (!text) return;
    mutation.mutate({ query: text, fallback: fallback(), rounds: rounds() });
  };

  return (
    <>
      <p class="text-sm text-muted">
        Describe features or a name. Online search builds a shortlist, then Laya picks a winner.
      </p>
      <form onSubmit={submit}>
        <Textarea
          name="query"
          placeholder="e.g. silver hair tsundere genius mage"
          value={query()}
          onInput={(event) => setQuery(event.currentTarget.value)}
        />
        <label class="mt-2 flex items-center gap-2 text-sm text-[#c5c8ce]">
          <input
            type="checkbox"
            checked={fallback()}
            onChange={(event) => prefsStore.getState().setFallback(event.currentTarget.checked)}
          />
          Force keyword fallback (skip Laya)
        </label>
        <label class="mt-2 flex items-center gap-2 text-sm text-[#c5c8ce]">
          Rounds
          <Input
            type="number"
            min={1}
            max={5}
            value={rounds()}
            class="w-16"
            onInput={(event) => {
              const next = Number(event.currentTarget.value);
              if (next >= 1 && next <= 5) prefsStore.getState().setRounds(next);
            }}
          />
        </label>
        <div class="mt-3">
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "Searching…" : "Determine"}
          </Button>
        </div>
      </form>
      <ShowEmpty when={!query().trim() && mutation.isIdle} />
      <Show when={mutation.data && !mutation.data.winner}>
        <Card>
          <p>{mutation.data?.notes || "No candidates."}</p>
        </Card>
      </Show>
      <ResultCard result={mutation.data} />
    </>
  );
}

function ShowEmpty(props: { when: boolean }) {
  return (
    <Card class={props.when ? "" : "hidden"}>
      <p class="text-sm text-muted">Enter some features, then determine.</p>
    </Card>
  );
}
