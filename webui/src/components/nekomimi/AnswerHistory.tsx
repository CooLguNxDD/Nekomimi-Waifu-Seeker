import { For } from "solid-js";
import { chipText, type AnswerChipData } from "@/components/nekomimi/history";
import { Chip } from "@/components/ui/chip";

/** Trail of what this round already heard. Empty until the first hint or answer. */
export function AnswerHistory(props: { chips: AnswerChipData[] }) {
  return (
    <div class="mt-3 flex gap-2 overflow-x-auto pb-1" aria-label="Answers so far">
      <For each={props.chips}>
        {(chip) => (
          <Chip title={chip.question ? `${chip.question} — ${chip.label}` : chip.label}>
            {chipText(chip.label, chip.question)}
          </Chip>
        )}
      </For>
    </div>
  );
}
