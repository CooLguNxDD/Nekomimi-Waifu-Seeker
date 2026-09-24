import type { AskedTurn } from "@/types/game";

export interface AnswerChipData {
  id: string;
  label: string;
  question: string;
}

/** Short chip text: the answer the player gave, plus a clipped question for context. */
export function chipText(label: string, question: string): string {
  if (!question || question === "Starting hint") return label;
  const clipped = question.length > 42 ? `${question.slice(0, 40)}…` : question;
  return `${label} · ${clipped}`;
}

/** Chips from a refreshed round. The pending question has no answer yet, so it stays off the trail. */
export function chipsFromAsked(seed: string | undefined, asked: AskedTurn[] | undefined): AnswerChipData[] {
  const chips: AnswerChipData[] = [];
  const hint = seed?.trim();
  if (hint) chips.push({ id: "seed", label: hint, question: "Starting hint" });
  for (const [index, turn] of (asked ?? []).entries()) {
    if (!turn.answer) continue;
    const label =
      turn.answer === "detail"
        ? (turn.detail || "Detail").trim()
        : turn.answer === "yes"
          ? "Yes"
          : turn.answer === "no"
            ? "No"
            : turn.answer;
    chips.push({
      id: turn.qid || `asked-${index}`,
      label,
      question: turn.text || "",
    });
  }
  return chips;
}
