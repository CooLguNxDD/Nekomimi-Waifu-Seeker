import type { AskedTurn, NekomimiQuestion } from "@/types/game";

export interface AnswerChipData {
  id: string;
  label: string;
  question: string;
}

/** Chips for the live round, kept outside the page so a remount does not drop them. */
const transcripts = new Map<string, AnswerChipData[]>();

/**
 * Rounds the player already left.
 *
 * A snapshot requested before Restart can resolve after the next session id
 * is on screen. Applying it would put the old trail under the new question.
 */
const retired = new Set<string>();

/** Whether ``sessionId`` was cleared and must not accept another trail write. */
export function isTranscriptRetired(sessionId: string): boolean {
  return retired.has(sessionId);
}

/** The trail remembered for ``sessionId``, or an empty list. */
export function loadTranscript(sessionId: string): AnswerChipData[] {
  if (retired.has(sessionId)) return [];
  return transcripts.get(sessionId) ?? [];
}

/** Replace the remembered trail. A retired round stays empty. */
export function saveTranscript(sessionId: string, chips: AnswerChipData[]): void {
  if (!sessionId || retired.has(sessionId)) return;
  transcripts.set(sessionId, chips);
}

/** Drop one round's trail when the player starts over, and ignore its late snapshots. */
export function clearTranscript(sessionId: string): void {
  transcripts.delete(sessionId);
  if (sessionId) retired.add(sessionId);
}

/** Player-facing chip text. Choice snapshots carry ``label``; yes/no stay words. */
export function displayLabel(answer: string, detail?: string | null, label?: string | null): string {
  if (label && label !== answer) return label;
  if (answer === "detail") return (detail || "Detail").trim();
  if (answer === "yes") return "Yes";
  if (answer === "no") return "No";
  return label || answer;
}

/** Short chip text: the answer the player gave, plus a clipped question for context. */
export function chipText(label: string, question: string): string {
  if (!question || question === "Starting hint") return label;
  const clipped = question.length > 42 ? `${question.slice(0, 40)}…` : question;
  return `${label} · ${clipped}`;
}

/** Chips from a state snapshot. The pending question has no answer yet, so it stays off the trail. */
export function chipsFromAsked(seed: string | undefined, asked: AskedTurn[] | undefined): AnswerChipData[] {
  const chips: AnswerChipData[] = [];
  const hint = seed?.trim();
  if (hint) chips.push({ id: "seed", label: hint, question: "Starting hint" });
  for (const [index, turn] of (asked ?? []).entries()) {
    if (!turn.answer) continue;
    chips.push({
      id: turn.qid || `asked-${index}`,
      label: displayLabel(turn.answer, turn.detail, turn.label),
      question: turn.text || "",
    });
  }
  return chips;
}

/** Keep server chips, then any local chips the snapshot does not know about (rejected guesses). */
export function mergeChips(server: AnswerChipData[], local: AnswerChipData[]): AnswerChipData[] {
  const ids = new Set(server.map((chip) => chip.id));
  return [...server, ...local.filter((chip) => !ids.has(chip.id))];
}

/** One chip for an accepted answer. The id is the question, so a retry does not duplicate it. */
export function chipForAnswer(
  question: NekomimiQuestion | undefined,
  answer: string,
  detail?: string,
  label?: string,
): AnswerChipData {
  return {
    id: question?.qid || `answer-${answer}`,
    label: displayLabel(answer, detail, label),
    question: question?.text || "",
  };
}

/** One chip for a rejected guess. It is not an asked turn, so the id is the candidate. */
export function chipForRejection(name: string, candidateId?: string): AnswerChipData {
  return {
    id: `reject:${candidateId || name}`,
    label: `Not ${name}`,
    question: "Rejected guess",
  };
}

/** Append ``chip`` unless that id is already on the trail. */
export function appendChip(chips: AnswerChipData[], chip: AnswerChipData): AnswerChipData[] {
  if (chips.some((item) => item.id === chip.id)) return chips;
  return [...chips, chip];
}
