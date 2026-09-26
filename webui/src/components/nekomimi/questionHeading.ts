/** The sentence shown above the answer buttons. */
export function questionHeading(
  question: { text?: string | null; kind?: string | null } | null | undefined,
): string {
  const text = question?.text?.trim() ?? "";
  if (text) return text;
  // Engine questions always include text. A blank string would still leave the
  // option buttons with nothing to answer.
  if (question?.kind === "choice") return "Pick the closest option.";
  return "Does this describe your character?";
}
