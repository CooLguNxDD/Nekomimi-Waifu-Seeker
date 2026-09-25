export type HostEmotion = "idle" | "thinking" | "curious" | "smug" | "sulk";

/** Pick the cat's face from the round, not from a timer. Busy always thinks, except a fresh miss. */
export function hostEmotion(input: {
  playing: boolean;
  busy: boolean;
  sulking: boolean;
  stage?: string;
  correct?: boolean;
  confidence: number;
}): HostEmotion {
  if (input.sulking) return "sulk";
  if (input.busy) return "thinking";
  if (!input.playing) return "idle";
  if (input.stage === "done") return input.correct ? "smug" : "sulk";
  if (input.stage === "guessing") return input.confidence >= 0.5 ? "smug" : "curious";
  if (input.confidence >= 0.72) return "smug";
  if (input.confidence >= 0.35) return "curious";
  return "idle";
}

/** One line under the cat when nothing is in flight. */
export function hostCaption(emotion: HostEmotion): string {
  switch (emotion) {
    case "thinking":
      return "Thinking…";
    case "curious":
      return "Hmm.";
    case "smug":
      return "I think I know.";
    case "sulk":
      return "…not that one.";
    default:
      return "Ears up.";
  }
}
