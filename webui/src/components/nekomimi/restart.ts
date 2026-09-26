/**
 * Whether restart should ask before discarding the round.
 *
 * A question or a pending guess still has a trail. A finished round does not,
 * so the result screen can start the next empty-seed game in one tap.
 */
export function restartNeedsConfirm(stage: string | undefined): boolean {
  return stage === "asking" || stage === "guessing";
}
