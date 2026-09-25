/** Whether the hero plate should show the photo. A failed load is sticky until the identity changes. */
export function portraitShowsImage(imageUrl: string | null | undefined, failed: boolean): boolean {
  return Boolean(imageUrl) && !failed;
}

/** A new URL or name clears a previous load error so the next portrait can try. */
export function portraitFailedAfterIdentityChange(): boolean {
  return false;
}
