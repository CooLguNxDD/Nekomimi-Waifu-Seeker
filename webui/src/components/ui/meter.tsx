/** Confidence meter. Colour shifts as the value climbs so the bar reads as a mood. */
export function Meter(props: { value: number; label?: string }) {
  const pct = () => Math.min(100, Math.max(0, Math.round(props.value * 100)));
  const tone = () => (pct() >= 72 ? "bg-success" : pct() >= 40 ? "bg-amber" : "bg-accent");
  return (
    <div>
      {props.label ? (
        <p class="mb-1 flex justify-between text-xs text-muted">
          <span>{props.label}</span>
          <span>{pct()}%</span>
        </p>
      ) : null}
      <div
        class="h-2 overflow-hidden rounded-pill bg-surface-raised"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct()}
        aria-label={props.label ?? "Confidence"}
      >
        <div class={`h-full rounded-pill ${tone()}`} style={{ width: `${pct()}%` }} />
      </div>
    </div>
  );
}
