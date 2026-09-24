/** Turn progress. The bar width is the fraction of the question cap already used. */
export function Progress(props: { value: number; max: number; label?: string }) {
  const pct = () => {
    const max = props.max || 1;
    return Math.min(100, Math.max(0, (props.value / max) * 100));
  };
  return (
    <div>
      {props.label ? <p class="text-sm text-muted">{props.label}</p> : null}
      <div
        class="mt-2 h-1.5 overflow-hidden rounded-pill bg-surface-raised"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={props.max}
        aria-valuenow={props.value}
        aria-label={props.label}
      >
        <div class="h-full rounded-pill bg-accent" style={{ width: `${pct()}%` }} />
      </div>
    </div>
  );
}
