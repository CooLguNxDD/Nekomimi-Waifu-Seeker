/** Shimmer block. Pulse stops under prefers-reduced-motion. */
export function Skeleton(props: { class?: string }) {
  return (
    <div
      class={`animate-pulse rounded-control bg-surface-raised motion-reduce:animate-none ${props.class ?? "h-4 w-full"}`}
      aria-hidden="true"
    />
  );
}
