import type { JSX } from "solid-js";

type ChipProps = {
  children: JSX.Element;
  title?: string;
  tone?: "neutral" | "accent" | "amber";
};

const tones: Record<NonNullable<ChipProps["tone"]>, string> = {
  neutral: "border-border bg-surface-raised text-ink-soft",
  accent: "border-accent/40 bg-accent-soft text-accent-strong",
  amber: "border-amber/30 bg-amber-soft text-amber-ink",
};

/** Compact label for a prior answer or a status. Title holds the full text. */
export function Chip(props: ChipProps) {
  return (
    <span
      title={props.title}
      class={`inline-flex max-w-[16rem] items-center truncate rounded-pill border px-2.5 py-1 text-xs ${tones[props.tone ?? "neutral"]}`}
    >
      {props.children}
    </span>
  );
}
