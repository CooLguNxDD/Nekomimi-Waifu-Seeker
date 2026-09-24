import type { JSX } from "solid-js";

/** Surface card used by both modes. */
export function Card(props: { class?: string; children: JSX.Element; id?: string }) {
  return (
    <section
      id={props.id}
      class={`mt-4 rounded-xl border border-border bg-surface p-4 ${props.class ?? ""}`}
    >
      {props.children}
    </section>
  );
}
