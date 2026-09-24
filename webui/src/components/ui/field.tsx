import type { JSX } from "solid-js";

const field =
  "rounded-[10px] border border-[#333] bg-[#1a1d24] px-3 py-2 text-inherit outline-none";

/** Text field. Scraped content is never passed in as HTML. */
export function Input(props: JSX.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} class={`${field} ${props.class ?? ""}`} />;
}

/** Multi-line feature description for the one-shot form. */
export function Textarea(props: JSX.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} class={`${field} min-h-24 w-full ${props.class ?? ""}`} />;
}
