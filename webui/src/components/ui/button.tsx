import { splitProps, type JSX } from "solid-js";

type ButtonProps = JSX.ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: "primary" | "ghost";
};

/** Shared button. Ghost is the secondary action on a question. */
export function Button(props: ButtonProps) {
  const [local, rest] = splitProps(props, ["tone", "class", "type"]);
  return (
    <button
      type={local.type ?? "button"}
      class={`rounded-[10px] px-4 py-2.5 text-sm font-semibold disabled:cursor-progress disabled:opacity-50 ${
        local.tone === "ghost" ? "bg-[#232833] text-[#c5c8ce]" : "bg-accent text-white"
      } ${local.class ?? ""}`}
      {...rest}
    />
  );
}
