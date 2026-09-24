import { splitProps, type JSX } from "solid-js";

type ButtonProps = JSX.ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: "primary" | "secondary" | "ghost" | "danger" | "success";
  size?: "sm" | "md" | "lg";
};

const tones: Record<NonNullable<ButtonProps["tone"]>, string> = {
  primary: "bg-accent text-white hover:bg-accent-strong",
  secondary: "bg-surface-raised text-ink-soft hover:brightness-110",
  ghost: "bg-transparent text-ink-soft hover:bg-surface-raised",
  danger: "bg-danger-soft text-danger hover:brightness-110",
  success: "bg-success text-bg hover:brightness-110",
};

const sizes: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm: "min-h-8 px-3 py-1 text-xs",
  md: "min-h-11 px-4 py-2.5 text-sm",
  lg: "min-h-12 px-5 py-3 text-base",
};

/** Shared button. Secondary is the quiet filled action; ghost has no fill. */
export function Button(props: ButtonProps) {
  const [local, rest] = splitProps(props, ["tone", "size", "class", "type"]);
  const tone = () => local.tone ?? "primary";
  const size = () => local.size ?? "md";
  return (
    <button
      type={local.type ?? "button"}
      class={`focus-ring rounded-control font-semibold disabled:cursor-progress disabled:opacity-50 ${tones[tone()]} ${sizes[size()]} ${local.class ?? ""}`}
      {...rest}
    />
  );
}
