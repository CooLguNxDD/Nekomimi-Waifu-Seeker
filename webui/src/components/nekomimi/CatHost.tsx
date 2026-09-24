import { Show } from "solid-js";
import type { HostEmotion } from "@/components/nekomimi/emotion";

/** SVG cat host. Emotion swaps the face; the pulse class is only the thinking beat. */
export function CatHost(props: { emotion: HostEmotion; line: string }) {
  const motion = () =>
    props.emotion === "thinking" ? "host-thinking" : props.emotion === "idle" ? "host-idle" : "";
  const tilt = () => (props.emotion === "curious" ? "rotate-6" : props.emotion === "sulk" ? "-rotate-3" : "");
  return (
    <aside class="flex items-center gap-3 lg:flex-col lg:items-center lg:text-center" aria-live="polite">
      <div class={`w-28 shrink-0 lg:w-40 ${motion()} ${tilt()}`}>
        <svg viewBox="0 0 160 168" class="h-auto w-full" role="img" aria-label={`Host is ${props.emotion}`}>
          <ellipse cx="80" cy="150" rx="36" ry="12" fill="var(--accent)" opacity="0.2" />
          <path
            d={props.emotion === "sulk" ? "M46 58 L28 28 L58 46 Z" : "M48 62 L22 22 L62 42 Z"}
            fill="var(--ink)"
          />
          <path
            d={props.emotion === "sulk" ? "M114 58 L132 28 L102 46 Z" : "M112 62 L138 22 L98 42 Z"}
            fill="var(--ink)"
          />
          <path
            d={props.emotion === "sulk" ? "M50 54 L36 32 L58 46 Z" : "M52 56 L32 30 L60 44 Z"}
            fill="var(--accent)"
          />
          <path
            d={props.emotion === "sulk" ? "M110 54 L124 32 L102 46 Z" : "M108 56 L128 30 L100 44 Z"}
            fill="var(--accent)"
          />
          <circle cx="80" cy="78" r="40" fill="var(--surface-raised)" stroke="var(--ink)" stroke-width="3" />
          <Show
            when={props.emotion === "smug" || props.emotion === "sulk"}
            fallback={
              <>
                <ellipse cx="66" cy="76" rx={props.emotion === "curious" ? 7 : 5} ry={props.emotion === "thinking" ? 2 : 6} fill="var(--ink)" />
                <ellipse cx="96" cy="76" rx={props.emotion === "curious" ? 7 : 5} ry={props.emotion === "thinking" ? 2 : 6} fill="var(--ink)" />
              </>
            }
          >
            <path
              d={props.emotion === "sulk" ? "M58 80 Q66 74 74 80" : "M58 80 Q66 70 74 80"}
              fill="none"
              stroke="var(--ink)"
              stroke-width="3"
              stroke-linecap="round"
            />
            <path
              d={props.emotion === "sulk" ? "M88 80 Q96 74 104 80" : "M88 80 Q96 70 104 80"}
              fill="none"
              stroke="var(--ink)"
              stroke-width="3"
              stroke-linecap="round"
            />
          </Show>
          <path d="M80 84 L74 92 L86 92 Z" fill="var(--accent)" />
          <path
            d={
              props.emotion === "sulk"
                ? "M70 104 Q80 96 90 104"
                : props.emotion === "smug"
                  ? "M66 98 Q80 112 94 98"
                  : "M70 100 Q80 108 90 100"
            }
            fill="none"
            stroke="var(--ink)"
            stroke-width="2.5"
            stroke-linecap="round"
          />
          <Show when={props.emotion === "smug"}>
            <ellipse cx="52" cy="92" rx="6" ry="3" fill="var(--danger)" opacity="0.55" />
            <ellipse cx="108" cy="92" rx="6" ry="3" fill="var(--danger)" opacity="0.55" />
          </Show>
          <path
            d="M118 96 Q148 110 128 140"
            fill="none"
            stroke="var(--ink)"
            stroke-width="4"
            stroke-linecap="round"
          />
        </svg>
      </div>
      <p class="text-sm text-ink-soft">{props.line}</p>
    </aside>
  );
}
