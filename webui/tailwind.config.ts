import type { Config } from "tailwindcss";

export default {
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        border: "var(--border)",
        amber: "var(--amber)",
        ink: "var(--ink)",
        muted: "var(--muted)",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
} satisfies Config;
