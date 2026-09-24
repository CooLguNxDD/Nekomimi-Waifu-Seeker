import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    tanstackRouter({ target: "solid", autoCodeSplitting: true }),
    solid(),
    tailwindcss(),
  ],
  resolve: { alias: { "@": path.resolve(root, "./src") } },
  server: {
    proxy: {
      "/api/": "http://127.0.0.1:7860",
      "/healthz": "http://127.0.0.1:7860",
    },
  },
  build: { outDir: "../waifu_engine/webui_dist", emptyOutDir: true },
});
