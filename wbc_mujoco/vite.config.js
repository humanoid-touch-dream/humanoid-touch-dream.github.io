import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "live",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
  },
});
