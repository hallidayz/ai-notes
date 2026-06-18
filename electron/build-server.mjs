/**
 * Bundles server.ts into dist-electron/server.cjs for the Electron main process.
 */

import { build } from "esbuild";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const EXTERNAL = [
  "electron",
  "express",
  "@google/genai",
  "dotenv",
  "vite",
  "tsx",
  "sharp",
  "onnxruntime-node",
  "@xenova/transformers",
];

async function main() {
  await build({
    entryPoints: [path.join(repoRoot, "server.ts")],
    bundle: true,
    platform: "node",
    format: "cjs",
    target: "node20",
    outfile: path.join(repoRoot, "dist-electron", "server.cjs"),
    external: EXTERNAL,
    sourcemap: false,
    logLevel: "info",
    loader: { ".node": "file" },
    banner: {
      js: [
        "const __electronPathToFileURL = require('node:url').pathToFileURL;",
        "const __electronImportMetaUrl = __electronPathToFileURL(__filename).href;",
      ].join("\n"),
    },
    define: {
      "import.meta.url": "__electronImportMetaUrl",
      "process.env.NODE_ENV": JSON.stringify(
        process.env.NODE_ENV || "production"
      ),
    },
  });
  console.log("[electron] server bundle written to dist-electron/server.cjs");
}

main().catch((err) => {
  console.error("[electron] server bundle failed:", err);
  process.exit(1);
});
