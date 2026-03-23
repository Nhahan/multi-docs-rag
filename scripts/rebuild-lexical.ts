import { readFile } from "node:fs/promises";
import { appConfig } from "../src/lib/config";
import { buildLexicalStore } from "../src/retrieval/lexical";
import type { CorpusChunk } from "../src/types/rag";

(async () => {
  const raw = await readFile(appConfig.data.chunksPath, "utf8");
  const parsed = JSON.parse(raw) as { chunks?: CorpusChunk[]; createdAt?: string };
  const chunks = parsed.chunks ?? [];
  await buildLexicalStore(chunks, appConfig.data.lexicalStorePath);
  console.log("rebuilt lexical", chunks.length);
})();
