if (typeof process.loadEnvFile === "function") {
  process.loadEnvFile();
}

import { ingestCorpus } from "../src/ingestion/pipeline";

const run = async () => {
  const result = await ingestCorpus();
  console.log("Ingestion completed:");
  console.log(`- Documents: ${result.totalDocuments}`);
  console.log(`- Chunks: ${result.totalChunks}`);
  console.log(`- Vector index: ${result.vectorStorePath}`);
  console.log(`- Lexical index: ${result.lexicalStorePath}`);
  console.log(`- Chunk metadata: ${result.chunksPath}`);
  if (!result.vectorStoreBuilt) {
    console.log("- WARNING: Dense embedding index was not built. Query will use lexical retrieval fallback.");
    if (result.vectorStoreError) {
      console.log(`- Vector error: ${result.vectorStoreError}`);
    }
  }
};

run().catch((error) => {
  console.error("Ingestion failed:", error instanceof Error ? error.message : error);
  process.exit(1);
});
