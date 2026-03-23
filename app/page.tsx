"use client";

import { FormEvent, useMemo, useState } from "react";
import { EvidenceAvailability, PipelineStageTrace } from "@/types/rag";

type APIChunk = {
  id: string;
  score: number;
  text: string;
  metadata: {
    document_id: string;
    source_file: string;
    page: number;
    section_title: string | null;
  };
};

type QueryResponse = {
  answer: string;
  citations: string[];
  quality: EvidenceAvailability;
  trace: PipelineStageTrace[];
  retrieval: {
    reranked: boolean;
    dense_enabled: boolean;
    lexical_enabled: boolean;
    degradation_reasons: string[];
    chunks: APIChunk[];
  };
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [quality, setQuality] = useState<EvidenceAvailability | null>(null);
  const [trace, setTrace] = useState<PipelineStageTrace[]>([]);
  const [citations, setCitations] = useState<string[]>([]);
  const [chunks, setChunks] = useState<APIChunk[]>([]);
  const [retrievalSummary, setRetrievalSummary] = useState<{
    dense_enabled: boolean;
    lexical_enabled: boolean;
    degradation_reasons: string[];
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [debug, setDebug] = useState(true);
  const [message, setMessage] = useState("");
  const [ingestBusy, setIngestBusy] = useState(false);

  const documentMix = useMemo(() => {
    const counts = new Map<
      string,
      { documentId: string; sourceFile: string; count: number; bestScore: number; share: number }
    >();

    for (const chunk of chunks) {
      const key = chunk.metadata.document_id;
      const current = counts.get(key);
      if (current) {
        current.count += 1;
        current.bestScore = Math.max(current.bestScore, chunk.score);
        continue;
      }
      counts.set(key, {
        documentId: chunk.metadata.document_id,
        sourceFile: chunk.metadata.source_file,
        count: 1,
        bestScore: chunk.score,
        share: 0,
      });
    }

    return Array.from(counts.values())
      .map((entry) => ({
        ...entry,
        share: chunks.length > 0 ? entry.count / chunks.length : 0,
      }))
      .sort((left, right) => {
      if (right.count !== left.count) return right.count - left.count;
      return right.bestScore - left.bestScore;
      });
  }, [chunks]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setMessage("");
    setChunks([]);
    setCitations([]);
    setQuality(null);
    setTrace([]);
    setRetrievalSummary(null);
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, debug }),
      });
      const payload = (await response.json()) as QueryResponse & { error?: string };
      if (!response.ok || payload.error) {
        setMessage(payload.error ?? "Request failed.");
      } else {
        setAnswer(payload.answer);
        setCitations(payload.citations ?? []);
        setQuality(payload.quality ?? null);
        setTrace(payload.trace ?? []);
        setChunks(payload.retrieval?.chunks ?? []);
        setRetrievalSummary({
          dense_enabled: payload.retrieval?.dense_enabled ?? true,
          lexical_enabled: payload.retrieval?.lexical_enabled ?? true,
          degradation_reasons: payload.retrieval?.degradation_reasons ?? [],
        });
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unknown error");
    }
    setLoading(false);
  };

  const ingest = async () => {
    setIngestBusy(true);
    setMessage("");
    try {
      const response = await fetch("/api/ingest", {
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) {
        setMessage(payload?.error ?? "Ingestion failed");
      } else {
        setMessage(
          `Ingested ${payload.totalChunks} chunks from ${payload.totalDocuments} documents. Vector index refreshed.`,
        );
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unknown error");
    }
    setIngestBusy(false);
  };

  return (
    <div className="container">
      <h1>Multi-Docs RAG (Local Ollama)</h1>
      <p>
        Ask questions against the indexed PDF corpus. The app uses local Qwen3.5:9b
        for generation and Qwen3-Embedding:4b for embeddings and fallback reranking.
      </p>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button type="button" onClick={ingest} disabled={ingestBusy}>
          {ingestBusy ? "Indexing..." : "Ingest PDFs"}
        </button>
        <button type="button" onClick={() => (window.location.href = "/api/ingest")}>
          Open ingest status
        </button>
      </div>

      <form onSubmit={submit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={4}
          cols={40}
          placeholder="Ask a question..."
          style={{ width: "100%", resize: "vertical" }}
          required
        />
        <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 12 }}>
          <button type="submit" disabled={loading || ingestBusy}>
            {loading ? "Thinking..." : "Ask"}
          </button>
          <label>
            <input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} />
            {" "}
            Show retrieval debug
          </label>
        </div>
      </form>

      {retrievalSummary ? (
        <p>
          Retrieval mode:{" "}
          <strong>
            {retrievalSummary.dense_enabled
              ? retrievalSummary.lexical_enabled
                ? "dense+lexical"
                : "dense-only"
              : "lexical-only"}
          </strong>
          {retrievalSummary.dense_enabled && !retrievalSummary.lexical_enabled
            ? " (no lexical overlap for query terms)"
            : ""}
          {retrievalSummary.degradation_reasons.length > 0
            ? ` · ${retrievalSummary.degradation_reasons.join("; ")}`
            : ""}
        </p>
      ) : null}
      {quality ? (
        <p>
          Evidence availability: <strong>{quality.passed ? "available" : "unavailable"}</strong>
          {" "}({quality.candidate_count} chunks, top
          {quality.top_score.toFixed(3)}, avg {quality.avg_score.toFixed(3)})
        </p>
      ) : null}
      {quality && quality.reasons?.length ? (
        <p>
          Availability notes: <span style={{ whiteSpace: "pre-line" }}>{quality.reasons.join("\n")}</span>
        </p>
      ) : null}
      {message ? <p>{message}</p> : null}

      {answer ? (
        <section>
          <h2>Answer</h2>
          <pre style={{ whiteSpace: "pre-wrap", lineHeight: 1.4 }}>{answer}</pre>
          {citations.length ? (
            <p>
              <strong>Citations:</strong> {citations.join(", ")}
            </p>
          ) : null}
        </section>
      ) : null}

      {debug && chunks.length ? (
        <section>
          <h3>Retrieved chunks</h3>
          {documentMix.length ? (
            <div
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                marginBottom: 12,
                padding: 12,
                backgroundColor: "#f8fafc",
              }}
            >
              <p style={{ marginTop: 0, marginBottom: 8 }}>
                <strong>Document mix</strong>
              </p>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {documentMix.map((entry) => (
                  <li key={entry.documentId}>
                    {entry.documentId} ({entry.sourceFile}) · {entry.count} chunks ·{" "}
                    {(entry.share * 100).toFixed(0)}% · best score {entry.bestScore.toFixed(3)}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {chunks.map((chunk) => (
            <div
              key={chunk.id}
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                marginBottom: 12,
                padding: 12,
              }}
            >
              <p style={{ marginTop: 0 }}>
                {chunk.metadata.document_id} · {chunk.metadata.source_file} ·{" "}
                {chunk.metadata.section_title ?? "unlabeled"} · p.{chunk.metadata.page} · score:{" "}
                {chunk.score.toFixed(3)}
              </p>
              <p style={{ color: "#334155" }}>{chunk.text.slice(0, 280)}...</p>
            </div>
          ))}
        </section>
      ) : null}

      {debug && trace.length ? (
        <section>
          <h3>Pipeline trace</h3>
          {trace.map((stage) => (
            <div
              key={`${stage.stage}-${stage.message}`}
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                marginBottom: 12,
                padding: 12,
                backgroundColor: stage.status === "failed" ? "#fef2f2" : stage.status === "degraded" ? "#fffbeb" : "#f8fafc",
              }}
            >
              <p style={{ marginTop: 0 }}>
                <strong>{stage.stage}</strong> · {stage.status}
                {typeof stage.score === "number" ? ` · score ${stage.score.toFixed(3)}` : ""}
              </p>
              <p style={{ marginBottom: 0 }}>{stage.message}</p>
            </div>
          ))}
        </section>
      ) : null}
    </div>
  );
}
