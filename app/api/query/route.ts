import { NextResponse } from "next/server";
import { runRagQuery } from "@/graph/run";
import { EvidenceAvailability, PipelineStageTrace } from "@/types/rag";
import { formatCitationResponse } from "@/citations/citationFormatter";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const question = body?.question?.toString().trim();
    if (!question) {
      return NextResponse.json({ error: "Missing 'question'." }, { status: 400 });
    }
    const debug = Boolean(body.debug);
    const result = await runRagQuery(question, debug);

    // Format citation-grounded response
    const chunks = result.retrieval?.chunks ?? [];
    const citationResponse = formatCitationResponse(
      result.answer ?? "",
      chunks,
    );

    return NextResponse.json({
      question,
      answer: citationResponse.cited_answer,
      citations: citationResponse.inline_citations,
      citation_entries: citationResponse.citation_entries,
      citation_footer: citationResponse.citation_footer,
      has_citations: citationResponse.has_citations,
      quality: result.quality as EvidenceAvailability,
      trace: (result.trace ?? []) as PipelineStageTrace[],
      retrieval: {
        reranked: result.retrieval?.reranked ?? false,
        dense_enabled: result.retrieval?.dense_enabled ?? true,
        lexical_enabled: result.retrieval?.lexical_enabled ?? true,
        degradation_reasons: result.retrieval?.degradation_reasons ?? [],
        chunks: chunks.map((item) => ({
          id: item.chunk.id,
          score: item.score,
          text: item.chunk.text,
          metadata: {
            document_id: item.chunk.metadata.document_id,
            source_file: item.chunk.metadata.source_file,
            page: item.chunk.metadata.page,
            section_title: item.chunk.metadata.section_title,
          },
        })),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Query failed." },
      { status: 500 },
    );
  }
}
