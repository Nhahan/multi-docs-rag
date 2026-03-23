import { NextResponse } from "next/server";
import { ingestCorpus } from "@/ingestion/pipeline";
import { clearIndexStore } from "@/lib/store";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    message: "POST to trigger ingestion. This endpoint expects local PDFs in data (or configured corpus path).",
  });
}

export async function POST() {
  try {
    const result = await ingestCorpus();
    clearIndexStore();
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Ingestion failed." },
      { status: 500 },
    );
  }
}
