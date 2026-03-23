import { START, END, StateGraph, Annotation } from "@langchain/langgraph";
import { evidenceGateNode, generateNode, retrieveNode, verifyNode } from "./nodes";
import { GraphState, RetrievalResult, EvidenceAvailability, PipelineStageTrace } from "../types/rag";

const GraphStateShape = Annotation.Root({
  question: Annotation<string>({
    reducer: (_, next) => next,
    default: () => "",
  }),
  retrieval: Annotation<RetrievalResult | undefined>({
    reducer: (_, next) => next,
    default: () => undefined,
  }),
  answer: Annotation<string | undefined>({
    reducer: (_, next) => next,
    default: () => undefined,
  }),
  citations: Annotation<string[]>({
    reducer: (_, next) => next ?? [],
    default: () => [],
  }),
  quality: Annotation<EvidenceAvailability | null | undefined>({
    reducer: (_, next) => next ?? undefined,
    default: () => undefined,
  }),
  trace: Annotation<PipelineStageTrace[] | undefined>({
    reducer: (_, next) => next ?? [],
    default: () => [],
  }),
  debug: Annotation<boolean>({
    reducer: (_, next) => next ?? false,
    default: () => false,
  }),
});

const graph = new StateGraph(GraphStateShape);

graph.addNode("retrieve", retrieveNode);
graph.addNode("evidence_gate", evidenceGateNode);
graph.addNode("generate", generateNode);
graph.addNode("verify", verifyNode);

graph.addEdge(START, "retrieve");
graph.addEdge("retrieve", "evidence_gate");
graph.addEdge("evidence_gate", "generate");
graph.addEdge("generate", "verify");
graph.addEdge("verify", END);

export const ragGraph = graph.compile();
