import { GraphState } from "../types/rag";
import { ragGraph } from "./graph";

export const runRagQuery = async (question: string, debug = false) => {
  const result = await ragGraph.invoke({ question, debug } as GraphState);
  return result as GraphState;
};
