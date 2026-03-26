import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { appConfig } from "./config";

const modelCache = new Map<string, ChatOllama>();
const embeddingCache = new Map<string, OllamaEmbeddings>();

type ChatModelOverrides = {
  numPredict?: number;
  format?: string | Record<string, any>;
};

const createTimedFetch = (timeoutMs = appConfig.runtime.ollamaRequestTimeoutMs) => {
  return async (input: Parameters<typeof fetch>[0], init: Parameters<typeof fetch>[1] = undefined) => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort(new Error(`Ollama request timeout after ${timeoutMs}ms: ${String(input)}`));
    }, timeoutMs);

    try {
      const response = await fetch(input, {
        ...init,
        signal: controller.signal,
      });
      return response;
    } finally {
      clearTimeout(timer);
    }
  };
};

const timedFetch = createTimedFetch();

export const getChatModel = (
  modelName = appConfig.models.chatModel,
  overrides: ChatModelOverrides = {},
) => {
  const numPredict = overrides.numPredict ?? appConfig.runtime.ollamaChatNumPredict;
  const format = overrides.format;
  const key = `${appConfig.models.baseUrl}|${modelName}|${numPredict}|${typeof format === "string" ? format : JSON.stringify(format ?? null)}`;
  if (!modelCache.has(key)) {
    modelCache.set(
      key,
      new ChatOllama({
        baseUrl: appConfig.models.baseUrl,
        model: modelName,
        numPredict,
        format,
        temperature: appConfig.runtime.ollamaChatTemperature,
        think: appConfig.runtime.ollamaChatThink,
        fetch: timedFetch,
      }),
    );
  }
  return modelCache.get(key)!;
};

export const getEmbeddingModel = (modelName = appConfig.models.embeddingModel) =>
  embeddingCache.get(`${appConfig.models.baseUrl}|${modelName}`) ??
  (() => {
    const model = new OllamaEmbeddings({
      baseUrl: appConfig.models.baseUrl,
      model: modelName,
      fetch: timedFetch,
    });
    embeddingCache.set(`${appConfig.models.baseUrl}|${modelName}`, model);
    return model;
  })();

export const getRerankEmbeddingModel = () => getEmbeddingModel(appConfig.models.rerankModel);
