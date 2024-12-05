import { convertToCoreMessages, streamText as _streamText } from 'ai';
import { getModel } from '~/lib/.server/llm/model';
import { MAX_TOKENS } from './constants';
import { getSystemPrompt } from './prompts';
import { DEFAULT_MODEL, DEFAULT_PROVIDER, MODEL_LIST, MODEL_REGEX, PROVIDER_REGEX } from '~/utils/constants';
import type { LanguageModelV1 } from 'ai';
import type { ProviderInfo } from '~/types/model';

interface ToolResult<Name extends string, Args, Result> {
  toolCallId: string;
  toolName: Name;
  args: Args;
  result: Result;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  toolInvocations?: ToolResult<string, unknown, unknown>[];
  model?: string;
}

interface ToolInvocation extends ToolResult<string, unknown, unknown> {
  state: 'result';
}

interface ProcessedMessage extends Omit<Message, 'toolInvocations'> {
  toolInvocations?: ToolInvocation[];
}

export type Messages = Message[];

type BaseAIOptions = Parameters<typeof _streamText>[0];

interface BaseStreamingOptions extends Omit<BaseAIOptions, 'model' | 'messages' | 'maxTokens' | 'system'> {
  toolChoice?: 'none' | 'auto';
}

export interface StreamingOptions extends BaseStreamingOptions {
  apiKeys?: Record<string, string>;
  model?: string;
}

type Provider = ProviderInfo['name'];

function extractPropertiesFromMessage(message: Message): { model: string; provider: Provider; content: string } {
  const textContent = Array.isArray(message.content)
    ? message.content.find((item) => item.type === 'text')?.text || ''
    : message.content;

  const modelMatch = textContent.match(MODEL_REGEX);
  const providerMatch = textContent.match(PROVIDER_REGEX);

  const model = modelMatch ? modelMatch[1] : DEFAULT_MODEL;
  const provider = (providerMatch ? providerMatch[1] : DEFAULT_PROVIDER) as Provider;

  const cleanedContent = Array.isArray(message.content)
    ? message.content.map((item) => {
        if (item.type === 'text') {
          return {
            ...item,
            text: item.text.replace(MODEL_REGEX, '').replace(PROVIDER_REGEX, ''),
          };
        }

        return item; // Preserve image_url and other types as is
      })
    : textContent.replace(MODEL_REGEX, '').replace(PROVIDER_REGEX, '');

  return { model, provider, content: cleanedContent };
}

export function streamText(messages: Messages, env: Env, options?: StreamingOptions, apiKeys?: Record<string, string>) {
  let currentModel = DEFAULT_MODEL;
  let currentProvider = DEFAULT_PROVIDER as unknown as Provider;

  const processedMessages = messages.map((message): ProcessedMessage => {
    // First create a copy without toolInvocations
    const { toolInvocations, ...rest } = message;
    const processed: ProcessedMessage = {
      ...rest,
      toolInvocations: toolInvocations?.map((invocation) => ({
        ...invocation,
        state: 'result' as const,
      })),
    };

    if (message.role === 'user') {
      const { model, provider, content } = extractPropertiesFromMessage(message);

      if (MODEL_LIST.find((m) => m.name === model)) {
        currentModel = model;
      }

      currentProvider = provider;
      processed.content = content;
    }

    return processed;
  });

  const modelDetails = MODEL_LIST.find((m) => m.name === currentModel);
  const dynamicMaxTokens = modelDetails && modelDetails.maxTokenAllowed ? modelDetails.maxTokenAllowed : MAX_TOKENS;

  // Cast the model to LanguageModelV1 to handle type mismatch between SDK versions
  const model = getModel(currentProvider, currentModel, env, apiKeys) as LanguageModelV1;

  return _streamText({
    ...options,
    model,
    system: getSystemPrompt(),
    maxTokens: dynamicMaxTokens,
    messages: convertToCoreMessages(processedMessages),
  });
}
