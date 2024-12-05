import { type ActionFunctionArgs } from '@remix-run/cloudflare';
import { MAX_RESPONSE_SEGMENTS, MAX_TOKENS } from '~/lib/.server/llm/constants';
import { CONTINUE_PROMPT } from '~/lib/.server/llm/prompts';
import { streamText, type Messages, type StreamingOptions } from '~/lib/.server/llm/stream-text';
import SwitchableStream from '~/lib/.server/llm/switchable-stream';
import { logger } from '~/utils/logger';

export async function action(args: ActionFunctionArgs) {
  return chatAction(args);
}

function parseCookies(cookieHeader: string | null): Record<string, string> {
  const cookies: Record<string, string> = {};

  if (!cookieHeader) {
    return cookies;
  }

  // Split the cookie string by semicolons and spaces
  const items = cookieHeader.split(';').map((cookie) => cookie.trim());

  items.forEach((item) => {
    const [name, ...rest] = item.split('=');

    if (name && rest) {
      // Decode the name and value, and join value parts in case it contains '='
      const decodedName = decodeURIComponent(name.trim());
      const decodedValue = decodeURIComponent(rest.join('=').trim());
      cookies[decodedName] = decodedValue;
    }
  });

  return cookies;
}

async function chatAction({ context, request }: ActionFunctionArgs) {
  try {
    const { messages, model } = await request.json<{
      messages: Messages;
      model: string;
    }>();

    const cookieHeader = request.headers.get('Cookie');

    // Parse the cookie's value (returns an object or null if no cookie exists)
    let apiKeys = {};

    try {
      apiKeys = JSON.parse(parseCookies(cookieHeader).apiKeys || '{}');
    } catch (error) {
      logger.error('Error parsing API keys from cookies:', error);

      // Continue with empty API keys object
    }

    const stream = new SwitchableStream();

    try {
      const options: StreamingOptions = {
        toolChoice: 'none',
        apiKeys,
        model,
        onFinish: async ({ text: content, finishReason }) => {
          if (finishReason !== 'length') {
            return stream.close();
          }

          if (stream.switches >= MAX_RESPONSE_SEGMENTS) {
            throw new Error('Cannot continue message: Maximum segments reached');
          }

          const switchesLeft = MAX_RESPONSE_SEGMENTS - stream.switches;

          logger.info(`Reached max token limit (${MAX_TOKENS}): Continuing message (${switchesLeft} switches left)`);

          messages.push({ role: 'assistant', content });
          messages.push({ role: 'user', content: CONTINUE_PROMPT });

          const result = await streamText(messages, context.cloudflare.env, options);

          return stream.switchSource(result.toAIStream());
        },
      };

      const result = await streamText(messages, context.cloudflare.env, options, apiKeys);

      stream.switchSource(result.toAIStream());

      return new Response(stream.readable, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
        },
      });
    } catch (error: unknown) {
      logger.error('Error in chat stream:', error);

      if (error instanceof Error) {
        if (error.message.toLowerCase().includes('api key')) {
          throw new Response('Invalid or missing API key', {
            status: 401,
            statusText: 'Unauthorized',
          });
        }

        if (error.message.toLowerCase().includes('rate limit')) {
          throw new Response('Rate limit exceeded', {
            status: 429,
            statusText: 'Too Many Requests',
          });
        }

        throw new Response(error.message, {
          status: 500,
          statusText: 'Internal Server Error',
        });
      }

      throw new Response('Unknown error occurred', {
        status: 500,
        statusText: 'Internal Server Error',
      });
    }
  } catch (error: unknown) {
    logger.error('Error in chat action:', error);

    if (error instanceof Response) {
      throw error;
    }

    if (error instanceof Error) {
      throw new Response(error.message, {
        status: 400,
        statusText: 'Bad Request',
      });
    }

    throw new Response('Invalid request format', {
      status: 400,
      statusText: 'Bad Request',
    });
  }
}
