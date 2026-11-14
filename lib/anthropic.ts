// The OpenAI-shaped pivot and the Anthropic Messages upstream wire.
//
// Every request pivots through an OpenAI Chat Completions shape: the client's body is rewritten into
// that pivot, the pivot is rewritten into the routed provider's wire, and the provider's answer comes
// back through the pivot to the client's wire. This file carries the Anthropic side: the pivot request
// as a Messages request, a Messages answer as a Chat Completions object, and a Messages SSE stream as
// Chat Completions chunks. Everything is a pure function of its arguments, so a replay reproduces it.
//
//   POST {base}/v1/messages   Authorization or x-api-key per the directory, anthropic-version header.
//   Body   {model, max_tokens, messages, system?, tools?, tool_choice?, temperature?, top_p?,
//           stop_sequences?, stream?}
//   Answer {id, type:'message', role:'assistant', content:[{type:'text'|'tool_use',...}],
//           stop_reason, usage:{input_tokens, output_tokens}}
//   SSE    message_start, content_block_start, content_block_delta (text_delta | input_json_delta),
//          content_block_stop, message_delta, message_stop

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every `created` this file writes: a replay must give the same document. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of an array's text parts or blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the wire's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** One Anthropic content block as an OpenAI content part: text or an image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * A Chat Completions request as an Anthropic Messages request that names `model`: system messages
 * become the system parameter, image parts become image blocks, tool_calls become tool_use blocks,
 * tool messages become tool_result blocks, and the tools, tool_choice, sampling knobs and streaming
 * pass through. max_tokens is required in Anthropic's shape and defaults to 4096.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const assistant = role === 'assistant';
    if (role !== 'user' && !assistant) continue;
    const content = raw.content;
    if (typeof content === 'string' || content === null) {
      const text = typeof content === 'string' ? content : '';
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      const blocks: Json[] = [];
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${blocks.length}`,
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length === 0) continue;
      messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      for (const block of content) {
        const part = partOf(isRecord(block) ? block : {});
        if (part !== null && part.type === 'text') blocks.push({ type: 'text', text: part.text });
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${blocks.length}`,
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length === 0) continue;
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
        continue;
      }
      const part = partOf(block);
      if (part !== null && part.type === 'text') blocks.push({ type: 'text', text: part.text });
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    const toolCallId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
    if (toolCallId !== '') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: toolCallId, content: toolResultText(raw.content) }],
      });
    }
  }

  const out: Json = { model, max_tokens: 4096, messages };
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null && Number.isFinite(maxTokens) && maxTokens > 0) out.max_tokens = maxTokens;
  if (systemParts.length > 0) out.system = systemParts.join('\n\n');
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Messages answer as a Chat Completions object that names `model`. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text = text === '' ? block.text : `${text}\n${block.text}`;
    if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' && block.id !== '' ? block.id : `call_${calls.length}`,
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  const message: Json = { role: 'assistant', content: text !== '' ? text : calls.length > 0 ? null : '' };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: typeof body.id === 'string' && body.id !== '' ? body.id : 'chatcmpl_aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * A Messages SSE stream as Chat Completions chunks: the first chunk carries the assistant role, each
 * text_delta and input_json_delta becomes a delta chunk, and the stream closes with the finish_reason
 * chunk, a usage chunk and [DONE]. `model` is the served model, named back to the client.
 */
export async function* anthropicEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let toolCount = 0;
  let stopReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const id = 'chatcmpl_aggregate';

  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      chunk({}, stopReason ?? 'stop'),
      `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`,
      'data: [DONE]\n\n',
    ];
  };

  /** One SSE frame's lines as OpenAI chunks, mutating the stream's state. */
  const chunksOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      let event: unknown;
      try {
        event = JSON.parse(trimmed.slice(5).trim());
      } catch {
        continue;
      }
      if (!isRecord(event) || typeof event.type !== 'string') continue;
      if (event.type === 'message_start') {
        const message = isRecord(event.message) ? event.message : {};
        const usage = isRecord(message.usage) ? message.usage : {};
        inputTokens = typeof usage.input_tokens === 'number' ? usage.input_tokens : inputTokens;
        out.push(...startEvents());
        continue;
      }
      if (event.type === 'content_block_start') {
        const block = isRecord(event.content_block) ? event.content_block : {};
        if (block.type === 'tool_use') {
          out.push(
            ...startEvents(),
            chunk({ tool_calls: [{ index: toolCount, id: typeof block.id === 'string' ? block.id : `call_${toolCount}`, type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null),
          );
          toolCount += 1;
        }
        continue;
      }
      if (event.type === 'content_block_delta') {
        const delta = isRecord(event.delta) ? event.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          out.push(...startEvents(), chunk({ content: delta.text }, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          out.push(chunk({ tool_calls: [{ index: Math.max(toolCount - 1, 0), function: { arguments: delta.partial_json } }] }, null));
        }
        continue;
      }
      if (event.type === 'message_delta') {
        const delta = isRecord(event.delta) ? event.delta : {};
        stopReason = anthropicStopToOpenAI(delta.stop_reason);
        const usage = isRecord(event.usage) ? event.usage : {};
        outputTokens = typeof usage.output_tokens === 'number' ? usage.output_tokens : outputTokens;
        continue;
      }
      if (event.type === 'message_stop') out.push(...finishEvents());
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const text of chunksOfFrame(frame)) yield text;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const text of chunksOfFrame(buffer)) yield text;
  } finally {
    reader.releaseLock();
  }
  if (!finished) for (const text of [...startEvents(), ...finishEvents()]) yield text;
}
