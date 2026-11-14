// The OpenAI-shaped pivot and the Gemini Generative Language wire.
//
// The pivot is an OpenAI Chat Completions shape (see lib/anthropic.ts). This file carries the Gemini
// side: the pivot request as a generateContent body, a generateContent answer as a Chat Completions
// object, and a streamGenerateContent SSE stream as Chat Completions chunks. Everything is a pure
// function of its arguments, so a replay reproduces it.
//
//   POST {base}/v1beta/models/{model}:generateContent            x-goog-api-key header
//   POST {base}/v1beta/models/{model}:streamGenerateContent?alt=sse
//   Body   {contents:[{role:'user'|'model', parts:[...]}], systemInstruction?, tools?, toolConfig?,
//           generationConfig?}
//   Answer {candidates:[{content:{role:'model',parts:[...]}, finishReason}], usageMetadata, modelVersion}
//   SSE    one GenerateContentResponse JSON per `data:` frame, partial parts included.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every `created` this file writes: a replay must give the same document. */
const FIXED_CREATED = 1767225600;

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION' || finish === 'BLOCKLIST' || finish === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

/** One OpenAI content part as a Gemini part: text, inline image data or a file URI; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
    if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  }
  return null;
}

/** A JSON-schema object without the keywords Gemini rejects, in place. */
function cleanSchema(value: unknown): Json {
  if (!isRecord(value)) return { type: 'object', properties: {} };
  const out: Json = {};
  for (const [key, item] of Object.entries(value)) {
    if (key === 'additionalProperties' || key === '$schema' || key === '$id' || key === 'definitions' || key === '$defs') continue;
    if (isRecord(item)) out[key] = cleanSchema(item);
    else if (Array.isArray(item)) out[key] = item.flatMap((x) => (isRecord(x) ? [cleanSchema(x)] : [x]));
    else out[key] = item;
  }
  return out;
}

/**
 * A Chat Completions request as a generateContent body that names `model` on its URL: system messages
 * become the systemInstruction, image parts become inlineData or fileData, tool_calls become
 * functionCall parts, tool messages become functionResponse parts, and the sampling knobs, JSON mode,
 * tool declarations and tool choice pass through generationConfig and toolConfig.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemParts: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      if (typeof raw.content === 'string' && raw.content !== '') systemParts.push({ text: raw.content });
      else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (isRecord(block) && block.type === 'text' && typeof block.text === 'string' && block.text !== '') systemParts.push({ text: block.text });
        }
      }
      continue;
    }
    if (role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const name = toolNames.get(id) ?? '';
      const content = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '');
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: content } } }] });
      continue;
    }
    const isModel = role === 'assistant';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string' && content !== '') parts.push({ text: content });
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        const part = geminiPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
      const id = typeof call.id === 'string' ? call.id : '';
      if (id !== '') toolNames.set(id, call.function.name);
      let args: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
      } else if (isRecord(call.function.arguments)) args = call.function.arguments;
      parts.push({ functionCall: { name: call.function.name, args } });
    }
    if (parts.length === 0) continue;
    contents.push({ role: isModel ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const generation: Json = {};
  if (typeof body.temperature === 'number') generation.temperature = body.temperature;
  if (typeof body.top_p === 'number') generation.topP = body.top_p;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null && Number.isFinite(maxTokens) && maxTokens > 0) generation.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) generation.stopSequences = body.stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') generation.responseMimeType = 'application/json';
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: cleanSchema(isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} }),
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  const config: Json = {};
  if (choice === 'auto') config.mode = 'AUTO';
  else if (choice === 'required') config.mode = 'ANY';
  else if (choice === 'none') config.mode = 'NONE';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    config.mode = 'ANY';
    config.allowedFunctionNames = [choice.function.name];
  }
  if (Object.keys(config).length > 0) out.toolConfig = { functionCallingConfig: config };
  return out;
}

/** A generateContent answer as a Chat Completions object that names `model`. */
export function geminiToOpenAIAnswer(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text = text === '' ? part.text : `${text}${part.text}`;
    if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const output = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  const message: Json = { role: 'assistant', content: text !== '' ? text : calls.length > 0 ? null : '' };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl_aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: geminiFinishToOpenAI(candidate.finishReason) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * A streamGenerateContent SSE stream as Chat Completions chunks: the first chunk carries the assistant
 * role, each text part becomes a content delta, each functionCall part becomes a tool_calls chunk, and
 * the stream closes with the finish_reason chunk, a usage chunk and [DONE]. `model` is the served model.
 */
export async function* geminiEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
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

  /** One GenerateContentResponse chunk as OpenAI chunk texts, mutating the stream's state. */
  const chunksOfResponse = (response: Json): string[] => {
    const out: string[] = [];
    const candidates = Array.isArray(response.candidates) ? response.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string' && part.text !== '') out.push(...startEvents(), chunk({ content: part.text }, null));
      if (isRecord(part.functionCall)) {
        out.push(
          ...startEvents(),
          chunk({ tool_calls: [{ index: toolCount, id: `call_${toolCount}`, type: 'function', function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) } }] }, null),
        );
        toolCount += 1;
      }
    }
    if (candidate.finishReason !== undefined && candidate.finishReason !== null) stopReason = geminiFinishToOpenAI(candidate.finishReason);
    const usage = isRecord(response.usageMetadata) ? response.usageMetadata : {};
    if (typeof usage.promptTokenCount === 'number') inputTokens = usage.promptTokenCount;
    if (typeof usage.candidatesTokenCount === 'number') outputTokens = usage.candidatesTokenCount;
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
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          let response: unknown;
          try {
            response = JSON.parse(trimmed.slice(5).trim());
          } catch {
            continue;
          }
          if (isRecord(response)) for (const text of chunksOfResponse(response)) yield text;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        let response: unknown;
        try {
          response = JSON.parse(trimmed.slice(5).trim());
        } catch {
          continue;
        }
        if (isRecord(response)) for (const text of chunksOfResponse(response)) yield text;
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) for (const text of [...startEvents(), ...finishEvents()]) yield text;
}
