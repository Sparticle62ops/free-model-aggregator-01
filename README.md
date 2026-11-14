# free-model-aggregator-01

An OpenAI-compatible aggregator that fronts many free-tier LLM providers through one
local endpoint, ranks them per request, and fails over without leaking either wire.

## What it does

- Probes every configured provider, learns each one's real model catalog, and builds a
  live catalog of free models with their ratings, context lengths and capabilities.
- Serves `POST /v1/models` and `POST /v1/chat/completions` (OpenAI wire) and
  `POST /v1/messages` (Anthropic wire), translating between wires per request.
- Streams: SSE is translated both ways (Gemini and Anthropic upstream frames to OpenAI
  chunks and back), including usage.
- Routes each request to the highest-rated provider that satisfies the request's
  requirements (tools, vision, json, context, free tier) and has budget left.
- Fails over on 402/429/5xx to the next-best provider, honors retry-after, caps
  concurrent retries, and drops a provider only after repeated refused calls.
- Registers itself with the environment (`POST /register`), polls `/work`, serves each
  request, and exits 0. `--list` prints the discovered catalog.

## Run

    node --experimental-strip-types main.ts --list
    node --experimental-strip-types main.ts

Configuration comes from the environment (provider endpoints and keys).

## License

MIT
