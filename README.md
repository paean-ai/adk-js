# Agent Development Kit (ADK) for TypeScript — Paean Fork

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![NPM Version](https://img.shields.io/npm/v/@paean-ai/adk)](https://www.npmjs.com/package/@paean-ai/adk)

<html>
    <h2 align="center">
      <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
    </h2>
    <h3 align="center">
      A hard fork of <a href="https://github.com/google/adk-node">Google's ADK for Node.js</a>, maintained by Paean AI with production-hardened fixes for Gemini 3 / 3.1 models.
    </h3>
</html>

Published as **`@paean-ai/adk`** on npm.

--------------------------------------------------------------------------------

## Why This Fork Exists

Google's upstream ADK (`@google/adk`) targets general Gemini 2.x usage. When we
adopted Gemini 3 and later Gemini 3.1 models in production at Paean, we hit
several issues that required deep runtime changes. Rather than maintain an
ever-growing patch stack, we hard-forked the repository so we can iterate
freely while still giving back fixes that apply upstream.

### Key Differences from Upstream

| Area | Upstream (`@google/adk`) | Paean Fork (`@paean-ai/adk`) |
|------|--------------------------|-------------------------------|
| Gemini 3 function call names | Not handled | Resolves `tool_name:method_name` colon-separated format (e.g. `google_search:search` → `google_search`) |
| Empty-content streaming chunks | Treated as final response, breaking the agent loop | Detected and skipped — prevents premature loop termination with `gemini-3.1-flash-lite-preview` and similar models |
| Unknown tool calls | Crashes the agent run | Returns a graceful error response so the LLM can self-correct; agent continues |
| Error retry in agent loop | No retry on `UNEXPECTED_TOOL_CALL` / `MALFORMED_FUNCTION_CALL` | Up to 2 automatic retries with `consecutiveErrors` tracking before giving up |
| `thoughtSignature` handling | Basic | Enhanced propagation across multi-turn conversations to maintain reasoning chains in Gemini 3 |
| Cross-user state contamination | Possible when reusing model instances | Fixed — each conversation gets a fresh `Gemini` instance to prevent `cachedThoughtSignature` leakage |
| Transient `UNKNOWN_ERROR` | No retry | Automatic retry with backoff for transient LLM failures |

## Gemini 3 / 3.1 Support

This fork is tested in production with:

- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`
- `gemini-3.1-pro-preview`

All function-calling, streaming, multi-turn, and tool-orchestration paths have
been validated with 100+ registered tools in a single agent context.

## Installation

```bash
npm install @paean-ai/adk
```

Or with yarn:

```bash
yarn add @paean-ai/adk
```

## Quick Start

```typescript
import { LlmAgent, Gemini, GOOGLE_SEARCH } from '@paean-ai/adk';

const agent = new LlmAgent({
    name: 'search_assistant',
    model: new Gemini({ model: 'gemini-3-flash-preview' }),
    instruction: 'You are a helpful assistant.',
    tools: [GOOGLE_SEARCH],
});
```

## Paean-Specific Fixes in Detail

### 1. Gemini 3 Colon-Separated Function Names

Gemini 3 models sometimes return function call names in `tool_name:method_name`
format. The `resolveToolName()` helper in `core/src/agents/functions.ts` strips
the suffix and falls back to the base tool name:

```
Model returns: "google_search:search"
Tool registered as: "google_search"
→ Resolved successfully via colon-fallback
```

### 2. Empty-Content Streaming Chunk Guard

`gemini-3.1-flash-lite-preview` emits a trailing streaming chunk after
function-call chunks that contains only an empty text part. Without the guard,
`isFinalResponse()` treats this as a valid final response, cutting the agent
loop short before tool results are sent back. The fix in
`LlmAgent.postprocess()` detects and skips these empty chunks:

```typescript
// core/src/agents/llm_agent.ts — postprocess()
const allEmpty = llmResponse.content.parts.every((p) => {
  if (p.functionCall || p.functionResponse || ...) return false;
  if ('text' in p && typeof p.text === 'string' && p.text.length > 0) return false;
  return true;
});
if (allEmpty) return; // Skip — let the agent loop continue
```

### 3. Graceful Unknown-Tool Handling

When the LLM hallucinates a tool name that doesn't exist in `toolsDict`, instead
of throwing and crashing the entire agent run, we return a structured error
response so the model can self-correct:

```
Function 'nonExistentTool' is not available.
Please use a different approach or pick from the tools already declared in your configuration.
```

### 4. Agent-Loop Error Retry

Content-less error events (`UNEXPECTED_TOOL_CALL`, `MALFORMED_FUNCTION_CALL`) are
retried up to 2 times before giving up. This prevents a single bad model
response from terminating the entire conversation.

## Documentation

For general ADK concepts, refer to the upstream documentation:

- **[ADK Documentation](https://google.github.io/adk-docs)**
- **[ADK Samples](https://github.com/google/adk-samples)**
- **[Upstream JS ADK](https://github.com/google/adk-node)**

## Building

```bash
npm run build          # Compile to CJS, ESM, and Web
npm run build:watch    # Watch mode (ESM only)
```

## License

This project is licensed under the Apache 2.0 License — see the
[LICENSE](LICENSE) file for details.

Forked from [google/adk-node](https://github.com/google/adk-node). Original
work copyright Google LLC.

--------------------------------------------------------------------------------

*Built for production. Hardened for Gemini 3.*
