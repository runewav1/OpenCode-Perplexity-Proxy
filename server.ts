import { serve } from "bun";

// ─── Config ───────────────────────────────────────────────────────────────────

const PPLX_API_KEY = process.env.PERPLEXITY_API_KEY!;
const PPLX_BASE = "https://api.perplexity.ai";
const PORT = 4099;
const MIN_GAP_MS = 150;

// Slow-network / weak-WiFi resilience
// Maximum time (ms) to wait for the upstream Perplexity fetch to complete.
// 10 minutes gives ample room for slow connections and large responses.
const UPSTREAM_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

// How often (ms) to emit an SSE keep-alive comment to the client while
// waiting for the next upstream data chunk during streaming.
const STREAM_KEEPALIVE_INTERVAL_MS = 15_000; // 15 seconds

// Bun v1.3.9 segfault workaround: close idle HTTP sockets before Bun's
// internal teardown path hits the buggy code path (idleTimeout is in seconds).
const SERVER_IDLE_TIMEOUT_S = 120; // 2 minutes

if (!PPLX_API_KEY) {
  console.error("[pplx-proxy] FATAL: PERPLEXITY_API_KEY is not set in environment");
  process.exit(1);
}

// Bun v1.3.9 segfault workaround: a periodic no-op timer keeps the event loop
// "warm" on Windows, preventing Bun's idle-GC path from triggering the
// null-pointer dereference in socket teardown.  Never cleared intentionally.
const _eventLoopKeepAlive = setInterval(() => { /* no-op — keep event loop warm */ }, 60_000);

// ─── Model Map ────────────────────────────────────────────────────────────────
// Keys  = flat IDs used in opencode.json "models" block
// Values = Perplexity Agent API "provider/model" format
// Source: https://docs.perplexity.ai/docs/agent-api/models

const MODEL_MAP: Record<string, string> = {
  "gpt-5.2":           "openai/gpt-5.2",
  "gpt-5.1":           "openai/gpt-5.1",
  "gpt-5-mini":        "openai/gpt-5-mini",
  "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
  "claude-haiku-4-5":  "anthropic/claude-haiku-4-5",
  "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",   // experimental
  "gemini-2.5-pro":    "google/gemini-2.5-pro",
  "gemini-2.5-flash":  "google/gemini-2.5-flash",
  "gemini-3-pro":      "google/gemini-3-pro-preview",
  "gemini-3-flash":    "google/gemini-3-flash-preview",
  "gemini-3.1-pro":    "google/gemini-3.1-pro-preview", // experimental
  "grok-4.1":          "xai/grok-4-1-fast-non-reasoning",
  "kimi-k2.5":         "moonshot/kimi-k2.5",            // experimental
};

const PPLX_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${PPLX_API_KEY}`,
};

// ─── Rate-limit Queue ─────────────────────────────────────────────────────────
// OpenCode probes all models simultaneously on startup which triggers 429s.
// This serialises outbound upstream requests with a minimum gap between them.

let lastRequestTime = 0;
let requestQueue = Promise.resolve();

function enqueueRequest<T>(fn: () => Promise<T>): Promise<T> {
  const result = requestQueue.then(async () => {
    const gap = lastRequestTime + MIN_GAP_MS - Date.now();
    if (gap > 0) await sleep(gap);
    lastRequestTime = Date.now();
    return fn();
  });
  requestQueue = result.then(() => {}, () => {});
  return result;
}

function sleep(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}

// ─── Types ────────────────────────────────────────────────────────────────────

type OAIContentPart =
  | { type: "text"; text: string }
  | { type: string; [key: string]: unknown };

type OAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type OAIMessage = {
  role: "system" | "user" | "assistant" | "tool" | string;
  content?: string | OAIContentPart[] | null;
  tool_calls?: OAIToolCall[];
  tool_call_id?: string;
  name?: string;
};

type OAITool = {
  type: string;
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
    strict?: boolean | null;
  };
};

// Perplexity Agent API input types
// Per OpenAPI spec: FunctionCallInput.arguments is a JSON *string*, NOT a parsed object.
type PplxTextContent  = { type: "input_text"; text: string };
type PplxInputMessage = { type: "message"; role: "system" | "user" | "assistant"; content: PplxTextContent[] };
type PplxFunctionCall = { type: "function_call"; call_id: string; name: string; arguments: string };
type PplxFunctionCallOutput = { type: "function_call_output"; call_id: string; output: string };
type PplxInputItem    = PplxInputMessage | PplxFunctionCall | PplxFunctionCallOutput;

// Perplexity Agent API output types
// Per OpenAPI spec: FunctionCallOutputItem.arguments is a JSON *string*.
type PplxOutputTextContent    = { type: "output_text"; text: string };
type PplxMessageOutputItem    = { type: "message"; id?: string; role: "assistant"; content: PplxOutputTextContent[] };
type PplxFunctionCallOutputItem = { type: "function_call"; id?: string; call_id: string; name: string; arguments: string; status?: string };
type PplxOutputItem =
  | PplxMessageOutputItem
  | PplxFunctionCallOutputItem
  | { type: "search_results" | "fetch_url_results"; [key: string]: unknown };

type PplxResponseUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
};

type PplxResponse = {
  id: string;
  object: "response";
  model: string;
  status: "completed" | "failed" | "in_progress" | "requires_action";
  created_at: number;
  output: PplxOutputItem[];
  usage?: PplxResponseUsage;
  error?: { message: string; code?: string; type?: string };
};

// ─── Argument Serialisation ───────────────────────────────────────────────────
// Per the Perplexity OpenAPI spec (FunctionCallOutputItem & FunctionCallInput):
//   arguments is a JSON *string* in BOTH directions.
// OpenAI Chat Completions also uses a JSON string for arguments.
//
// So the proxy mostly just passes the string through. The only conversion needed
// is ensuring we have a valid JSON string (not null/undefined/empty).

function ensureArgsString(args: unknown): string {
  if (args == null) return "{}";

  if (typeof args === "string") {
    if (args.trim() === "") return "{}";
    // Validate it's actually JSON; if not, wrap it
    try {
      JSON.parse(args);
      return args;
    } catch {
      // Received a raw non-JSON string — wrap it so callers don't explode
      return JSON.stringify({ _raw: args });
    }
  }

  // Defensive: if somehow we get a parsed object, stringify it
  if (typeof args === "object") {
    return JSON.stringify(args);
  }

  return JSON.stringify({ _value: args });
}

// ─── Content Normalisation ────────────────────────────────────────────────────
// Converts any OpenAI content shape into a plain string.

function contentToString(content: OAIMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c) => (c.type === "text" && "text" in c ? (c as { type: "text"; text: string }).text : ""))
      .join("");
  }
  return String(content);
}

// ─── Message Translation ──────────────────────────────────────────────────────
// OpenAI messages[] → Perplexity { instructions?, input[] }
//
// Rules from the Agent API spec (OpenAPI schema):
//  - System messages → top-level `instructions` field (NOT in input array)
//  - role: "tool"     → FunctionCallOutputInput  { type: "function_call_output", call_id, output }
//  - assistant with tool_calls → FunctionCallInput { type: "function_call", call_id, name, arguments (JSON string) }
//  - user / assistant text → InputMessage { type: "message", role, content: [{ type: "input_text", text }] }
//
// CRITICAL: Per the OpenAPI spec, FunctionCallInput.arguments is a JSON STRING,
// not a parsed object. OpenAI also sends arguments as a JSON string. So we pass
// the string through directly without parsing/re-serializing.

function translateMessages(messages: OAIMessage[]): {
  instructions: string | undefined;
  input: PplxInputItem[];
} {
  const systemParts: string[] = [];
  const input: PplxInputItem[] = [];

  for (const msg of messages) {

    // ── System → instructions ──
    if (msg.role === "system") {
      const text = contentToString(msg.content);
      if (text.trim()) systemParts.push(text.trim());
      continue;
    }

    // ── Tool result → function_call_output ──
    if (msg.role === "tool") {
      const output = contentToString(msg.content);
      input.push({
        type: "function_call_output",
        call_id: msg.tool_call_id ?? "",
        output: output || "null",
      });
      continue;
    }

    // ── Assistant with tool_calls → function_call items ──
    if (msg.role === "assistant" && msg.tool_calls?.length) {
      for (const tc of msg.tool_calls) {
        input.push({
          type: "function_call",
          call_id: tc.id,
          name: tc.function.name,
          // Agent API expects arguments as a JSON string (same as OpenAI format)
          arguments: ensureArgsString(tc.function.arguments),
        });
      }
      // If assistant turn also has text content alongside tool_calls, emit it too
      const text = contentToString(msg.content);
      if (text.trim()) {
        input.push({
          type: "message",
          role: "assistant",
          content: [{ type: "input_text", text: text.trim() }],
        });
      }
      continue;
    }

    // ── Normal user / assistant text message ──
    const text = contentToString(msg.content);
    const role = msg.role === "assistant" ? "assistant" : "user";

    // Skip empty messages — Agent API rejects them
    if (!text.trim()) continue;

    input.push({
      type: "message",
      role,
      content: [{ type: "input_text", text: text.trim() }],
    });
  }

  return {
    instructions: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
    input,
  };
}

// ─── Tool Translation ─────────────────────────────────────────────────────────
// OpenAI tools[]  →  Perplexity FunctionTool[]
//
// Perplexity FunctionTool schema (from OpenAPI spec):
//   { type: "function", name: string, description?: string, parameters?: object, strict?: boolean }

function translateTools(tools: OAITool[]): Record<string, unknown>[] {
  return tools
    .filter((t) => t.type === "function" && !!t.function?.name)
    .map((tool) => {
      const entry: Record<string, unknown> = {
        type: "function",
        name: tool.function.name,
      };
      if (tool.function.description?.trim()) {
        entry.description = tool.function.description.trim();
      }
      if (tool.function.parameters != null) {
        entry.parameters = sanitiseParameters(tool.function.parameters);
      }
      // strict is valid per the OpenAPI spec — pass it through if present
      if (tool.function.strict != null) {
        entry.strict = Boolean(tool.function.strict);
      }
      return entry;
    });
}

// Recursively removes null/undefined values from JSON Schema objects that can
// cause validation errors on strict API endpoints.
function sanitiseParameters(params: unknown): unknown {
  if (params == null || typeof params !== "object" || Array.isArray(params)) return params;
  const obj = params as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue;
    out[k] = typeof v === "object" && !Array.isArray(v) ? sanitiseParameters(v) : v;
  }
  return out;
}

// ─── Response Translation ─────────────────────────────────────────────────────
// Perplexity Agent API response  →  OpenAI chat.completions shape

function translateResponse(res: PplxResponse, requestedModel: string) {
  const output = res.output ?? [];

  // ── Extract text from MessageOutputItem ──
  const msgItem = output.find((o): o is PplxMessageOutputItem => o.type === "message");
  const textContent =
    (msgItem?.content ?? [])
      .filter((c) => c.type === "output_text")
      .map((c) => c.text)
      .join("") || null;

  // ── Extract tool calls from FunctionCallOutputItem entries ──
  const rawToolCalls = output.filter(
    (o): o is PplxFunctionCallOutputItem => o.type === "function_call"
  );

  const toolCalls = rawToolCalls.map((tc, idx) => {
    // Per the OpenAPI spec, tc.arguments is already a JSON string
    const argsString = ensureArgsString(tc.arguments);

    // Debug log — remove once tool calling is confirmed stable
    console.log(
      `[pplx-proxy] tool call [${idx}] name="${tc.name}" call_id="${tc.call_id}" args=${argsString}`
    );

    return {
      id: tc.call_id || `call_${idx}_${Date.now()}`,
      type: "function" as const,
      function: {
        name: tc.name ?? "",
        arguments: argsString,
      },
    };
  });

  const hasToolCalls = toolCalls.length > 0;
  const usage = res.usage;

  return {
    id: res.id ?? `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: res.created_at ?? Math.floor(Date.now() / 1000),
    model: requestedModel,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          // OpenAI spec: content must be null (not empty string) when tool_calls present
          content: hasToolCalls ? null : textContent,
          ...(hasToolCalls ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: hasToolCalls ? "tool_calls" : "stop",
      },
    ],
    usage: usage
      ? {
          prompt_tokens:      usage.input_tokens  ?? 0,
          completion_tokens:  usage.output_tokens ?? 0,
          total_tokens:
            usage.total_tokens ??
            (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
        }
      : undefined,
  };
}

// ─── Streaming Translation ────────────────────────────────────────────────────
// Perplexity Agent API SSE  →  OpenAI SSE chunk format
//
// Perplexity SSE event types (from OpenAPI spec):
//   response.created                          — initial response object
//   response.in_progress                      — processing started
//   response.output_item.added                — new output item started
//   response.output_text.delta                — { item_id, output_index, delta }
//   response.output_text.done                 — final text content
//   response.output_item.done                 — output item complete (includes full function_call with arguments)
//   response.completed                        — final response with output
//   response.failed                           — error occurred
//   response.reasoning.*                      — reasoning phase events
//
// IMPORTANT: The OpenAPI spec does NOT list response.function_call_arguments.delta
// as a supported event type. Function call arguments arrive COMPLETE in the
// response.output_item.done event. We handle both patterns defensively.

function translateStream(
  upstreamBody: ReadableStream<Uint8Array>,
  requestedModel: string
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const reader  = upstreamBody.getReader();

  // State
  let buffer       = "";
  let roleEmitted  = false;
  let streamClosed = false;
  const callIdToIndex = new Map<string, number>();
  let toolCallCount   = 0;
  // Track which tool calls have had their arguments fully emitted
  const emittedArgs = new Set<string>();

  function emitChunk(
    delta: Record<string, unknown>,
    finishReason: string | null = null
  ): Uint8Array {
    return encoder.encode(
      `data: ${JSON.stringify({
        id:      "chatcmpl-stream",
        object:  "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model:   requestedModel,
        choices: [{ index: 0, delta, finish_reason: finishReason }],
      })}\n\n`
    );
  }

  function closeStream(controller: ReadableStreamDefaultController<Uint8Array>) {
    if (streamClosed) return;
    streamClosed = true;
    try {
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    } catch { /* already closed */ }
  }

  // Ensure a tool call header has been emitted for the given call_id
  function ensureToolCallHeader(
    controller: ReadableStreamDefaultController<Uint8Array>,
    callId: string,
    name: string
  ): number {
    if (callIdToIndex.has(callId)) {
      return callIdToIndex.get(callId)!;
    }
    const idx = toolCallCount++;
    callIdToIndex.set(callId, idx);
    controller.enqueue(
      emitChunk({
        tool_calls: [
          {
            index: idx,
            id:    callId,
            type:  "function",
            function: {
              name,
              arguments: "",
            },
          },
        ],
      })
    );
    return idx;
  }

  // Promisified helper: races a reader.read() against a keep-alive timeout.
  // Returns { done, value } on data, or { keepAlive: true } on timeout.
  function readWithKeepalive(): Promise<
    { done: boolean; value: Uint8Array | undefined; keepAlive?: false } |
    { keepAlive: true; done?: never; value?: never }
  > {
    return Promise.race([
      reader.read() as Promise<{ done: boolean; value: Uint8Array | undefined }>,
      sleep(STREAM_KEEPALIVE_INTERVAL_MS).then(() => ({ keepAlive: true as const })),
    ]);
  }

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        while (true) {
          const result = await readWithKeepalive();

          // Keep-alive: no data from upstream for STREAM_KEEPALIVE_INTERVAL_MS —
          // emit an SSE comment to prevent the client connection from timing out.
          if ("keepAlive" in result && result.keepAlive) {
            try {
              controller.enqueue(encoder.encode(": keep-alive\n\n"));
            } catch { /* stream already closed */ }
            continue;
          }

          const { done, value } = result as { done: boolean; value: Uint8Array | undefined };

          if (done) {
            closeStream(controller);
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw || raw === "[DONE]") continue;

            let event: Record<string, unknown>;
            try {
              event = JSON.parse(raw);
            } catch {
              continue;
            }

            const type = event.type as string;

            // ── New output item added ──
            if (type === "response.output_item.added") {
              const item = event.item as Record<string, unknown> | undefined;
              if (!item) continue;

              if (item.type === "message" && !roleEmitted) {
                roleEmitted = true;
                controller.enqueue(emitChunk({ role: "assistant", content: "" }));
              }

              if (item.type === "function_call") {
                const callId = (item.call_id ?? item.id) as string;
                const name = (item.name as string) ?? "";
                if (callId) {
                  ensureToolCallHeader(controller, callId, name);
                }
              }
            }

            // ── Text delta ──
            if (type === "response.output_text.delta") {
              const delta = event.delta as string | undefined;
              if (delta != null) {
                if (!roleEmitted) {
                  roleEmitted = true;
                  controller.enqueue(emitChunk({ role: "assistant", content: "" }));
                }
                controller.enqueue(emitChunk({ content: delta }));
              }
            }

            // ── Function call argument delta (NOT in OpenAPI spec but handle defensively) ──
            if (type === "response.function_call_arguments.delta") {
              const callId = event.call_id as string | undefined;
              const delta  = event.delta  as string | undefined;
              if (delta == null) continue;

              const idx = callId
                ? ensureToolCallHeader(controller, callId, "")
                : toolCallCount++; // should not happen

              controller.enqueue(
                emitChunk({
                  tool_calls: [{ index: idx, function: { arguments: delta } }],
                })
              );
              if (callId) emittedArgs.add(callId);
            }

            // ── Output item done — PRIMARY path for function call arguments ──
            // This event contains the complete output item. For function_call items,
            // this is where we get the full arguments string if it wasn't streamed
            // incrementally via response.function_call_arguments.delta.
            if (type === "response.output_item.done") {
              const item = event.item as Record<string, unknown> | undefined;
              if (!item) continue;

              if (item.type === "function_call") {
                const callId = (item.call_id ?? item.id) as string;
                const name = (item.name as string) ?? "";
                const args = ensureArgsString(item.arguments);

                // Ensure header was emitted
                const idx = ensureToolCallHeader(controller, callId, name);

                // Only emit full arguments if we haven't already streamed them
                // incrementally via delta events
                if (!emittedArgs.has(callId)) {
                  console.log(
                    `[pplx-proxy] stream: emitting full args for tool call [${idx}] ` +
                    `name="${name}" call_id="${callId}" args=${args.substring(0, 200)}`
                  );
                  controller.enqueue(
                    emitChunk({
                      tool_calls: [{ index: idx, function: { arguments: args } }],
                    })
                  );
                  emittedArgs.add(callId);
                }
              }
            }

            // ── Response completed ──
            if (type === "response.completed") {
              const resp = event.response as PplxResponse | undefined;
              const hasFunctionCalls =
                resp?.output?.some((o) => o.type === "function_call") ??
                toolCallCount > 0;
              controller.enqueue(
                emitChunk({}, hasFunctionCalls ? "tool_calls" : "stop")
              );
              closeStream(controller);
              return;
            }

            // ── Response failed ──
            if (type === "response.failed") {
              const resp = event.response as Record<string, unknown> | undefined;
              const error = resp?.error ?? event.error;
              console.error("[pplx-proxy] Stream response.failed:", error ?? event);
              // Emit graceful stop so OpenCode doesn't hang
              controller.enqueue(emitChunk({}, "stop"));
              closeStream(controller);
              return;
            }
          }
        }
      } catch (err) {
        console.error("[pplx-proxy] Stream read error:", err);
        try {
          controller.enqueue(emitChunk({}, "stop"));
          closeStream(controller);
        } catch {
          controller.error(err);
        }
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });
}

// ─── Error Response Helper ────────────────────────────────────────────────────

function errorResponse(message: string, type: string, status: number): Response {
  return new Response(
    JSON.stringify({ error: { message, type, code: status } }),
    {
      status,
      headers: {
        "Content-Type": "application/json",
        // Bun v1.3.9 segfault workaround: close connections so idle sockets
        // don't accumulate in a state that triggers the null-deref on teardown.
        "Connection": "close",
      },
    }
  );
}

// ─── Server ───────────────────────────────────────────────────────────────────

serve({
  port: PORT,

  // Bun v1.3.9 segfault workaround: force-close idle sockets after 2 minutes
  // so they are torn down cleanly before Bun's internal GC/event-loop idle path
  // hits the dangling-pointer dereference (address 0xFFFFFFFFFFFFFFFF crash).
  idleTimeout: SERVER_IDLE_TIMEOUT_S,

  async fetch(req: Request): Promise<Response> {
    // Bun v1.3.9 segfault workaround: top-level catch prevents unhandled errors
    // from propagating into Bun's internal error handler which can dereference a
    // dangling socket pointer on Windows.
    try {
    const url    = new URL(req.url);
    const method = req.method;

    // ── Health check ──
    if (url.pathname === "/health" && method === "GET") {
      return Response.json(
        { status: "ok", models: Object.keys(MODEL_MAP).length, uptime: process.uptime() },
        { headers: { "Connection": "close" } }
      );
    }

    // ── Model list ──
    if (url.pathname === "/v1/models" && method === "GET") {
      return Response.json(
        {
          object: "list",
          data: Object.keys(MODEL_MAP).map((id) => ({
            id,
            object:    "model",
            owned_by:  "perplexity-proxy",
            created:   0,
          })),
        },
        { headers: { "Connection": "close" } }
      );
    }

    // ── Chat completions ──
    if (url.pathname === "/v1/chat/completions" && method === "POST") {

      // Parse body
      let body: Record<string, unknown>;
      try {
        body = await req.json();
      } catch {
        return errorResponse("Invalid JSON body", "invalid_request_error", 400);
      }

      const requestedModel = ((body.model as string | undefined) ?? "").trim();
      if (!requestedModel) {
        return errorResponse("model field is required", "invalid_request_error", 400);
      }

      const resolvedModel = MODEL_MAP[requestedModel] ?? requestedModel;
      const isStreaming   = Boolean(body.stream);
      const messages      = (body.messages as OAIMessage[] | undefined) ?? [];
      const rawTools      = body.tools as OAITool[] | undefined;

      if (!messages.length) {
        return errorResponse(
          "messages array is required and must not be empty",
          "invalid_request_error",
          400
        );
      }

      // ── Translate messages ──
      const { instructions, input } = translateMessages(messages);

      // ── Build Perplexity request body ──
      const pplxBody: Record<string, unknown> = {
        model:  resolvedModel,
        input,
        stream: isStreaming,
      };

      if (instructions) {
        pplxBody.instructions = instructions;
      }

      // Tools — translate and attach only if non-empty after filtering
      if (rawTools && rawTools.length > 0) {
        const translated = translateTools(rawTools);
        if (translated.length > 0) {
          pplxBody.tools = translated;
        }
      }

      // max_tokens → max_output_tokens (Agent API field name)
      if (body.max_tokens != null) {
        pplxBody.max_output_tokens = body.max_tokens;
      }

      // Reasoning config — Agent API uses { reasoning: { effort: "low"|"medium"|"high" } }
      if (body.reasoning != null) {
        pplxBody.reasoning = body.reasoning;
      } else if (body.reasoning_effort != null) {
        pplxBody.reasoning = { effort: body.reasoning_effort };
      }

      // temperature / top_p / presence_penalty / frequency_penalty are NOT
      // supported by the Agent API — intentionally dropped.

      console.log(
        `[pplx-proxy] ${requestedModel} → ${resolvedModel} | stream=${isStreaming} | tools=${rawTools?.length ?? 0} | msgs=${messages.length}`
      );

      // Debug: log translated input summary for tool-call troubleshooting
      if (rawTools && rawTools.length > 0) {
        const funcCallInputs = input.filter((i: PplxInputItem) => i.type === "function_call");
        const funcCallOutputs = input.filter((i: PplxInputItem) => i.type === "function_call_output");
        if (funcCallInputs.length > 0 || funcCallOutputs.length > 0) {
          console.log(
            `[pplx-proxy] input contains ${funcCallInputs.length} function_call(s) and ${funcCallOutputs.length} function_call_output(s)`
          );
        }
      }

      // ── Fire upstream request via rate-limit queue ──
      let upstream: Response;
      try {
        upstream = await enqueueRequest(() =>
          fetch(`${PPLX_BASE}/v1/responses`, {
            method:  "POST",
            headers: PPLX_HEADERS,
            body:    JSON.stringify(pplxBody),
            // Generous timeout for slow/weak-WiFi connections (Issue 1)
            signal:  AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
          })
        );
      } catch (err) {
        console.error("[pplx-proxy] Network error:", err);
        return errorResponse("Upstream network error", "proxy_error", 502);
      }

      // Surface upstream errors verbatim
      if (!upstream.ok) {
        const errText = await upstream.text().catch(() => "");
        console.error(`[pplx-proxy] Upstream ${upstream.status}:`, errText);
        return new Response(
          errText ||
            JSON.stringify({ error: { message: "Upstream error", code: upstream.status } }),
          {
            status:  upstream.status,
            headers: { "Content-Type": "application/json", "Connection": "close" },
          }
        );
      }

      // ── Streaming path ──
      if (isStreaming) {
        if (!upstream.body) {
          return errorResponse(
            "Upstream returned no body for streaming request",
            "proxy_error",
            502
          );
        }
        return new Response(translateStream(upstream.body, requestedModel), {
          headers: {
            "Content-Type":    "text/event-stream",
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":      "keep-alive",
          },
        });
      }

      // ── Non-streaming path ──
      // stream=false → Perplexity returns plain application/json (confirmed in docs)
      let pplxRes: PplxResponse;
      try {
        pplxRes = (await upstream.json()) as PplxResponse;
      } catch {
        const raw = await upstream.text().catch(() => "(unreadable)");
        console.error("[pplx-proxy] Failed to parse upstream JSON:", raw);
        return errorResponse("Invalid JSON from upstream", "proxy_error", 502);
      }

      if (pplxRes.status === "failed") {
        console.error("[pplx-proxy] Response failed:", pplxRes.error);
        return new Response(
          JSON.stringify({
            error: pplxRes.error ?? { message: "Model returned failed status" },
          }),
          { status: 500, headers: { "Content-Type": "application/json", "Connection": "close" } }
        );
      }

      if (pplxRes.status === "in_progress") {
        console.error("[pplx-proxy] Unexpected in_progress on non-streaming response");
        return errorResponse(
          "Upstream response still in progress",
          "proxy_error",
          502
        );
      }

      // Debug: log the response output types
      const outputTypes = pplxRes.output?.map((o) => o.type) ?? [];
      console.log(`[pplx-proxy] Response status=${pplxRes.status} outputs=[${outputTypes.join(", ")}]`);

      return Response.json(
        translateResponse(pplxRes, requestedModel),
        { headers: { "Connection": "close" } }
      );
    }

    return new Response("Not Found", { status: 404, headers: { "Connection": "close" } });

    } catch (err) {
      // Top-level safety net — see Bun v1.3.9 segfault workaround comment above
      console.error("[pplx-proxy] Unhandled fetch error:", err);
      return new Response(
        JSON.stringify({ error: { message: "Internal proxy error", type: "proxy_error", code: 500 } }),
        { status: 500, headers: { "Content-Type": "application/json", "Connection": "close" } }
      );
    }
  },
});

console.log(`[pplx-proxy] Listening on http://localhost:${PORT}`);
console.log(`[pplx-proxy] ${Object.keys(MODEL_MAP).length} models registered: ${Object.keys(MODEL_MAP).join(", ")}`);
