type JsonObject = Record<string, unknown>;

const SUPABASE_URL = requiredEnv("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = requiredEnv("SUPABASE_SERVICE_ROLE_KEY");
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") || Deno.env.get("SUPABASE_PUBLISHABLE_KEY") || SUPABASE_SERVICE_ROLE_KEY;
const BEDROCK_API_KEY = Deno.env.get("AWS_BEARER_TOKEN_BEDROCK") || Deno.env.get("BEDROCK_API_KEY") || "";
const BEDROCK_BASE_URL = (Deno.env.get("BEDROCK_OPENAI_BASE_URL") || Deno.env.get("ROUTER_BEDROCK_BASE_URL") || "https://bedrock-mantle.us-east-1.api.aws/v1").replace(/\/+$/, "");

const FREE_TOKEN_BUDGET = positiveIntEnv("FREE_TIER_TOKEN_BUDGET", 600_000);
const FREE_REQUESTS_PER_HOUR = positiveIntEnv("FREE_TIER_REQUESTS_PER_HOUR", 120);
const MAX_COMPLETION_TOKENS = positiveIntEnv("FREE_TIER_MAX_COMPLETION_TOKENS", 8192);

const MODEL_PRICES: Record<string, { inputPerMillionUsd: number; outputPerMillionUsd: number }> = {
  "minimax.minimax-m2.5": { inputPerMillionUsd: 0.36, outputPerMillionUsd: 1.44 },
  "moonshotai.kimi-k2.5": { inputPerMillionUsd: 0.72, outputPerMillionUsd: 3.60 },
};

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-user-token",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

Deno.serve(async (request: Request) => {
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (request.method !== "POST") {
    return jsonResponse({ error: "method_not_allowed", message: "Use POST." }, 405);
  }

  try {
    const payload = await request.json();
    const provider = stringValue(payload.provider);
    const model = normalizeModel(stringValue(payload.model));
    const requestId = stringValue(payload.request_id);
    const input = isObject(payload.input) ? payload.input : {};

    if (!requestId) {
      return jsonResponse({ error: "bad_request", message: "Missing request_id." }, 400);
    }
    if (provider !== "bedrock" || !(model in MODEL_PRICES)) {
      return jsonResponse({
        error: "unsupported_provider_or_model",
        message: "api-generate currently supports CADAgent managed Bedrock free-tier models only.",
      }, 400);
    }
    if (!BEDROCK_API_KEY) {
      return jsonResponse({ error: "provider_unavailable", message: "Managed Bedrock provider is not configured." }, 503);
    }

    const userToken = extractUserToken(request);
    if (!userToken) {
      return jsonResponse({ error: "unauthorized", message: "Missing user token." }, 401);
    }

    const user = await getUser(userToken);
    const userId = stringValue(user.id);
    if (!userId) {
      return jsonResponse({ error: "unauthorized", message: "Invalid user token." }, 401);
    }

    const entitlement = await ensureEntitlement(userId);
    const now = new Date();
    const periodStart = parseDate(stringValue(entitlement.period_start)) || now;
    const periodEnd = parseDate(stringValue(entitlement.period_end)) || new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
    const tokenBudget = Number(entitlement.token_budget) > 0 ? Number(entitlement.token_budget) : FREE_TOKEN_BUDGET;

    const requestsLastHour = await countRecentRequests(userId);
    if (requestsLastHour >= FREE_REQUESTS_PER_HOUR) {
      return jsonResponse({
        error: "rate_limited",
        message: "Free-tier hourly request limit reached. Please try again later.",
        usage: { requests_last_hour: requestsLastHour, request_limit_per_hour: FREE_REQUESTS_PER_HOUR },
      }, 429);
    }

    const usedTokens = await sumUsedTokens(userId, periodStart, periodEnd);
    const maxTokens = clampInt(Number(input.max_tokens), 1, MAX_COMPLETION_TOKENS);
    const estimatedInputTokens = estimateTokens({
      system: input.system,
      messages: input.messages,
      tools: input.tools,
    });
    const estimatedTotalTokens = estimatedInputTokens + maxTokens;
    if (usedTokens + estimatedTotalTokens > tokenBudget) {
      return jsonResponse({
        error: "quota_exceeded",
        message: "Free-tier token budget exceeded.",
        usage: {
          used_tokens: usedTokens,
          estimated_tokens: estimatedTotalTokens,
          token_budget: tokenBudget,
          remaining_tokens: Math.max(tokenBudget - usedTokens, 0),
        },
      }, 402);
    }

    const bedrockRequest = buildBedrockRequest(model, input, maxTokens);
    const bedrockResponse = await fetch(`${BEDROCK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${BEDROCK_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(bedrockRequest),
    });

    const bedrockData = await bedrockResponse.json().catch(() => ({}));
    if (!bedrockResponse.ok) {
      await recordUsage({
        userId,
        requestId,
        provider,
        model,
        inputTokens: 0,
        outputTokens: 0,
        costCents: 0,
        status: "provider_error",
        metadata: { status: bedrockResponse.status, error: bedrockData },
      });
      return jsonResponse({
        error: "provider_error",
        message: "Managed Bedrock request failed.",
        provider_status: bedrockResponse.status,
        provider_error: bedrockData,
      }, 502);
    }

    const result = convertOpenAIChatResponseToAnthropic(bedrockData);
    const inputTokens = intFromPath(bedrockData, ["usage", "prompt_tokens"]);
    const outputTokens = intFromPath(bedrockData, ["usage", "completion_tokens"]);
    const costCents = calculateCostCents(model, inputTokens, outputTokens);
    const remainingTokens = Math.max(tokenBudget - usedTokens - inputTokens - outputTokens, 0);

    await recordUsage({
      userId,
      requestId,
      provider,
      model,
      inputTokens,
      outputTokens,
      costCents,
      status: "succeeded",
      metadata: {
        bedrock_id: bedrockData.id,
        bedrock_model: bedrockData.model,
        period_start: periodStart.toISOString(),
        period_end: periodEnd.toISOString(),
      },
    });

    return jsonResponse({
      result,
      usage: {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
        cost_cents: costCents,
        used_tokens: usedTokens + inputTokens + outputTokens,
        remaining_tokens: remainingTokens,
        token_budget: tokenBudget,
        requests_last_hour: requestsLastHour + 1,
        request_limit_per_hour: FREE_REQUESTS_PER_HOUR,
      },
    });
  } catch (error) {
    if (error instanceof HttpError) {
      return jsonResponse({ error: error.code, message: error.message }, error.status);
    }
    console.error("api-generate failed", error);
    return jsonResponse({
      error: "internal_error",
      message: error instanceof Error ? error.message : "Unknown api-generate error.",
    }, 500);
  }
});

function buildBedrockRequest(model: string, input: JsonObject, maxTokens: number): JsonObject {
  const tools = convertToolsToOpenAIChat(input.tools);
  const body: JsonObject = {
    model,
    messages: convertMessagesToOpenAIChat(stringValue(input.system), input.messages),
    max_tokens: maxTokens,
  };
  if (tools.length > 0) {
    body.tools = tools;
    body.tool_choice = "auto";
  }
  if (model === "minimax.minimax-m2.5" && stringValue(input.reasoning_effort)) {
    body.reasoning_split = true;
  }
  return body;
}

function convertToolsToOpenAIChat(tools: unknown): JsonObject[] {
  if (!Array.isArray(tools)) return [];
  return tools.filter(isObject).map((tool) => ({
    type: "function",
    function: {
      name: stringValue(tool.name),
      description: stringValue(tool.description),
      parameters: isObject(tool.input_schema) ? tool.input_schema : isObject(tool.parameters) ? tool.parameters : {},
    },
  }));
}

function convertMessagesToOpenAIChat(systemPrompt: string, messages: unknown): JsonObject[] {
  const chatMessages: JsonObject[] = [];
  if (systemPrompt) {
    chatMessages.push({ role: "system", content: systemPrompt });
  }
  if (!Array.isArray(messages)) {
    return chatMessages;
  }

  for (const rawMessage of messages) {
    if (!isObject(rawMessage)) continue;
    const role = stringValue(rawMessage.role) || "user";
    const content = rawMessage.content;

    if (typeof content === "string") {
      chatMessages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) {
      chatMessages.push({ role, content: String(content ?? "") });
      continue;
    }

    const textParts: string[] = [];
    const toolCalls: JsonObject[] = [];
    for (const rawBlock of content) {
      if (!isObject(rawBlock)) {
        textParts.push(String(rawBlock));
        continue;
      }
      const blockType = stringValue(rawBlock.type);
      if (blockType === "text") {
        textParts.push(stringValue(rawBlock.text));
      } else if (blockType === "image") {
        throw new Error("Image attachments are not supported by CADAgent free managed models yet.");
      } else if (blockType === "tool_use") {
        toolCalls.push({
          id: stringValue(rawBlock.id) || `call_${chatMessages.length}_${toolCalls.length}`,
          type: "function",
          function: {
            name: stringValue(rawBlock.name),
            arguments: JSON.stringify(isObject(rawBlock.input) ? rawBlock.input : {}),
          },
        });
      } else if (blockType === "tool_result") {
        chatMessages.push({
          role: "tool",
          tool_call_id: stringValue(rawBlock.tool_use_id),
          content: toolResultToText(rawBlock.content),
        });
      }
    }

    const assistantText = textParts.filter(Boolean).join("\n").trim();
    if (toolCalls.length > 0) {
      chatMessages.push({ role: "assistant", content: assistantText || null, tool_calls: toolCalls });
    } else if (textParts.length > 0) {
      chatMessages.push({ role, content: assistantText });
    }
  }
  return chatMessages;
}

function convertOpenAIChatResponseToAnthropic(response: JsonObject): JsonObject {
  const choice = Array.isArray(response.choices) && isObject(response.choices[0]) ? response.choices[0] : {};
  const message = isObject(choice.message) ? choice.message : {};
  const contentBlocks: JsonObject[] = [];
  let stopReason = "end_turn";

  const text = stringValue(message.content);
  if (text) {
    contentBlocks.push({ type: "text", text });
  }

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls.filter(isObject) : [];
  if (toolCalls.length > 0) {
    stopReason = "tool_use";
    for (const toolCall of toolCalls) {
      const fn = isObject(toolCall.function) ? toolCall.function : {};
      contentBlocks.push({
        type: "tool_use",
        id: stringValue(toolCall.id),
        name: stringValue(fn.name),
        input: parseObjectJson(stringValue(fn.arguments)),
      });
    }
  }

  const result: JsonObject = {
    stop_reason: stopReason,
    content: contentBlocks,
    usage: {
      input_tokens: intFromPath(response, ["usage", "prompt_tokens"]),
      output_tokens: intFromPath(response, ["usage", "completion_tokens"]),
    },
  };
  const reasoning = extractReasoning(message);
  if (reasoning) {
    result.reasoning = reasoning;
  }
  return result;
}

function extractReasoning(message: JsonObject): string {
  const parts: string[] = [];
  for (const key of ["reasoning", "reasoning_content", "thinking"]) {
    const value = message[key];
    if (typeof value === "string" && value.trim()) {
      parts.push(value.trim());
    }
  }

  const details = message.reasoning_details;
  if (Array.isArray(details)) {
    for (const item of details) {
      if (!isObject(item)) continue;
      const text = stringValue(item.text) || stringValue(item.content);
      if (text.trim()) {
        parts.push(text.trim());
      }
    }
  }
  return parts.join("\n\n");
}

async function getUser(userToken: string): Promise<JsonObject> {
  const response = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
    headers: {
      "Authorization": `Bearer ${userToken}`,
      "apikey": SUPABASE_ANON_KEY,
    },
  });
  if (!response.ok) {
    throw new HttpError(401, "unauthorized", "Invalid or expired user token.");
  }
  return await response.json();
}

async function ensureEntitlement(userId: string): Promise<JsonObject> {
  const query = `/rest/v1/user_entitlements?user_id=eq.${encodeURIComponent(userId)}&select=user_id,plan,token_budget,period_start,period_end&limit=1`;
  const existing = await restJson(query);
  if (Array.isArray(existing) && isObject(existing[0])) {
    return existing[0];
  }

  const now = new Date();
  const periodEnd = new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
  const inserted = await restJson("/rest/v1/user_entitlements", {
    method: "POST",
    headers: { "Prefer": "return=representation" },
    body: JSON.stringify({
      user_id: userId,
      plan: "free",
      token_budget: FREE_TOKEN_BUDGET,
      period_start: now.toISOString(),
      period_end: periodEnd.toISOString(),
    }),
  });
  if (Array.isArray(inserted) && isObject(inserted[0])) {
    return inserted[0];
  }
  return {
    user_id: userId,
    plan: "free",
    token_budget: FREE_TOKEN_BUDGET,
    period_start: now.toISOString(),
    period_end: periodEnd.toISOString(),
  };
}

async function sumUsedTokens(userId: string, periodStart: Date, periodEnd: Date): Promise<number> {
  const rows = await restJson(`/rest/v1/usage_ledger?user_id=eq.${encodeURIComponent(userId)}&created_at=gte.${encodeURIComponent(periodStart.toISOString())}&created_at=lt.${encodeURIComponent(periodEnd.toISOString())}&select=input_tokens,output_tokens`);
  if (!Array.isArray(rows)) return 0;
  return rows.reduce((sum, row) => {
    if (!isObject(row)) return sum;
    return sum + Math.max(Number(row.input_tokens) || 0, 0) + Math.max(Number(row.output_tokens) || 0, 0);
  }, 0);
}

async function countRecentRequests(userId: string): Promise<number> {
  const since = new Date(Date.now() - 60 * 60 * 1000).toISOString();
  const rows = await restJson(`/rest/v1/usage_ledger?user_id=eq.${encodeURIComponent(userId)}&created_at=gte.${encodeURIComponent(since)}&select=id`);
  return Array.isArray(rows) ? rows.length : 0;
}

async function recordUsage(args: {
  userId: string;
  requestId: string;
  provider: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  costCents: number;
  status: string;
  metadata: JsonObject;
}): Promise<void> {
  await restJson("/rest/v1/usage_ledger", {
    method: "POST",
    headers: { "Prefer": "resolution=ignore-duplicates" },
    body: JSON.stringify({
      user_id: args.userId,
      request_id: args.requestId,
      provider: args.provider,
      model: args.model,
      input_tokens: args.inputTokens,
      output_tokens: args.outputTokens,
      cost_cents: args.costCents,
      status: args.status,
      metadata: args.metadata,
    }),
  });
}

async function restJson(path: string, init: RequestInit = {}): Promise<unknown> {
  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`);
  headers.set("apikey", SUPABASE_SERVICE_ROLE_KEY);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${SUPABASE_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    const error = await response.text();
    throw new HttpError(response.status, "supabase_rest_error", error);
  }
  if (response.status === 204) return null;
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

function extractUserToken(request: Request): string {
  const explicit = request.headers.get("x-user-token");
  if (explicit) return explicit.trim();
  const auth = request.headers.get("authorization") || "";
  return auth.replace(/^Bearer\s+/i, "").trim();
}

function normalizeModel(model: string): string {
  const normalized = model.toLowerCase().trim();
  if (normalized === "minimax-m2.5" || normalized === "minimax m2.5") return "minimax.minimax-m2.5";
  if (normalized === "kimi-k2.5" || normalized === "kimi k2.5") return "moonshotai.kimi-k2.5";
  return normalized;
}

function toolResultToText(content: unknown): string {
  if (!Array.isArray(content)) return String(content ?? "");
  return content
    .filter((item) => !(isObject(item) && item.type === "image"))
    .map((item) => isObject(item) ? stringValue(item.text) || JSON.stringify(item) : String(item))
    .join("\n");
}

function parseObjectJson(value: string): JsonObject {
  if (!value) return {};
  try {
    const parsed = JSON.parse(value);
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function estimateTokens(value: unknown): number {
  return Math.ceil(JSON.stringify(value ?? "").length / 4);
}

function calculateCostCents(model: string, inputTokens: number, outputTokens: number): number {
  const prices = MODEL_PRICES[model];
  if (!prices) return 0;
  const usd = (inputTokens / 1_000_000) * prices.inputPerMillionUsd + (outputTokens / 1_000_000) * prices.outputPerMillionUsd;
  return Number((usd * 100).toFixed(6));
}

function intFromPath(obj: unknown, path: string[]): number {
  let current = obj;
  for (const key of path) {
    if (!isObject(current)) return 0;
    current = current[key];
  }
  return Math.max(Math.trunc(Number(current) || 0), 0);
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return max;
  return Math.min(Math.max(Math.trunc(value), min), max);
}

function parseDate(value: string): Date | null {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function positiveIntEnv(name: string, fallback: number): number {
  const value = Number(Deno.env.get(name));
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : fallback;
}

function jsonResponse(body: JsonObject, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
    },
  });
}

class HttpError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
