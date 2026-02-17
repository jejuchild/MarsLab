/**
 * MARVIS Chat API client — streams Gemini responses via SSE.
 * Supports tool_call events for grounded UI actions.
 */

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type SessionContext = {
  loaded_instruments: string[];
  visible_product_count: number;
  current_lat?: number | null;
  current_lon?: number | null;
};

export type ChatEvent =
  | { event: "chunk"; data: { text: string } }
  | { event: "done"; data: { full_text: string } }
  | { event: "tool_call"; data: { tool: string; params: Record<string, unknown>; message: string } }
  | { event: "error"; data: { error: string } };

/**
 * Stream a MARVIS chat response.
 * Yields ChatEvent objects including tool_call events for UI actions.
 */
export async function* streamMarvisChat(
  message: string,
  history: ChatMessage[] = [],
  context?: SessionContext,
): AsyncGenerator<ChatEvent> {
  const response = await fetch("/api/marvis/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, context: context ?? null }),
  });

  if (!response.ok) {
    yield { event: "error", data: { error: `HTTP ${response.status}` } };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { event: "error", data: { error: "No response body" } };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            yield parsed as ChatEvent;
          } catch {
            // skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
