import { useEffect, useRef, useState } from "react";

type SourceChunk = {
  text: string;
  source: string;
  chunk_index: number;
  score?: number;
  rerank?: number;
  page_start?: number | null;
  page_end?: number | null;
};
type ChatResponse = { session_id: string; answer: string; sources: SourceChunk[] };
type ModeInfo = { use_openai: boolean; openai_model: string };

type Message = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: SourceChunk[];
};

const API_BASE = "http://127.0.0.1:8000";
const uid = () => Math.random().toString(36).slice(2);

export default function App() {
  // Chat/session
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "system",
      content:
        "Upload a regulation PDF, then ask a question. Follow-up questions will use conversation context.",
    },
  ]);
  const [input, setInput] = useState("");

  // Controls
  const [topN, setTopN] = useState(3); // kept for compatibility
  const [strict, setStrict] = useState(true);
  const [minScore, setMinScore] = useState(0.25);
  const [selectedSource, setSelectedSource] = useState<string>("");

  // UI state
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  // Server mode badge
  const [serverMode, setServerMode] = useState<ModeInfo | null>(null);

  // Available sources (PDF list)
  const [sourcesList, setSourcesList] = useState<string[]>([]);

  // autoscroll
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Fetch server mode + sources
  async function refreshServerMode() {
    try {
      const r = await fetch(`${API_BASE}/config`);
      if (r.ok) setServerMode(await r.json());
    } catch {}
  }
  async function refreshSources() {
    try {
      const r = await fetch(`${API_BASE}/sources`);
      if (r.ok) {
        const data = await r.json();
        setSourcesList(data.sources || []);
        if (selectedSource && !data.sources?.includes(selectedSource)) {
          setSelectedSource("");
        }
      }
    } catch {}
  }

  useEffect(() => {
    refreshServerMode();
    refreshSources();
  }, []);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg: Message = { id: uid(), role: "user", content: question };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);
    setToast(null);

    try {
      const payload = {
        message: question,
        session_id: sessionId,
        top_n: topN,
        strict,
        vec_min_score: minScore,
        source: selectedSource || null,
      };
      const r = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as ChatResponse;

      if (!sessionId) setSessionId(data.session_id);

      const assistant: Message = {
        id: uid(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((m) => [...m, assistant]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { id: uid(), role: "assistant", content: "Sorry — I couldn’t process that request. Please try again in a moment." },
      ]);
      setToast(`Chat failed: ${e?.message ?? e}`);
    } finally {
      setLoading(false);
    }
  }

  async function uploadPdf(file: File) {
    const form = new FormData();
    form.append("file", file);
    setUploading(true);
    setToast(null);
    try {
      const r = await fetch(`${API_BASE}/upload_pdf`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setToast(`Uploaded ${data.file} • ${data.chunks_added} chunks indexed`);
      refreshSources();
    } catch (e: any) {
      setToast(`Upload failed: ${e?.message ?? e}`);
    } finally {
      setUploading(false);
    }
  }

  function newChat() {
    setSessionId(null);
    setMessages([{ id: uid(), role: "system", content: "New chat started. Upload a PDF (if needed) and ask your question." }]);
    setToast("New chat session created.");
  }

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: 24, fontFamily: "Inter, system-ui, Arial" }}>
      <header style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <h1 style={{ margin: 0, flex: 1 }}>PowerAI — Chat</h1>
        {serverMode && (
          <span
            title={serverMode.use_openai ? `Using OpenAI model: ${serverMode.openai_model}` : "Local synthesis (no API usage)"}
            style={{
              padding: "4px 8px",
              borderRadius: 6,
              fontSize: 13,
              background: serverMode.use_openai ? "#e7f5ff" : "#f2f2f2",
              border: serverMode.use_openai ? "1px solid #91d5ff" : "1px solid #ccc",
              color: serverMode.use_openai ? "#096dd9" : "#555",
            }}
          >
            {serverMode.use_openai ? `Hybrid (OpenAI: ${serverMode.openai_model})` : "Local mode"}
          </span>
        )}
        <button onClick={refreshServerMode} style={{ padding: "8px 12px" }}>Refresh mode</button>
        <button onClick={newChat} style={{ padding: "8px 12px" }}>New chat</button>
      </header>

      <section style={{ border: "1px solid #eee", padding: 16, borderRadius: 8, marginBottom: 12 }}>
        <h3>Upload PDF</h3>
        <input type="file" accept="application/pdf" onChange={(e) => e.target.files && e.target.files[0] && uploadPdf(e.target.files[0])} disabled={uploading} />
        {uploading && <p>Uploading & indexing…</p>}
      </section>

      <section style={{ border: "1px solid #eee", padding: 16, borderRadius: 8, marginBottom: 12 }}>
        <h3>Chat</h3>

        {/* Controls row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, alignItems: "center", marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 13, color: "#666" }}><b>Top N (compat):</b> {topN}</div>
            <input type="range" min={1} max={10} value={topN} onChange={(e) => setTopN(parseInt(e.target.value))} style={{ width: "100%" }} />
          </div>

          <div>
            <div style={{ fontSize: 13, color: "#666", marginBottom: 4 }}><b>Strict mode:</b></div>
            <label style={{ fontSize: 13, color: "#444" }}>
              <input type="checkbox" checked={strict} onChange={(e) => setStrict(e.target.checked)} style={{ marginRight: 8 }} />
              Only answer if confident (min score)
            </label>
            <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>Min score: {minScore.toFixed(2)}</div>
            <input type="range" min={0} max={1} step={0.01} value={minScore} onChange={(e) => setMinScore(parseFloat(e.target.value))} style={{ width: "100%" }} />
          </div>

          <div>
            <div style={{ fontSize: 13, color: "#666", marginBottom: 4 }}><b>Source:</b></div>
            <select value={selectedSource} onChange={(e) => setSelectedSource(e.target.value)} style={{ width: "100%", padding: "4px 6px", borderRadius: 6, border: "1px solid #ccc" }}>
              <option value="">All documents</option>
              {sourcesList.map((s) => (<option key={s} value={s}>{s}</option>))}
            </select>
            <button onClick={refreshSources} style={{ marginTop: 6, padding: "4px 8px" }}>Refresh sources</button>
          </div>
        </div>

        {/* Chat window */}
        <div style={{ border: "1px solid #eee", borderRadius: 8, padding: 12, background: "#fff", minHeight: 260, maxHeight: 460, overflowY: "auto" }}>
          {messages.map((m) => (<MessageBubble key={m.id} msg={m} serverMode={serverMode} />))}
          {loading && <div style={{ color: "#888", fontSize: 13 }}>thinking…</div>}
          <div ref={endRef} />
        </div>

        {/* Composer */}
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask a question… (press Enter to send)"
            style={{ flex: 1, padding: 10, border: "1px solid #ddd", borderRadius: 6 }}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()} style={{ padding: "10px 16px" }}>Send</button>
        </div>

        {sessionId && <div style={{ marginTop: 8, fontSize: 12, color: "#999" }}>session: {sessionId}</div>}
      </section>

      {toast && <div style={{ background: "#f7f7ff", border: "1px solid #e0e0ff", padding: 12, borderRadius: 8 }}>{toast}</div>}
    </div>
  );
}

function MessageBubble({ msg, serverMode }: { msg: Message; serverMode: ModeInfo | null }) {
  const isAssistant = msg.role === "assistant";
  const bg = msg.role === "user" ? "#f2fff5" : isAssistant ? (serverMode?.use_openai ? "#f8fbff" : "#fdfdfd") : "#fff7e6";
  const border = msg.role === "user" ? "#d7f7de" : isAssistant ? (serverMode?.use_openai ? "#b7d7ff" : "#eee") : "#ffe3b3";
  const label = msg.role === "user" ? "You" : msg.role === "assistant" ? "Assistant" : "System";

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>{label}</div>
      <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: 10, whiteSpace: "pre-wrap" }}>
        {msg.content}

        {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 12 }}>
            <b>Citations</b>
            <ul style={{ margin: "6px 0 0 16px" }}>
              {msg.sources.map((s, i) => {
                const hasStart = typeof s.page_start === "number";
                const hasEnd = typeof s.page_end === "number";
                const pageText =
                  hasStart && hasEnd && s.page_start !== s.page_end
                    ? `pp. ${s.page_start}–${s.page_end}`
                    : hasStart
                    ? `p. ${s.page_start}`
                    : null;
                return (
                  <li key={i}>
                    <span style={{ color: "#444" }}>
                      {s.source}
                      {pageText ? ` • ${pageText}` : ""}
                    </span>
                    {typeof s.score === "number" && <span style={{ color: "#999" }}> (score {s.score.toFixed(3)})</span>}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
