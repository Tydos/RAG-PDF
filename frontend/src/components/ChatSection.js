import { useEffect, useRef, useState } from 'react';
import { chat, listDocuments } from '../api/api';


const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

const ArrowIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="7" y1="17" x2="17" y2="7"/>
    <polyline points="7 7 17 7 17 17"/>
  </svg>
);

export default function ChatSection({ messages, setMessages }) {
  const [input, setInput]                   = useState('');
  const [loading, setLoading]               = useState(false);
  const [error, setError]                   = useState(null);
  const [blobByFilename, setBlobByFilename] = useState({});
  const threadRef = useRef(null);

  useEffect(() => {
    listDocuments()
      .then((docs) => setBlobByFilename(Object.fromEntries(docs.map((d) => [d.filename, d.blob_url || '']))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages, loading]);

  async function handleSend(e) {
    e?.preventDefault?.();
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setError(null);
    setMessages((prev) => [...prev, { role: 'user', content: text, chunks: [] }]);
    setLoading(true);
    try {
      const res = await chat(text);
      const assistantMsg = { role: 'assistant', content: res.answer ?? '', chunks: res.chunks ?? [], latency: res.latency };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg = String(err).replace(/^Error:\s*/, '');
      setError(msg);
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  const hrefFor = (c) => { const url = blobByFilename[c.filename]; return url ? `${url}#page=${c.page}` : null; };

  return (
    <div className="chat-container">
      <div ref={threadRef} className="chat-thread" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <p>Upload a document, then ask anything about it.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role}`}>
            <div className="msg-bubble">{msg.content}</div>
            {msg.role === 'assistant' && msg.chunks?.length > 0 && (
              <div className="msg-sources" aria-label="Sources">
                {msg.chunks.slice(0, 3).map((c, j) => {
                  const href = hrefFor(c);
                  const label = `${c.filename} p.${c.page}`;
                  return href ? (
                    <a key={j} className="source-chip" href={href} target="_blank" rel="noopener noreferrer">
                      {label}<ArrowIcon />
                    </a>
                  ) : (
                    <span key={j} className="source-chip">{label}</span>
                  );
                })}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="msg msg-assistant" aria-label="Assistant is responding">
            <div className="msg-bubble typing-indicator"><span /><span /><span /></div>
          </div>
        )}
      </div>

      {error && <p className="chat-error" role="alert">{error}</p>}

      <div className="chat-input-bar">
        <form className="chat-input-row" onSubmit={handleSend}>
          <input
            className="chat-input"
            type="text"
            placeholder="Ask about your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Message"
          />
          <button
            className="send-btn"
            type="submit"
            disabled={!input.trim() || loading}
            aria-label="Send message"
          >
            <SendIcon />
          </button>
        </form>
        <p className="chat-disclaimer">AI-generated content may be inaccurate.</p>
      </div>
    </div>
  );
}
