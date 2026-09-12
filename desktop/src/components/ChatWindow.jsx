import { useEffect, useRef, useState } from 'react'
import Message from './Message'

export default function ChatWindow({ conversation, messages, models, onChangeModel, onSend, sending }) {
  const [draft, setDraft] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (!conversation) {
    return <div className="chat-window chat-window-empty">Start a new chat to begin.</div>
  }

  function handleSubmit(e) {
    e.preventDefault()
    const text = draft.trim()
    if (!text || sending) return
    setDraft('')
    onSend(text)
  }

  return (
    <div className="chat-window">
      <div className="chat-topbar">
        <span className="chat-title">{conversation.title}</span>
        <select
          className="model-select"
          value={conversation.model}
          onChange={(e) => onChangeModel(e.target.value)}
        >
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div className="chat-messages">
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        {sending && <div className="message message-assistant"><div className="message-bubble thinking">thinking...</div></div>}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e)
          }}
          placeholder="Ask a pandas/numpy question..."
          rows={2}
        />
        <button type="submit" disabled={sending || !draft.trim()}>Send</button>
      </form>
    </div>
  )
}
