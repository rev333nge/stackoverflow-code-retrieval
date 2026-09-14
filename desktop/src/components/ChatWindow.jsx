import { useEffect, useRef, useState } from 'react'
import Message from './Message'

const SUGGESTIONS = [
  'explain a pandas function',
  'debug this traceback',
  'compare two approaches',
]

function Composer({ draft, setDraft, onSubmit, sending }) {
  const textareaRef = useRef(null)

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  function handleChange(e) {
    setDraft(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 180)}px`
    }
  }

  // The textarea grows with content, so its height resets to the single-line
  // default whenever a send clears the draft back to ''.
  useEffect(() => {
    if (draft === '' && textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [draft])

  return (
    <form
      className="chat-input"
      onSubmit={(e) => {
        e.preventDefault()
        onSubmit()
      }}
    >
      <textarea
        ref={textareaRef}
        value={draft}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask a pandas/numpy question..."
        rows={1}
      />
      <button className="mono" type="submit" disabled={sending || !draft.trim()}>send</button>
    </form>
  )
}

function ModelDropdown({ activeModel, modelOptions, models, disabled, onChange }) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (!open) return
    function onPointerDown(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    function onKeyDown(e) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  return (
    <div className="model-select-wrap" ref={wrapRef}>
      <button
        type="button"
        className="model-select-trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="model-dot" />
        <span className="model-name mono">{activeModel}</span>
        <span className={`model-chevron mono${open ? ' open' : ''}`}>&#9662;</span>
      </button>

      {open && (
        <div className="model-dropdown-panel" role="listbox">
          {modelOptions.map((m) => (
            <button
              type="button"
              key={m}
              role="option"
              aria-selected={m === activeModel}
              className={`model-option${m === activeModel ? ' selected' : ''}`}
              onClick={() => {
                onChange(m)
                setOpen(false)
              }}
            >
              <span className="model-option-name mono">
                {models.includes(m) ? m : `${m} (not installed)`}
              </span>
              {m === activeModel && <span className="model-option-check">&#10003;</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function ThinkingIndicator() {
  return (
    <div className="message">
      <div className="message-label mono assistant">&rarr; ASSISTANT</div>
      <div className="thinking-indicator" role="status" aria-label="thinking">
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span className="thinking-dot" />
      </div>
    </div>
  )
}

export default function ChatWindow({ conversation, messages, models, onChangeModel, onSend, sending }) {
  const [draft, setDraft] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleSubmit() {
    const text = draft.trim()
    if (!text || sending) return
    setDraft('')
    onSend(text)
  }

  const activeModel = conversation?.model ?? models[0] ?? 'no model'
  const modelInstalled = conversation ? models.includes(conversation.model) : true
  const modelOptions = conversation && !modelInstalled ? [conversation.model, ...models] : models

  return (
    <div className="chat-window">
      <div className="chat-topbar">
        <ModelDropdown
          activeModel={activeModel}
          modelOptions={modelOptions}
          models={models}
          disabled={!conversation}
          onChange={onChangeModel}
        />
      </div>

      {!conversation ? (
        <div className="empty-state">
          <div className="empty-sq" />
          <div className="empty-caption mono">ready to run &middot; {activeModel}</div>
          <div className="composer-wrap">
            <Composer draft={draft} setDraft={setDraft} onSubmit={handleSubmit} sending={sending} />
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="suggestion-chip mono" onClick={() => setDraft(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="chat-messages">
            {messages.map((m) => (
              <Message key={m.id} message={m} />
            ))}
            {sending && <ThinkingIndicator />}
            <div ref={bottomRef} />
          </div>

          <div className="composer-dock">
            <Composer draft={draft} setDraft={setDraft} onSubmit={handleSubmit} sending={sending} />
            <div className="composer-caption mono">
              {activeModel} &middot; local inference via Ollama, no data leaves this machine
            </div>
          </div>
        </>
      )}
    </div>
  )
}
