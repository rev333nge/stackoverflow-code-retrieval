import { useEffect, useRef, useState } from 'react'
import Message from './Message'

const SUGGESTIONS = [
  'explain a python function',
  'debug this traceback',
  'compare two approaches',
]

// Models are now absolute filesystem paths to a .gguf file (there's no
// Ollama registry giving us a short name anymore), so display just the
// filename while still passing the full path around as the value.
function basename(modelPath) {
  return modelPath.split(/[\\/]/).pop()
}

// Parse a text-input value to an int, clamped to [min, max]; empty/garbage
// falls back to `fallback` so a half-typed field can't persist a bad value.
function clampInt(value, min, max, fallback) {
  const n = parseInt(value, 10)
  if (Number.isNaN(n)) return fallback
  return Math.min(max, Math.max(min, n))
}

function SettingsIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M2 4.5h6M11 4.5h2M2 10.5h2M7 10.5h6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="9.5" cy="4.5" r="1.7" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="4.5" cy="10.5" r="1.7" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

// The controls, remounted per conversation (via key) so each field starts
// from that conversation's saved value. Generation settings (temperature,
// max tokens) take effect on the next message; load settings (context, GPU
// layers) reload the model on the next message -- flagged in their hints.
function SettingsFields({ conversation, onChange }) {
  const [temp, setTemp] = useState(conversation.temperature)
  const [maxTokens, setMaxTokens] = useState(conversation.max_tokens)
  const [nCtx, setNCtx] = useState(conversation.n_ctx)
  const [nGpu, setNGpu] = useState(conversation.n_gpu_layers)

  return (
    <>
      <div className="settings-row">
        <div className="settings-label-line">
          <span className="settings-label mono">temperature</span>
          <span className="settings-val mono">{Number(temp).toFixed(2)}</span>
        </div>
        <input
          type="range" min="0" max="1.5" step="0.05" value={temp}
          onChange={(e) => setTemp(parseFloat(e.target.value))}
          onMouseUp={() => onChange({ temperature: temp })}
          onKeyUp={() => onChange({ temperature: temp })}
        />
        <div className="settings-hint mono">0 = deterministic · higher = more varied</div>
      </div>

      <div className="settings-row">
        <div className="settings-label-line">
          <span className="settings-label mono">max tokens</span>
          <input
            className="settings-num mono" type="number" min="64" max="8192" step="64" value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value)}
            onBlur={() => { const v = clampInt(maxTokens, 64, 8192, 1024); setMaxTokens(v); onChange({ max_tokens: v }) }}
          />
        </div>
        <div className="settings-hint mono">longest a reply can get</div>
      </div>

      <div className="settings-divider" />
      <div className="settings-note mono">changing these reloads the model</div>

      <div className="settings-row">
        <div className="settings-label-line">
          <span className="settings-label mono">context size</span>
          <input
            className="settings-num mono" type="number" min="512" max="32768" step="512" value={nCtx}
            onChange={(e) => setNCtx(e.target.value)}
            onBlur={() => { const v = clampInt(nCtx, 512, 32768, 8192); setNCtx(v); onChange({ n_ctx: v }) }}
          />
        </div>
        <div className="settings-hint mono">how much history + docs fit at once</div>
      </div>

      <div className="settings-row">
        <div className="settings-label-line">
          <span className="settings-label mono">GPU layers</span>
          <input
            className="settings-num mono" type="number" min="-1" max="200" step="1" value={nGpu}
            onChange={(e) => setNGpu(e.target.value)}
            onBlur={() => { const v = clampInt(nGpu, -1, 200, -1); setNGpu(v); onChange({ n_gpu_layers: v }) }}
          />
        </div>
        <div className="settings-hint mono">-1 = all on GPU · lower it if a big model won&apos;t fit</div>
      </div>
    </>
  )
}

function SettingsPanel({ conversation, disabled, onChange }) {
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
    <div className="settings-wrap" ref={wrapRef}>
      <button
        type="button"
        className="settings-trigger"
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Model settings"
        onClick={() => setOpen((o) => !o)}
      >
        <SettingsIcon />
      </button>
      {open && conversation && (
        <div className="settings-panel" role="dialog" aria-label="Model settings">
          <SettingsFields key={conversation.id} conversation={conversation} onChange={onChange} />
        </div>
      )}
    </div>
  )
}

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
        placeholder="Ask a Python question..."
        rows={1}
      />
      <button className="mono" type="submit" disabled={sending || !draft.trim()}>send</button>
    </form>
  )
}

function ModelDropdown({ activeModel, modelOptions, models, disabled, onChange, onBrowse }) {
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
        <span className="model-name mono">{activeModel ? basename(activeModel) : 'no model'}</span>
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
                {models.includes(m) ? basename(m) : `${basename(m)} (not recently used)`}
              </span>
              {m === activeModel && <span className="model-option-check">&#10003;</span>}
            </button>
          ))}
          <button
            type="button"
            className="model-option model-option-browse"
            onClick={() => {
              setOpen(false)
              onBrowse()
            }}
          >
            <span className="model-option-name mono">+ browse for a .gguf file&hellip;</span>
          </button>
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

export default function ChatWindow({ conversation, messages, models, onChangeModel, onBrowseModel, onChangeSettings, onSend, sending }) {
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

  const activeModel = conversation?.model ?? models[0] ?? null
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
          onBrowse={onBrowseModel}
        />
        <SettingsPanel
          conversation={conversation}
          disabled={!conversation}
          onChange={onChangeSettings}
        />
      </div>

      {!conversation ? (
        <div className="empty-state">
          <div className="empty-sq" />
          <div className="empty-caption mono">ready to run &middot; {activeModel ? basename(activeModel) : 'no model yet'}</div>
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
              {activeModel ? basename(activeModel) : 'no model'} &middot; local inference via llama.cpp, no data leaves this machine
            </div>
          </div>
        </>
      )}
    </div>
  )
}
