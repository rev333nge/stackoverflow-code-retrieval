import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

// The trace badge is the point of the whole pipeline: it shows *why* the
// assistant answered the way it did (routed straight from the model, or
// grounded in retrieved StackOverflow answers), instead of hiding it.
function TraceBadge({ message }) {
  if (message.role !== 'assistant' || !message.route) return null

  if (message.route === 'ANSWER') {
    return <div className="trace-badge mono">answered directly (no search needed)</div>
  }

  const cosine = message.top_cosine?.toFixed(2)
  if (message.gate === 'open') {
    return (
      <div className="trace-badge trace-badge-search mono">
        searched StackOverflow (confidence {cosine}) &middot; grounded in {message.sources?.length ?? 0} answer(s)
      </div>
    )
  }
  return (
    <div className="trace-badge mono">
      searched StackOverflow but confidence was low ({cosine}) &middot; answered from model knowledge instead
    </div>
  )
}

function SourcesIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M2 3H10M2 6H10M2 9H7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  )
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="message-sources">
      <button
        type="button"
        className="sources-toggle mono"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <SourcesIcon />
        <span>sources</span>
        <span className="sources-count">{sources.length}</span>
        <span className={`sources-chevron${open ? ' open' : ''}`}>&#9656;</span>
      </button>

      <div className={`sources-collapse${open ? ' open' : ''}`}>
        <div className="sources-collapse-inner">
          <ol className="source-list" key={open ? 'open' : 'closed'}>
            {sources.map((raw, i) => {
              // Conversations saved before sources carried links have plain
              // title strings -- show those without a broken/empty href.
              const s = typeof raw === 'string' ? { title: raw, url: null } : raw
              return (
                <li key={s.url ?? i} className="source-row" style={{ '--stagger': i }}>
                  <span className="source-index mono">{i + 1}</span>
                  {s.url ? (
                    <a
                      className="source-link"
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={s.title}
                    >
                      <span className="source-title">{s.title}</span>
                      <span className="source-arrow" aria-hidden="true">&#8599;</span>
                    </a>
                  ) : (
                    <span className="source-link source-link-plain" title={s.title}>
                      <span className="source-title">{s.title}</span>
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>
      </div>
    </div>
  )
}

export default function Message({ message }) {
  const isUser = message.role === 'user'
  return (
    <div className="message">
      <div className={`message-label mono ${isUser ? 'user' : 'assistant'}`}>
        &rarr; {isUser ? 'USER' : 'ASSISTANT'}
      </div>
      <div className={message.isError ? 'message-body message-error' : 'message-body'}>
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
      <TraceBadge message={message} />
      {message.used_docs && message.sources?.length > 0 && <Sources sources={message.sources} />}
    </div>
  )
}
