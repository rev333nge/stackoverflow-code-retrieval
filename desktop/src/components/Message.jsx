import ReactMarkdown from 'react-markdown'

// The trace badge is the point of the whole pipeline: it shows *why* the
// assistant answered the way it did (routed straight from the model, or
// grounded in retrieved StackOverflow answers), instead of hiding it.
function TraceBadge({ message }) {
  if (message.role !== 'assistant' || !message.route) return null

  if (message.route === 'ANSWER') {
    return <div className="trace-badge">answered directly (no search needed)</div>
  }

  const cosine = message.top_cosine?.toFixed(2)
  if (message.gate === 'open') {
    return (
      <div className="trace-badge trace-badge-search">
        searched StackOverflow (confidence {cosine}) · grounded in {message.sources?.length ?? 0} answer(s)
      </div>
    )
  }
  return (
    <div className="trace-badge">
      searched StackOverflow but confidence was low ({cosine}) · answered from model knowledge instead
    </div>
  )
}

export default function Message({ message }) {
  return (
    <div className={`message message-${message.role}`}>
      <div className={message.isError ? 'message-bubble message-error' : 'message-bubble'}>
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
      <TraceBadge message={message} />
      {message.used_docs && message.sources?.length > 0 && (
        <details className="message-sources">
          <summary>Sources ({message.sources.length})</summary>
          <ul>
            {message.sources.map((title, i) => (
              <li key={i}>{title}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
