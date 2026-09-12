export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">
      <button className="new-chat-btn" onClick={onNew}>+ New chat</button>
      <ul className="conversation-list">
        {conversations.map((c) => (
          <li
            key={c.id}
            className={c.id === activeId ? 'conversation-item active' : 'conversation-item'}
            onClick={() => onSelect(c.id)}
          >
            <span className="conversation-title">{c.title}</span>
            <button
              className="conversation-delete"
              title="Delete conversation"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(c.id)
              }}
            >
              x
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
