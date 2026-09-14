import { useMemo, useState } from 'react'

function groupOf(createdAtSeconds) {
  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const today = startOfDay(new Date())
  const day = startOfDay(new Date(createdAtSeconds * 1000))
  const diffDays = Math.round((today - day) / 86_400_000)
  if (diffDays <= 0) return 'today'
  if (diffDays === 1) return 'yesterday'
  return 'earlier'
}

const GROUP_ORDER = ['today', 'yesterday', 'earlier']

export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, isDark, onToggleDark }) {
  const [search, setSearch] = useState('')

  const groups = useMemo(() => {
    const q = search.trim().toLowerCase()
    const filtered = q
      ? conversations.filter((c) => c.title.toLowerCase().includes(q))
      : conversations
    return GROUP_ORDER.map((label) => ({
      label,
      chats: filtered.filter((c) => groupOf(c.created_at) === label),
    })).filter((g) => g.chats.length > 0)
  }, [conversations, search])

  return (
    <aside className="sidebar">
      <div className="wordmark">
        <div className="wordmark-sq" />
        <span className="mono">SO CHAT</span>
      </div>

      <button className="new-chat-btn mono" onClick={onNew}>&rarr; new chat</button>

      <input
        className="search-input mono"
        placeholder="search history"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="conversation-list">
        {groups.map((group) => (
          <div key={group.label}>
            <div className="conversation-group-label mono">{group.label}</div>
            <div className="conversation-group-rows">
              {group.chats.map((c) => (
                <div
                  key={c.id}
                  className={c.id === activeId ? 'conversation-item active' : 'conversation-item'}
                  onClick={() => onSelect(c.id)}
                >
                  <div className="conversation-item-main">
                    <span className="conversation-title">{c.title}</span>
                    <span className="conversation-model mono">{c.model}</span>
                  </div>
                  <button
                    className="conversation-delete mono"
                    title="Delete conversation"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(c.id)
                    }}
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          </div>
        ))}
        {groups.length === 0 && (
          <div className="conversation-empty-note mono">
            {search ? `no chats match “${search}”` : 'no chats yet'}
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="avatar" />
        <span className="status-label mono">local &middot; offline</span>
        <button className="dark-toggle mono" onClick={onToggleDark}>
          {isDark ? '☀ light' : '☮ dark'}
        </button>
      </div>
    </aside>
  )
}
