// Thin wrapper around the local FastAPI backend. Every call is same-origin
// in the packaged app (backend serves the built UI too) and CORS-allowed
// during development (Vite dev server -> 127.0.0.1:8756).

const BASE = 'http://127.0.0.1:8756/api'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`${options?.method ?? 'GET'} ${path} failed: ${res.status}`)
  return res.json()
}

export const api = {
  getModels: () => request('/models'),
  getConversations: () => request('/conversations'),
  createConversation: (title, model) =>
    request('/conversations', { method: 'POST', body: JSON.stringify({ title, model }) }),
  updateConversation: (id, { title, model } = {}) =>
    request(`/conversations/${id}`, { method: 'PATCH', body: JSON.stringify({ title, model }) }),
  deleteConversation: (id) => request(`/conversations/${id}`, { method: 'DELETE' }),
  getMessages: (id) => request(`/conversations/${id}/messages`),
  sendMessage: (id, content) =>
    request(`/conversations/${id}/messages`, { method: 'POST', body: JSON.stringify({ content }) }),
}
