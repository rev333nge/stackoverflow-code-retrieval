import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import { api } from './api'
import './App.css'

const DEFAULT_TITLE = 'New chat'

export default function App() {
  const [backendReady, setBackendReady] = useState(false)
  const [models, setModels] = useState([])
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [sending, setSending] = useState(false)

  // Wait for the FastAPI backend (Electron starts it, but model/index
  // loading onto the GPU takes a while) before hitting any endpoint.
  useEffect(() => {
    let cancelled = false
    async function waitForBackend() {
      while (!cancelled) {
        try {
          const res = await fetch('http://127.0.0.1:8756/api/health')
          if (res.ok) {
            setBackendReady(true)
            return
          }
        } catch {
          // not up yet
        }
        await new Promise((r) => setTimeout(r, 500))
      }
    }
    waitForBackend()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!backendReady) return
    api.getModels().then(setModels)
    api.getConversations().then(setConversations)
  }, [backendReady])

  useEffect(() => {
    if (activeId == null) {
      setMessages([])
      return
    }
    api.getMessages(activeId).then(setMessages)
  }, [activeId])

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null

  async function handleNew() {
    const model = activeConversation?.model ?? models[0]
    if (!model) return
    const conv = await api.createConversation(DEFAULT_TITLE, model)
    setConversations((prev) => [conv, ...prev])
    setActiveId(conv.id)
  }

  async function handleDelete(id) {
    await api.deleteConversation(id)
    setConversations((prev) => prev.filter((c) => c.id !== id))
    if (id === activeId) setActiveId(null)
  }

  async function handleChangeModel(model) {
    if (!activeConversation) return
    await api.updateConversation(activeConversation.id, { model })
    setConversations((prev) =>
      prev.map((c) => (c.id === activeConversation.id ? { ...c, model } : c)),
    )
  }

  async function handleSend(text) {
    if (!activeConversation) return
    const isFirstMessage = messages.length === 0
    setMessages((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: text }])
    setSending(true)
    try {
      const reply = await api.sendMessage(activeConversation.id, text)
      setMessages((prev) => [...prev, reply])
      if (isFirstMessage) {
        const title = text.length > 40 ? `${text.slice(0, 40)}...` : text
        await api.updateConversation(activeConversation.id, { title })
        setConversations((prev) =>
          prev.map((c) => (c.id === activeConversation.id ? { ...c, title } : c)),
        )
      }
    } finally {
      setSending(false)
    }
  }

  if (!backendReady) {
    return <div className="loading-screen">Loading retrieval indexes and model...</div>
  }

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNew}
        onDelete={handleDelete}
      />
      <ChatWindow
        conversation={activeConversation}
        messages={messages}
        models={models}
        onChangeModel={handleChangeModel}
        onSend={handleSend}
        sending={sending}
      />
    </div>
  )
}
