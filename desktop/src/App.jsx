import { useEffect, useRef, useState } from 'react'
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
  const [error, setError] = useState(null)

  // Always holds the *current* activeId, unlike a value closed over inside
  // handleSend -- lets an in-flight send detect that the user has since
  // switched to a different conversation before it appends its reply.
  const activeIdRef = useRef(activeId)
  useEffect(() => { activeIdRef.current = activeId }, [activeId])

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
    if (!model) {
      setError('No local models found. Make sure Ollama is running and has at least one model pulled.')
      return
    }
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
    const convId = activeConversation.id
    const isFirstMessage = messages.length === 0
    setMessages((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: text }])
    setSending(true)
    try {
      const reply = await api.sendMessage(convId, text)
      // The user may have switched to a different conversation while this
      // was in flight -- only splice the reply into the view it belongs to.
      if (activeIdRef.current === convId) {
        setMessages((prev) => [...prev, reply])
      }
      if (isFirstMessage) {
        const title = text.length > 40 ? `${text.slice(0, 40)}...` : text
        await api.updateConversation(convId, { title })
        setConversations((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, title } : c)),
        )
      }
    } catch (err) {
      if (activeIdRef.current === convId) {
        setMessages((prev) => [
          ...prev,
          { id: `error-${Date.now()}`, role: 'assistant', content: `Something went wrong: ${err.message}`, isError: true },
        ])
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
      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>&times;</button>
        </div>
      )}
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
