import { useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import { api } from './api'
import './App.css'

const DEFAULT_TITLE = 'New chat'

export default function App() {
  const [backendReady, setBackendReady] = useState(false)
  const [backendFailed, setBackendFailed] = useState(false)
  const [retryTick, setRetryTick] = useState(0)
  const [models, setModels] = useState([])
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const [isDark, setIsDark] = useState(() => localStorage.getItem('theme') === 'dark')
  // Path of the .gguf the backend is currently loading (drives the top
  // loading bar), or null. Set from real backend state -- while a message is
  // in flight, we poll GET /api/status, which reports whichever model the
  // backend is loading into VRAM right now. This is authoritative (it knows
  // things the client can't guess, like the backend process having just
  // restarted and lost its cached model), unlike the earlier client-side
  // heuristic it replaces.
  const [loadingModelName, setLoadingModelName] = useState(null)
  // When loading ends, briefly hold the bar at 100% ("ready") so it reads as
  // finished rather than vanishing mid-fill. Holds the just-loaded model name
  // during that snap, or null. prevLoadingRef tracks the previous value so we
  // can detect the loading -> not-loading transition.
  const [barFinishingName, setBarFinishingName] = useState(null)
  const prevLoadingRef = useRef(null)

  useEffect(() => {
    const prev = prevLoadingRef.current
    prevLoadingRef.current = loadingModelName
    // loading -> not loading: model finished loading (generation begins).
    // Flash the bar to 100% for a moment before it disappears.
    if (prev && !loadingModelName) {
      setBarFinishingName(prev)
      const t = setTimeout(() => setBarFinishingName(null), 450)
      return () => clearTimeout(t)
    }
  }, [loadingModelName])

  useEffect(() => {
    document.documentElement.dataset.theme = isDark ? 'dark' : 'light'
    localStorage.setItem('theme', isDark ? 'dark' : 'light')
  }, [isDark])

  // Always holds the *current* activeId, unlike a value closed over inside
  // handleSend -- lets an in-flight send detect that the user has since
  // switched to a different conversation before it appends its reply.
  const activeIdRef = useRef(activeId)
  useEffect(() => { activeIdRef.current = activeId }, [activeId])

  // Wait for the FastAPI backend (Electron starts it, but model/index
  // loading onto the GPU takes a while) before hitting any endpoint. Give up
  // after a deadline instead of polling forever, so a backend that never comes
  // up (crash, port still held by a previous run) surfaces an actionable error
  // rather than a loading screen that hangs indefinitely. `retryTick` re-runs
  // this effect when the user clicks Retry.
  useEffect(() => {
    let cancelled = false
    const deadline = Date.now() + 120_000
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
        if (Date.now() > deadline) {
          setBackendFailed(true)
          return
        }
        await new Promise((r) => setTimeout(r, 500))
      }
    }
    waitForBackend()
    return () => { cancelled = true }
  }, [retryTick])

  useEffect(() => {
    if (!backendReady) return
    api.getModels().then(setModels)
    api.getConversations().then(setConversations)
  }, [backendReady])

  // When handleSend lazily creates a conversation and switches to it in the
  // same call, we already know it's empty and set that locally -- suppress
  // the fetch this effect would otherwise fire, which could resolve after
  // the optimistic user message is appended and wipe it back to [].
  const skipNextLoadRef = useRef(false)
  useEffect(() => {
    if (activeId == null) {
      setMessages([])
      return
    }
    if (skipNextLoadRef.current) {
      skipNextLoadRef.current = false
      return
    }
    api.getMessages(activeId).then(setMessages)
  }, [activeId])

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null

  // Opens the native "choose a .gguf file" dialog (implemented in Electron's
  // main process -- see electron/preload.cjs). Returns the picked absolute
  // path, or null if the user cancelled or this isn't running inside Electron.
  async function pickModel() {
    if (!window.electronAPI?.pickGgufModel) {
      setError('Model picker unavailable outside the desktop app.')
      return null
    }
    const picked = await window.electronAPI.pickGgufModel()
    if (picked) {
      setModels((prev) => (prev.includes(picked) ? prev : [picked, ...prev]))
    }
    return picked
  }

  async function handleBrowseModel() {
    const model = await pickModel()
    if (model) await handleChangeModel(model)
  }

  async function handleNew() {
    let model = activeConversation?.model ?? models[0]
    if (!model) {
      model = await pickModel()
      if (!model) return
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

  // patch is a subset of {temperature, max_tokens, n_ctx, n_gpu_layers}.
  // Load settings (n_ctx/n_gpu_layers) take effect on the next message, when
  // the backend reloads the model with them -- nothing to do here but persist.
  async function handleChangeSettings(patch) {
    if (!activeConversation) return
    await api.updateConversation(activeConversation.id, patch)
    setConversations((prev) =>
      prev.map((c) => (c.id === activeConversation.id ? { ...c, ...patch } : c)),
    )
  }

  async function handleSend(text) {
    let conv = activeConversation
    let isFirstMessage = messages.length === 0
    if (!conv) {
      let model = models[0]
      if (!model) {
        model = await pickModel()
        if (!model) return
      }
      conv = await api.createConversation(DEFAULT_TITLE, model)
      isFirstMessage = true
      skipNextLoadRef.current = true
      setConversations((prev) => [conv, ...prev])
      setActiveId(conv.id)
      setMessages([])
    }
    const convId = conv.id
    setMessages((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: text }])
    setSending(true)
    // Poll the backend's real loading state for as long as this send is in
    // flight: if the request blocks on a slow .gguf load, /api/status reports
    // which model, and the top bar shows it; once generation starts (or the
    // model was already warm) it reports null and the bar clears.
    const poll = setInterval(async () => {
      try {
        const { loading_model: loadingModel } = await api.getStatus()
        setLoadingModelName(loadingModel)
      } catch {
        // status endpoint momentarily unavailable -- ignore, try next tick
      }
    }, 400)
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
      clearInterval(poll)
      setLoadingModelName(null)
      setSending(false)
    }
  }

  if (!backendReady) {
    if (backendFailed) {
      return (
        <div className="loading-screen">
          <p>Couldn&apos;t reach the backend. Try again.</p>
          <button className="btn-retry mono" onClick={() => { setBackendFailed(false); setRetryTick((n) => n + 1) }}>retry</button>
        </div>
      )
    }
    return <div className="loading-screen mono">loading retrieval indexes and model&hellip;</div>
  }

  return (
    <div className="app">
      {(loadingModelName || barFinishingName) && (
        <div className="model-load-bar" role="status" aria-label={`loading model ${loadingModelName || barFinishingName}`}>
          <div className="model-load-track">
            <div className={`model-load-fill${barFinishingName ? ' done' : ''}`} />
          </div>
          <span className="model-load-label mono">
            {barFinishingName
              ? 'ready'
              : `loading ${loadingModelName.split(/[\\/]/).pop()}…`}
          </span>
        </div>
      )}
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
        isDark={isDark}
        onToggleDark={() => setIsDark((d) => !d)}
      />
      <ChatWindow
        conversation={activeConversation}
        messages={messages}
        models={models}
        onChangeModel={handleChangeModel}
        onBrowseModel={handleBrowseModel}
        onChangeSettings={handleChangeSettings}
        onSend={handleSend}
        sending={sending}
      />
    </div>
  )
}
