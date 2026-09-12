// Electron main process: owns the native window and the Python backend's
// lifecycle. The backend (FastAPI + AdaptiveRAG) is spawned as a child
// process bound to 127.0.0.1 -- nothing here is reachable from the network.

import { app, BrowserWindow } from 'electron'
import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '..', '..')
const BACKEND_HOST = '127.0.0.1'
const BACKEND_PORT = 8756
const HEALTH_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}/api/health`
const DEV_URL = 'http://localhost:5173'

let backendProcess = null

function startBackend() {
  // Routed through cmd.exe /c so PATH resolution matches a normal terminal
  // (picks the real python.exe first). Spawning 'python' directly can land
  // on Windows' python.exe "app execution alias" stub instead and crash.
  backendProcess = spawn(
    'cmd.exe',
    ['/c', 'python', '-m', 'uvicorn', 'app.server:app', '--host', BACKEND_HOST, '--port', String(BACKEND_PORT)],
    { cwd: REPO_ROOT, stdio: 'inherit' },
  )
  backendProcess.on('error', (err) => {
    console.error('failed to start backend:', err)
  })
  backendProcess.on('exit', (code) => {
    console.log(`backend exited with code ${code}`)
  })
}

async function waitForBackend(timeoutMs = 120_000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(HEALTH_URL)
      if (res.ok) return true
    } catch {
      // backend not up yet, keep polling
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}

async function createWindow() {
  const ready = await waitForBackend()
  if (!ready) {
    console.error('backend did not become healthy in time')
  }

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    title: 'StackOverflow Retrieval',
    webPreferences: {
      contextIsolation: true,
    },
  })

  if (!app.isPackaged) {
    win.loadURL(DEV_URL)
  } else {
    win.loadURL(`http://${BACKEND_HOST}:${BACKEND_PORT}/`)
  }
}

app.whenReady().then(() => {
  startBackend()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  if (backendProcess) backendProcess.kill()
})
