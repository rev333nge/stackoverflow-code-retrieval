// CommonJS, not ESM (see main.js's comment on the preload path): Electron's
// preload loader wants require(), even though the rest of this package is
// "type": "module". Runs in an isolated world with access to Node + Electron
// APIs; contextBridge is what safely punches window.electronAPI through into
// the renderer (plain React/web code), which has neither.
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  pickGgufModel: () => ipcRenderer.invoke('pick-gguf-model'),
})
