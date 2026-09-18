# PY Retriever (desktop app)

Electron + React frontend for the Stack Overflow retrieval/RAG pipeline. See the [top-level README](../README.md) for what the app does and how to run it; this folder is just the UI shell.

## Layout

```
electron/   Electron main process and preload script
src/        React UI (chat window, sidebar, message rendering)
```

## Development

```bash
npm install
npm run dev
```

`npm run dev` runs the Vite dev server and Electron together. `npm run build` produces a production Vite build; `npm run lint` runs Oxlint.
