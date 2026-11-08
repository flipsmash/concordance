import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Keep the previous build's hashed chunks: a tab opened before a deploy
  // still lazy-loads the chunk names it knows. Emptying dist on every build
  // 404'd them (a blank page until a hard refresh). Old chunks are tiny;
  // clear dist/assets by hand now and then if it grows.
  build: {
    emptyOutDir: false,
    // ".v2" in every built file name (2026-11-08): a one-time change so all
    // asset URLs differ from any a browser cached as a 404 during a deploy.
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-[hash].v2.js',
        chunkFileNames: 'assets/[name]-[hash].v2.js',
        assetFileNames: 'assets/[name]-[hash].v2[extname]',
      },
    },
  },
  // WSL: this project lives on the /mnt/c Windows-drive mount, where native
  // fs-change notifications don't reliably reach chokidar — without polling,
  // edits never trigger HMR and the dev server just keeps serving what it
  // first read.
  server: {
    watch: {
      usePolling: true,
    },
    // lets the frontend call relative /api/* paths in both dev (proxied here)
    // and when FastAPI serves the built app directly (same origin, no proxy
    // needed) — one code path for both.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
