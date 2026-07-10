import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // WSL: this project lives on the /mnt/c Windows-drive mount, where native
  // fs-change notifications don't reliably reach chokidar — without polling,
  // edits never trigger HMR and the dev server just keeps serving what it
  // first read.
  server: {
    watch: {
      usePolling: true,
    },
  },
})
