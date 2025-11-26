import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // This ensures that all unhandled requests are routed to index.html
    // which is necessary for client-side routing to work correctly with refresh.
    historyApiFallback: true,
    // Optional: If you need to specify a base for your routes
    // base: '/',
  },
})
