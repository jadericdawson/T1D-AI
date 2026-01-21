import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Polyfill Node.js modules for Plotly.js (buffer, stream, etc.)
    nodePolyfills({
      include: ['buffer', 'stream', 'assert', 'util'],
      globals: {
        Buffer: true,
      },
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate Plotly into its own chunk for better caching
          plotly: ['plotly.js', 'react-plotly.js'],
          // Separate vendor libs
          vendor: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
})
