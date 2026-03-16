import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import pkg from './package.json' with { type: 'json' }

const frontendTarget = (process.env.CHROMAPRINT3D_FRONTEND_TARGET ?? '').trim().toLowerCase()
const isElectronTarget = frontendTarget === 'electron'

export default defineConfig({
  base: isElectronTarget ? './' : '/',
  plugins: [vue()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'naive-ui': ['naive-ui'],
        },
      },
    },
  },
})
