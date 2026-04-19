import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import i18n from './locales'
import { initAnalytics } from './services/analytics'
import './styles/base.css'
import './styles/app-shell.css'
import './styles/calibration.css'

initAnalytics()

const app = createApp(App)
app.use(createPinia())
app.use(i18n)
app.mount('#app')
