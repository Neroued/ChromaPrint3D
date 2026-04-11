import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import i18n from './locales'
import './styles/base.css'
import './styles/app-shell.css'
import './styles/calibration.css'

const umamiHost = import.meta.env.VITE_UMAMI_HOST
const umamiWebsiteId = import.meta.env.VITE_UMAMI_WEBSITE_ID
if (umamiHost && umamiWebsiteId) {
  const script = document.createElement('script')
  script.defer = true
  script.src = `${umamiHost}/script.js`
  script.dataset.websiteId = umamiWebsiteId
  document.head.appendChild(script)
}

const app = createApp(App)
app.use(createPinia())
app.use(i18n)
app.mount('#app')
