import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import i18n from './locales'
import './styles/base.css'
import './styles/app-shell.css'
import './styles/calibration.css'

const app = createApp(App)
app.use(createPinia())
app.use(i18n)
app.mount('#app')
