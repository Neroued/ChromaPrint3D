<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { NAlert, NSpace, NText, NButton } from 'naive-ui'
import type { Announcement, AnnouncementSeverity, AnnouncementType } from '../types'
import { useAnnouncements } from '../composables/useAnnouncements'

const { t, locale } = useI18n()
const announcements = useAnnouncements()

const now = ref(Date.now())
let tickTimer: number | null = null

onMounted(async () => {
  await announcements.initialize()
  tickTimer = window.setInterval(() => {
    now.value = Date.now()
  }, 1000)
})

onBeforeUnmount(() => {
  if (tickTimer !== null) {
    window.clearInterval(tickTimer)
    tickTimer = null
  }
})

function severityToAlertType(
  severity: AnnouncementSeverity,
): 'info' | 'warning' | 'error' | 'default' {
  if (severity === 'error') return 'error'
  if (severity === 'warning') return 'warning'
  return 'info'
}

function pickLocalized(zh: string | null, en: string | null): string {
  const zhText = zh ?? ''
  const enText = en ?? ''
  if (locale.value === 'zh-CN') {
    return zhText || enText
  }
  return enText || zhText
}

function titleFor(a: Announcement): string {
  const picked = pickLocalized(a.title.zh, a.title.en)
  if (picked) return picked
  return t(`app.announcements.typeLabels.${a.type}` as const)
}

function bodyFor(a: Announcement): string {
  return pickLocalized(a.body.zh, a.body.en)
}

// Plain-text split on newlines so the banner body supports bullet-style
// messages while NEVER invoking v-html. URLs are intentionally NOT
// auto-linkified — anchor rendering from untrusted content would be a
// policy violation.
function bodyLines(a: Announcement): string[] {
  const text = bodyFor(a)
  if (!text) return []
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
}

function formatLocalizedTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  try {
    return new Intl.DateTimeFormat(locale.value === 'zh-CN' ? 'zh-CN' : 'en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    }).format(date)
  } catch {
    return iso
  }
}

function relativeLabel(targetMs: number): string {
  const deltaSec = Math.round((targetMs - now.value) / 1000)
  if (deltaSec <= 0) return t('app.announcements.relative.now')
  if (deltaSec < 60) return t('app.announcements.relative.seconds', { count: deltaSec })
  const deltaMin = Math.round(deltaSec / 60)
  if (deltaMin < 60) return t('app.announcements.relative.minutes', { count: deltaMin })
  const deltaHours = Math.round(deltaMin / 60)
  if (deltaHours < 24) return t('app.announcements.relative.hours', { count: deltaHours })
  const deltaDays = Math.round(deltaHours / 24)
  return t('app.announcements.relative.days', { count: deltaDays })
}

function scheduledLine(a: Announcement): string | null {
  if (!a.scheduled_update_at) return null
  const ms = new Date(a.scheduled_update_at).getTime()
  if (Number.isNaN(ms)) return null
  const formatted = formatLocalizedTime(a.scheduled_update_at)
  if (ms <= now.value) {
    return t('app.announcements.scheduledUpdatePast', { time: formatted })
  }
  return t('app.announcements.scheduledUpdate', {
    time: formatted,
    relative: relativeLabel(ms),
  })
}

type BannerType = 'info' | 'warning' | 'error' | 'default'

interface Item {
  id: string
  alertType: BannerType
  title: string
  lines: string[]
  scheduled: string | null
  dismissible: boolean
  kind: AnnouncementType
}

const items = computed<Item[]>(() =>
  announcements.visible.value.map((a) => ({
    id: a.id,
    alertType: severityToAlertType(a.severity),
    title: titleFor(a),
    lines: bodyLines(a),
    scheduled: scheduledLine(a),
    dismissible: a.dismissible,
    kind: a.type,
  })),
)

async function handleDismiss(id: string) {
  await announcements.dismiss(id)
}
</script>

<template>
  <div v-if="items.length > 0" class="announcement-banner">
    <div v-for="item in items" :key="item.id" class="announcement-banner__item">
      <NAlert :type="item.alertType" :show-icon="true" :bordered="false">
        <template #header>
          <span class="announcement-banner__title">{{ item.title }}</span>
        </template>
        <NSpace vertical :size="6">
          <ul v-if="item.lines.length > 1" class="announcement-banner__lines">
            <li v-for="(line, idx) in item.lines" :key="idx">
              <NText depth="2">{{ line }}</NText>
            </li>
          </ul>
          <NText v-else-if="item.lines.length === 1" depth="2">{{ item.lines[0] }}</NText>
          <NText v-if="item.scheduled" depth="1" class="announcement-banner__scheduled">
            {{ item.scheduled }}
          </NText>
          <NSpace v-if="item.dismissible">
            <NButton size="small" quaternary @click="handleDismiss(item.id)">
              {{ t('app.announcements.dismiss') }}
            </NButton>
          </NSpace>
        </NSpace>
      </NAlert>
    </div>
  </div>
</template>

<style scoped>
.announcement-banner {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.announcement-banner__title {
  font-weight: 600;
}

.announcement-banner__lines {
  margin: 0;
  padding-left: 18px;
}

.announcement-banner__lines li {
  margin: 2px 0;
}

.announcement-banner__scheduled {
  font-size: 12px;
}
</style>
