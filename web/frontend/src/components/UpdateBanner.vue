<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { NAlert, NButton, NSpace, NText } from 'naive-ui'
import { useUpdateChecker } from '../composables/feature/useUpdateChecker'

const { t } = useI18n()
const { latestVersion, changelog, downloadUrl, showBanner, dismiss } = useUpdateChecker()

const changelogLines = computed(() => {
  const raw = changelog.value
  if (!raw) return []
  return raw
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
})
</script>

<template>
  <div v-if="showBanner" class="update-banner">
    <NAlert type="info" closable @close="dismiss">
      <template #header>
        {{ t('app.update.newVersionAvailable', { version: latestVersion }) }}
      </template>
      <NSpace vertical :size="8">
        <ul v-if="changelogLines.length" class="update-banner__changelog">
          <li v-for="(line, i) in changelogLines" :key="i">
            <NText depth="2">{{ line.replace(/^[-·•]\s*/, '') }}</NText>
          </li>
        </ul>
        <NSpace :size="12">
          <NButton
            v-if="downloadUrl"
            tag="a"
            :href="downloadUrl"
            target="_blank"
            rel="noopener noreferrer"
            type="info"
            size="small"
          >
            {{ t('app.update.download') }}
          </NButton>
        </NSpace>
      </NSpace>
    </NAlert>
  </div>
</template>

<style scoped>
.update-banner {
  margin-bottom: 12px;
}

.update-banner__changelog {
  margin: 0;
  padding-left: 18px;
}

.update-banner__changelog li {
  margin: 2px 0;
}
</style>
