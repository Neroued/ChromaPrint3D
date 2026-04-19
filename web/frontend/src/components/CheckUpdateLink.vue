<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NText, useMessage } from 'naive-ui'
import { useUpdateChecker } from '../composables/feature/useUpdateChecker'
import { trackEvent } from '../services/analytics'

const { t } = useI18n()
const message = useMessage()
const { checking, checkForUpdate, hasUpdate, lastCheckFailed } = useUpdateChecker()

async function handleClick() {
  const found = await checkForUpdate()
  const result: 'has-update' | 'no-update' | 'fail' = found
    ? 'has-update'
    : lastCheckFailed.value
      ? 'fail'
      : 'no-update'
  trackEvent('check-update-click', { result })
  if (!found && !hasUpdate.value) {
    if (lastCheckFailed.value) {
      message.warning(t('app.update.checkFailed'))
    } else {
      message.success(t('app.update.upToDate'))
    }
  }
}
</script>

<template>
  <span class="check-update-link" @click="handleClick">
    <NText depth="3" class="app-shell__meta-text check-update-link__text">
      {{ checking ? t('app.update.checking') : t('app.update.checkForUpdate') }}
    </NText>
  </span>
</template>

<style scoped>
.check-update-link {
  cursor: pointer;
}

.check-update-link__text:hover {
  text-decoration: underline;
}
</style>
