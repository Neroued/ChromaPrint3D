<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NText, useMessage } from 'naive-ui'
import { useUpdateChecker } from '../composables/feature/useUpdateChecker'

const { t } = useI18n()
const message = useMessage()
const { checking, checkForUpdate, hasUpdate } = useUpdateChecker()

async function handleClick() {
  const found = await checkForUpdate()
  if (!found && !hasUpdate.value) {
    message.success(t('app.update.upToDate'))
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
