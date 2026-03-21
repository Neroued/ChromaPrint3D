<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NModal, NCard, NButton, NSpace, NText } from 'naive-ui'
import { useWhatsNew } from '../composables/feature/useWhatsNew'

const { t } = useI18n()
const { visible, version, changelogLines, markAsSeen } = useWhatsNew()
</script>

<template>
  <NModal v-model:show="visible" :mask-closable="true" @after-leave="markAsSeen">
    <NCard
      class="whats-new-card"
      :title="t('app.whatsNew.title', { version })"
      :bordered="false"
      role="dialog"
      aria-modal="true"
      closable
      @close="visible = false"
    >
      <NSpace vertical :size="6">
        <ul v-if="changelogLines.length" class="whats-new-card__list">
          <li v-for="(line, i) in changelogLines" :key="i">
            <NText>{{ line }}</NText>
          </li>
        </ul>
      </NSpace>
      <template #footer>
        <NSpace justify="end">
          <NButton type="primary" @click="visible = false">
            {{ t('app.whatsNew.gotIt') }}
          </NButton>
        </NSpace>
      </template>
    </NCard>
  </NModal>
</template>

<style scoped>
.whats-new-card {
  width: min(480px, 90vw);
}

.whats-new-card__list {
  margin: 0;
  padding-left: 20px;
}

.whats-new-card__list li {
  margin: 6px 0;
  line-height: 1.6;
}
</style>
