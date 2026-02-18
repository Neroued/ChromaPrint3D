<script setup lang="ts">
import { computed } from 'vue'
import {
  NCard,
  NImage,
  NButton,
  NDescriptions,
  NDescriptionsItem,
  NSpace,
  NGrid,
  NGridItem,
  NText,
} from 'naive-ui'
import { getPreviewUrl, getSourceMaskUrl, getResultUrl } from '../api'
import type { TaskStatus } from '../types'

const props = defineProps<{
  task: TaskStatus | null
}>()

const isCompleted = computed(() => props.task?.status === 'completed')
const result = computed(() => props.task?.result ?? null)
const taskId = computed(() => props.task?.id ?? '')

function handleDownload3MF() {
  if (!taskId.value) return
  const url = getResultUrl(taskId.value)
  const a = document.createElement('a')
  a.href = url
  a.download = `${taskId.value.substring(0, 8)}.3mf`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}
</script>

<template>
  <NCard v-if="isCompleted && result" title="转换结果" size="small">
    <NSpace vertical :size="16">
      <!-- Preview images -->
      <NGrid :cols="2" :x-gap="16" :y-gap="16">
        <NGridItem v-if="result.has_preview">
          <NCard title="预览图" size="small" embedded>
            <NImage
              :src="getPreviewUrl(taskId)"
              fallback-src=""
              object-fit="contain"
              :img-props="{ style: 'max-width: 100%; max-height: 480px; object-fit: contain; cursor: zoom-in;' }"
              style="border-radius: 4px"
            />
          </NCard>
        </NGridItem>

        <NGridItem v-if="result.has_source_mask">
          <NCard title="颜色源掩码" size="small" embedded>
            <NImage
              :src="getSourceMaskUrl(taskId)"
              fallback-src=""
              object-fit="contain"
              :img-props="{ style: 'max-width: 100%; max-height: 480px; object-fit: contain; cursor: zoom-in;' }"
              style="border-radius: 4px"
            />
          </NCard>
        </NGridItem>
      </NGrid>

      <!-- Download button -->
      <NSpace v-if="result.has_3mf">
        <NButton type="primary" @click="handleDownload3MF">
          下载 3MF 文件
        </NButton>
        <NText depth="3" style="font-size: 12px; line-height: 34px">
          图像尺寸: {{ result.image_width }} × {{ result.image_height }}
        </NText>
      </NSpace>

      <!-- Match statistics -->
      <NDescriptions label-placement="left" bordered :column="2" size="small" title="匹配统计">
        <NDescriptionsItem label="聚类总数">
          {{ result.stats.clusters_total }}
        </NDescriptionsItem>
        <NDescriptionsItem label="数据库匹配">
          {{ result.stats.db_only }}
        </NDescriptionsItem>
        <NDescriptionsItem label="模型回退">
          {{ result.stats.model_fallback }}
        </NDescriptionsItem>
        <NDescriptionsItem label="模型查询">
          {{ result.stats.model_queries }}
        </NDescriptionsItem>
        <NDescriptionsItem label="数据库平均色差">
          {{ result.stats.avg_db_de.toFixed(2) }}
        </NDescriptionsItem>
        <NDescriptionsItem label="模型平均色差">
          {{ result.stats.avg_model_de.toFixed(2) }}
        </NDescriptionsItem>
      </NDescriptions>
    </NSpace>
  </NCard>
</template>
