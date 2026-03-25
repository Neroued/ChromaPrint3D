<script setup lang="ts">
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import {
  NConfigProvider,
  NLayout,
  NLayoutHeader,
  NLayoutContent,
  NLayoutFooter,
  NSpace,
  NGrid,
  NGridItem,
  NText,
  NTag,
  NTabs,
  NTabPane,
  NMessageProvider,
  NSwitch,
  NButton,
  darkTheme,
  zhCN,
  dateZhCN,
  enUS,
  dateEnUS,
} from 'naive-ui'
import ImageUpload from './components/ImageUpload.vue'
import ParamPanel from './components/ParamPanel.vue'
import ConvertPanel from './components/ConvertPanel.vue'
import ResultPanel from './components/ResultPanel.vue'
import RecipeEditorPanel from './components/recipeEditor/RecipeEditorPanel.vue'
import CalibrationPanel from './components/CalibrationPanel.vue'
import Calibration8ColorPanel from './components/Calibration8ColorPanel.vue'
import MattingPanel from './components/MattingPanel.vue'
import VectorizePanel from './components/VectorizePanel.vue'
import ColorDBBuildSection from './components/calibration/ColorDBBuildSection.vue'
import ColorDBUploadSection from './components/calibration/ColorDBUploadSection.vue'
import UpdateBanner from './components/UpdateBanner.vue'
import WhatsNewModal from './components/WhatsNewModal.vue'
import CheckUpdateLink from './components/CheckUpdateLink.vue'
import { darkThemeOverrides, lightThemeOverrides } from './theme'
import { useAppStore } from './stores/app'
import { useAppLifecycle } from './composables/feature/useAppLifecycle'
import { useUpdateChecker } from './composables/feature/useUpdateChecker'
import { useWhatsNew } from './composables/feature/useWhatsNew'
import { isElectronRuntime } from './runtime/platform'
import {
  getSiteIcpNumber,
  getSiteIcpUrl,
  getSitePublicSecurityRecordNumber,
  getSitePublicSecurityRecordUrl,
} from './runtime/env'

const { t, locale } = useI18n()

const appStore = useAppStore()
const {
  activeTab,
  serverOnline,
  serverVersion,
  activeTasks,
  totalTasks,
  isDark,
  recipeEditorTaskId,
  completedTask,
} = storeToRefs(appStore)

const activeTheme = computed(() => (isDark.value ? darkTheme : null))
const activeThemeOverrides = computed(() =>
  isDark.value ? darkThemeOverrides : lightThemeOverrides,
)
const naiveLocale = computed(() => (locale.value === 'zh-CN' ? zhCN : enUS))
const naiveDateLocale = computed(() => (locale.value === 'zh-CN' ? dateZhCN : dateEnUS))
const makerWorldUrl = computed(() =>
  locale.value === 'zh-CN'
    ? 'https://makerworld.com.cn/@Neroued'
    : 'https://makerworld.com/@Neroued',
)
const runtimeLabel = isElectronRuntime() ? 'Electron' : 'Browser'
const siteIcpNumber = getSiteIcpNumber()
const siteIcpUrl = getSiteIcpUrl()
const sitePublicSecurityRecordNumber = getSitePublicSecurityRecordNumber()
const sitePublicSecurityRecordUrl = getSitePublicSecurityRecordUrl()
const showSiteIcpRecord = !isElectronRuntime() && Boolean(siteIcpNumber)
const showSitePublicSecurityRecord = !isElectronRuntime() && Boolean(sitePublicSecurityRecordNumber)
const activePreprocessTab = ref('vectorize')
const activeCalibrationTab = ref('calibration')

// 兼容旧 tab 名称（热更新或旧状态回放场景）。
if (activeTab.value === 'matting' || activeTab.value === 'vectorize') {
  activePreprocessTab.value = activeTab.value
  activeTab.value = 'preprocess'
}
if (
  activeTab.value === 'calibration' ||
  activeTab.value === 'calibration-8color' ||
  activeTab.value === 'colordb-build'
) {
  activeCalibrationTab.value = activeTab.value
  activeTab.value = 'calibration-tools'
}

useAppLifecycle()

const { initOnce: initUpdateChecker } = useUpdateChecker()
initUpdateChecker()

const { initOnce: initWhatsNew, show: showWhatsNew } = useWhatsNew()
initWhatsNew()

function handleColorDBUpdated() {
  appStore.refreshColorDBs()
}

function goToColorDBUpload() {
  activeTab.value = 'colordb-upload'
}

function goToColorDBBuild() {
  activeCalibrationTab.value = 'colordb-build'
}

function goToConvert() {
  activeTab.value = 'convert'
}

function toggleLocale() {
  appStore.setLocale(locale.value === 'zh-CN' ? 'en' : 'zh-CN')
}
</script>

<template>
  <NConfigProvider
    :theme="activeTheme"
    :theme-overrides="activeThemeOverrides"
    :locale="naiveLocale"
    :date-locale="naiveDateLocale"
  >
    <NMessageProvider>
      <NLayout class="app-shell">
        <NLayoutHeader bordered class="app-shell__header">
          <div class="app-shell__header-inner">
            <NSpace align="center" :size="12">
              <NText strong class="app-shell__brand-title">ChromaPrint3D</NText>
              <NText
                v-if="serverVersion"
                depth="3"
                class="app-shell__version app-shell__version--clickable"
                @click="showWhatsNew"
              >
                v{{ serverVersion }}
              </NText>
            </NSpace>
            <NSpace align="center" :size="10" class="app-shell__header-status">
              <NText v-if="serverOnline && totalTasks > 0" depth="3" class="app-shell__meta-text">
                {{
                  activeTasks > 0
                    ? t('common.activeTasks', { count: activeTasks })
                    : t('common.historyTasks', { count: totalTasks })
                }}
              </NText>
              <NTag size="small" :bordered="false" type="info">
                {{ runtimeLabel }}
              </NTag>
              <NSpace align="center" :size="6">
                <NText depth="3" class="app-shell__meta-text">{{ t('common.darkMode') }}</NText>
                <NSwitch v-model:value="isDark" size="small" />
              </NSpace>
              <NButton
                tag="a"
                href="https://www.bilibili.com/video/BV1cPATzyEz9/"
                target="_blank"
                size="small"
                quaternary
              >
                {{ t('common.videoTutorial') }}
              </NButton>
              <NButton size="small" quaternary @click="toggleLocale">
                {{ locale === 'zh-CN' ? 'EN' : '中' }}
              </NButton>
              <NTag :type="serverOnline ? 'success' : 'error'" size="small" round>
                {{ serverOnline ? t('common.serverOnline') : t('common.serverOffline') }}
              </NTag>
            </NSpace>
          </div>
        </NLayoutHeader>

        <NLayoutContent class="app-shell__content">
          <div class="app-shell__content-inner">
            <UpdateBanner />
            <WhatsNewModal />
            <div class="app-shell__announce">
              <div class="app-shell__announce-inner">
                <span class="app-shell__announce-text">
                  {{ t('app.announcement.starHint') }}
                  <a
                    href="https://github.com/neroued/ChromaPrint3D"
                    target="_blank"
                    rel="noopener noreferrer"
                    >{{ t('app.announcement.starLink') }}</a
                  >{{ t('app.announcement.starSuffix') }}
                  <span class="app-shell__announce-sep">&middot;</span>
                  {{ t('app.announcement.issueHint') }}
                  <a
                    href="https://github.com/neroued/ChromaPrint3D/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                    >{{ t('app.announcement.issueLink') }}</a
                  >
                  <span class="app-shell__announce-sep">&middot;</span>
                  {{ t('app.announcement.makerWorldHint') }}
                  <a :href="makerWorldUrl" target="_blank" rel="noopener noreferrer">{{
                    t('app.announcement.makerWorldLink')
                  }}</a
                  >{{ t('app.announcement.makerWorldSuffix') }}
                </span>
              </div>
            </div>
            <NTabs
              v-model:value="activeTab"
              type="line"
              size="large"
              animated
              class="top-level-tabs fade-in-up"
            >
              <NTabPane name="convert" :tab="t('app.tabs.convert')" display-directive="show">
                <NSpace vertical :size="20" class="tab-pane-content">
                  <div class="convert-colordb-entry">
                    <NText depth="3">{{ t('app.convertEntry.hint') }}</NText>
                    <NButton tertiary size="small" @click="goToColorDBUpload">
                      {{ t('app.convertEntry.goUpload') }}
                    </NButton>
                  </div>
                  <NGrid :cols="2" :x-gap="16" :y-gap="16" responsive="screen" item-responsive>
                    <NGridItem span="2 m:1">
                      <NSpace vertical :size="16">
                        <ImageUpload />
                        <ConvertPanel />
                      </NSpace>
                    </NGridItem>
                    <NGridItem span="2 m:1">
                      <ParamPanel />
                    </NGridItem>
                  </NGrid>
                  <RecipeEditorPanel v-if="recipeEditorTaskId" :task-id="recipeEditorTaskId" />
                  <ResultPanel
                    v-if="
                      !recipeEditorTaskId ||
                      (completedTask?.status === 'completed' && completedTask?.result?.has_3mf)
                    "
                  />
                </NSpace>
              </NTabPane>

              <NTabPane name="preprocess" :tab="t('app.tabs.preprocess')" display-directive="show">
                <div class="tab-pane-content">
                  <NTabs
                    v-model:value="activePreprocessTab"
                    type="segment"
                    animated
                    class="nested-tabs fade-in-up"
                  >
                    <NTabPane
                      name="vectorize"
                      :tab="t('app.tabs.vectorize')"
                      display-directive="show"
                    >
                      <div class="nested-tab-pane-content">
                        <VectorizePanel />
                      </div>
                    </NTabPane>
                    <NTabPane name="matting" display-directive="show">
                      <template #tab>
                        <NSpace :size="4" align="center">
                          <span>{{ t('app.tabs.matting') }}</span>
                          <NTag size="tiny" type="warning" :bordered="false">Beta</NTag>
                        </NSpace>
                      </template>
                      <div class="nested-tab-pane-content">
                        <MattingPanel />
                      </div>
                    </NTabPane>
                  </NTabs>
                </div>
              </NTabPane>

              <NTabPane
                name="calibration-tools"
                :tab="t('app.tabs.calibrationTools')"
                display-directive="show"
              >
                <div class="tab-pane-content">
                  <NTabs
                    v-model:value="activeCalibrationTab"
                    type="segment"
                    animated
                    class="nested-tabs fade-in-up"
                  >
                    <NTabPane
                      name="calibration"
                      :tab="t('app.tabs.calibration')"
                      display-directive="show"
                    >
                      <div class="nested-tab-pane-content">
                        <CalibrationPanel @go-colordb-build="goToColorDBBuild" />
                      </div>
                    </NTabPane>
                    <NTabPane
                      name="calibration-8color"
                      :tab="t('app.tabs.calibration8color')"
                      display-directive="show"
                    >
                      <div class="nested-tab-pane-content">
                        <Calibration8ColorPanel @go-colordb-build="goToColorDBBuild" />
                      </div>
                    </NTabPane>
                    <NTabPane
                      name="colordb-build"
                      :tab="t('app.tabs.colordbBuild')"
                      display-directive="show"
                    >
                      <div class="nested-tab-pane-content">
                        <NSpace vertical :size="20" class="calibration-layout">
                          <ColorDBBuildSection
                            :tips="t('colordb.build.standaloneTip')"
                            @colordb-updated="handleColorDBUpdated"
                          />
                        </NSpace>
                      </div>
                    </NTabPane>
                  </NTabs>
                </div>
              </NTabPane>

              <NTabPane
                name="colordb-upload"
                :tab="t('app.tabs.colordbUpload')"
                display-directive="show"
              >
                <div class="tab-pane-content">
                  <NSpace vertical :size="20" class="calibration-layout">
                    <ColorDBUploadSection
                      :title="t('app.colordbUploadSection.title')"
                      :tips="t('app.colordbUploadSection.tips')"
                      @colordb-updated="handleColorDBUpdated"
                      @go-convert="goToConvert"
                    />
                  </NSpace>
                </div>
              </NTabPane>
            </NTabs>
          </div>
        </NLayoutContent>

        <NLayoutFooter bordered class="app-shell__footer">
          <NSpace align="center" :size="16" justify="center" class="app-shell__footer-content">
            <NText depth="3" class="app-shell__meta-text">
              ChromaPrint3D{{ serverVersion ? ` v${serverVersion}` : '' }}
            </NText>
            <CheckUpdateLink />
            <NText depth="3" class="app-shell__meta-text">
              Multi-color 3D Print Image Processor
            </NText>
            <a
              v-if="showSiteIcpRecord"
              :href="siteIcpUrl"
              target="_blank"
              rel="noopener noreferrer"
              class="app-shell__footer-link"
            >
              <NText depth="3" class="app-shell__meta-text">{{ siteIcpNumber }}</NText>
            </a>
            <a
              v-if="showSitePublicSecurityRecord"
              :href="sitePublicSecurityRecordUrl"
              target="_blank"
              rel="noopener noreferrer"
              class="app-shell__footer-link"
            >
              <NText depth="3" class="app-shell__meta-text">
                {{ sitePublicSecurityRecordNumber }}
              </NText>
            </a>
            <a
              href="https://github.com/neroued/ChromaPrint3D"
              target="_blank"
              rel="noopener noreferrer"
              class="app-shell__footer-link"
            >
              <NText depth="3" class="app-shell__meta-text">GitHub</NText>
            </a>
            <a href="mailto:neroued@gmail.com" class="app-shell__footer-link">
              <NText depth="3" class="app-shell__meta-text">Neroued@gmail.com</NText>
            </a>
          </NSpace>
        </NLayoutFooter>
      </NLayout>
    </NMessageProvider>
  </NConfigProvider>
</template>
