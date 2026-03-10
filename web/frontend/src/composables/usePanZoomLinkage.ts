import { computed, ref, watch } from 'vue'
import type { SelectOption } from 'naive-ui'
import type { PanZoomGroupMap } from './usePanZoomGroups'
import i18n from '../locales'

export type LinkageMode = 'linked' | 'independent' | 'custom'

export function getLinkageModeOptions(): SelectOption[] {
  const { t } = i18n.global
  return [
    { label: t('result.linkMode.all'), value: 'linked' },
    { label: t('result.linkMode.independent'), value: 'independent' },
    { label: t('result.linkMode.custom'), value: 'custom' },
  ]
}

export function getLinkageGroupOptions(): SelectOption[] {
  const { t } = i18n.global
  return [
    { label: t('result.linkGroup.a'), value: 'A' },
    { label: t('result.linkGroup.b'), value: 'B' },
    { label: t('result.linkGroup.independent'), value: '__self__' },
  ]
}

type PanZoomGroupsController = {
  setGroups: (groups: PanZoomGroupMap) => void
  getViewGroup: (viewId: string) => string
  setViewGroup: (viewId: string, groupId: string) => void
  resetAll: () => void
}

type UsePanZoomLinkageOptions<TViewId extends string> = {
  panZoomGroups: PanZoomGroupsController
  linkedGroups: Record<TViewId, string>
  independentGroups: Record<TViewId, string>
}

export function usePanZoomLinkage<TViewId extends string>({
  panZoomGroups,
  linkedGroups,
  independentGroups,
}: UsePanZoomLinkageOptions<TViewId>) {
  const linkageMode = ref<LinkageMode>('linked')

  function applyLinkageMode(mode: LinkageMode): void {
    if (mode === 'custom') return
    panZoomGroups.setGroups(mode === 'linked' ? linkedGroups : independentGroups)
    panZoomGroups.resetAll()
  }

  function groupValueFor(viewId: TViewId): string {
    const group = panZoomGroups.getViewGroup(viewId)
    return group.startsWith('self:') ? '__self__' : group
  }

  function setViewGroup(viewId: TViewId, value: string): void {
    panZoomGroups.setViewGroup(viewId, value === '__self__' ? `self:${viewId}` : value)
    panZoomGroups.resetAll()
  }

  watch(
    () => linkageMode.value,
    (mode) => {
      applyLinkageMode(mode)
    },
    { immediate: true },
  )

  const computedLinkageModeOptions = computed(() => getLinkageModeOptions())
  const computedLinkageGroupOptions = computed(() => getLinkageGroupOptions())

  return {
    linkageMode,
    linkageModeOptions: computedLinkageModeOptions,
    linkageGroupOptions: computedLinkageGroupOptions,
    applyLinkageMode,
    groupValueFor,
    setViewGroup,
  }
}
