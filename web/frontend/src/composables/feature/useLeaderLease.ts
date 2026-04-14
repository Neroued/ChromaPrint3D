import { onMounted, onUnmounted, readonly, ref } from 'vue'

type LeaderHeartbeatMessage = {
  type: 'heartbeat'
  tabId: string
  expiresAt: number
}

type LeaderResignMessage = {
  type: 'resign'
  tabId: string
}

type LeaderLeaseMessage = LeaderHeartbeatMessage | LeaderResignMessage

type BroadcastChannelLike = {
  onmessage: ((event: MessageEvent<LeaderLeaseMessage>) => void) | null
  postMessage: (message: LeaderLeaseMessage) => void
  close: () => void
}

type BroadcastChannelCtor = new (name: string) => BroadcastChannelLike

export type LeaderLeaseOptions = {
  channelName: string
  leaseMs?: number
  heartbeatMs?: number
  electionDelayMs?: number
  randomBackoffMs?: number
  now?: () => number
  random?: () => number
  createTabId?: () => string
  BroadcastChannelCtor?: BroadcastChannelCtor | undefined
}

const DEFAULT_LEASE_MS = 15000
const DEFAULT_HEARTBEAT_MS = 5000
const DEFAULT_ELECTION_DELAY_MS = 1200
const DEFAULT_RANDOM_BACKOFF_MS = 800

function defaultNow(): number {
  return Date.now()
}

function defaultRandom(): number {
  return Math.random()
}

function defaultCreateTabId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.()
  if (randomUuid) return randomUuid
  return `tab-${Date.now()}-${Math.floor(Math.random() * 1e9)}`
}

function compareTabIds(a: string, b: string): number {
  if (a === b) return 0
  return a < b ? -1 : 1
}

function isHeartbeatMessage(data: LeaderLeaseMessage): data is LeaderHeartbeatMessage {
  return data.type === 'heartbeat'
}

function resolveBroadcastChannelCtor(
  options: LeaderLeaseOptions,
): BroadcastChannelCtor | undefined {
  if ('BroadcastChannelCtor' in options) return options.BroadcastChannelCtor
  if (typeof BroadcastChannel === 'undefined') return undefined
  return BroadcastChannel as unknown as BroadcastChannelCtor
}

export function createLeaderLeaseController(options: LeaderLeaseOptions) {
  const leaseMs = options.leaseMs ?? DEFAULT_LEASE_MS
  const heartbeatMs = options.heartbeatMs ?? DEFAULT_HEARTBEAT_MS
  const electionDelayMs = options.electionDelayMs ?? DEFAULT_ELECTION_DELAY_MS
  const randomBackoffMs = options.randomBackoffMs ?? DEFAULT_RANDOM_BACKOFF_MS
  const now = options.now ?? defaultNow
  const random = options.random ?? defaultRandom
  const tabId = (options.createTabId ?? defaultCreateTabId)()

  const isLeader = ref(false)

  let started = false
  let channel: BroadcastChannelLike | null = null
  let knownLeaderId: string | null = null
  let knownLeaderExpiresAt = 0
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let electionTimer: ReturnType<typeof setTimeout> | null = null

  function clearElectionTimer() {
    if (electionTimer !== null) {
      clearTimeout(electionTimer)
      electionTimer = null
    }
  }

  function clearHeartbeatTimer() {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer)
      heartbeatTimer = null
    }
  }

  function nextBackoffMs(baseDelayMs = electionDelayMs): number {
    return baseDelayMs + Math.floor(random() * randomBackoffMs)
  }

  function publishHeartbeat() {
    if (!started || !channel) return
    const expiresAt = now() + leaseMs
    knownLeaderId = tabId
    knownLeaderExpiresAt = expiresAt
    channel.postMessage({
      type: 'heartbeat',
      tabId,
      expiresAt,
    })
  }

  function startHeartbeatLoop() {
    clearHeartbeatTimer()
    publishHeartbeat()
    heartbeatTimer = setInterval(() => {
      publishHeartbeat()
    }, heartbeatMs)
  }

  function scheduleElection(delayMs = electionDelayMs) {
    if (!started || isLeader.value) return
    clearElectionTimer()
    electionTimer = setTimeout(() => {
      attemptLeadership()
    }, nextBackoffMs(delayMs))
  }

  function scheduleElectionAtLeaseExpiry() {
    if (!started || isLeader.value) return
    const delayMs = Math.max(0, knownLeaderExpiresAt - now())
    scheduleElection(delayMs)
  }

  function preferCandidate(candidateId: string, candidateExpiresAt: number): boolean {
    const currentTime = now()
    if (!knownLeaderId || knownLeaderExpiresAt <= currentTime) return true
    if (candidateId === knownLeaderId) return candidateExpiresAt >= knownLeaderExpiresAt
    return compareTabIds(candidateId, knownLeaderId) < 0
  }

  function stepDownToLeader(leaderId: string, expiresAt: number) {
    clearHeartbeatTimer()
    isLeader.value = false
    knownLeaderId = leaderId
    knownLeaderExpiresAt = expiresAt
    scheduleElectionAtLeaseExpiry()
  }

  function claimLeadership() {
    clearElectionTimer()
    isLeader.value = true
    startHeartbeatLoop()
  }

  function handleHeartbeat(message: LeaderHeartbeatMessage) {
    if (message.tabId === tabId) {
      knownLeaderId = tabId
      knownLeaderExpiresAt = message.expiresAt
      return
    }
    if (message.expiresAt <= now()) return

    if (isLeader.value) {
      if (compareTabIds(message.tabId, tabId) < 0) {
        stepDownToLeader(message.tabId, message.expiresAt)
      } else {
        publishHeartbeat()
      }
      return
    }

    if (preferCandidate(message.tabId, message.expiresAt)) {
      knownLeaderId = message.tabId
      knownLeaderExpiresAt = message.expiresAt
    }
    scheduleElectionAtLeaseExpiry()
  }

  function handleResign(tabIdToResign: string) {
    if (tabIdToResign !== knownLeaderId) return
    knownLeaderId = null
    knownLeaderExpiresAt = 0
    scheduleElection(0)
  }

  function handleMessage(event: MessageEvent<LeaderLeaseMessage>) {
    const message = event.data
    if (!message) return
    if (isHeartbeatMessage(message)) {
      handleHeartbeat(message)
      return
    }
    handleResign(message.tabId)
  }

  function attemptLeadership() {
    if (!started || isLeader.value) return
    const currentTime = now()
    if (knownLeaderId !== null && knownLeaderId !== tabId && knownLeaderExpiresAt > currentTime) {
      scheduleElectionAtLeaseExpiry()
      return
    }
    claimLeadership()
  }

  function start() {
    if (started) return
    started = true

    const BroadcastCtor = resolveBroadcastChannelCtor(options)
    if (!BroadcastCtor) {
      knownLeaderId = tabId
      knownLeaderExpiresAt = Number.POSITIVE_INFINITY
      isLeader.value = true
      return
    }

    channel = new BroadcastCtor(options.channelName)
    channel.onmessage = handleMessage
    scheduleElection(0)
  }

  function stop() {
    if (!started) return
    started = false
    clearElectionTimer()
    clearHeartbeatTimer()
    if (isLeader.value && channel) {
      channel.postMessage({
        type: 'resign',
        tabId,
      })
    }
    channel?.close()
    channel = null
    knownLeaderId = null
    knownLeaderExpiresAt = 0
    isLeader.value = false
  }

  return {
    isLeader: readonly(isLeader),
    start,
    stop,
  }
}

export function useLeaderLease(options: LeaderLeaseOptions) {
  const controller = createLeaderLeaseController(options)
  onMounted(() => {
    controller.start()
  })
  onUnmounted(() => {
    controller.stop()
  })
  return controller
}
