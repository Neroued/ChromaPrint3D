import { describe, expect, it, vi } from 'vitest'
import { createLeaderLeaseController } from './useLeaderLease'

type LeaseMessage =
  | { type: 'heartbeat'; tabId: string; expiresAt: number }
  | { type: 'resign'; tabId: string }

class FakeBroadcastChannel {
  static channels = new Map<string, Set<FakeBroadcastChannel>>()

  onmessage: ((event: MessageEvent<LeaseMessage>) => void) | null = null
  private readonly name: string

  constructor(name: string) {
    this.name = name
    const peers = FakeBroadcastChannel.channels.get(name) ?? new Set<FakeBroadcastChannel>()
    peers.add(this)
    FakeBroadcastChannel.channels.set(name, peers)
  }

  postMessage(message: LeaseMessage) {
    const peers = FakeBroadcastChannel.channels.get(this.name)
    if (!peers) return
    for (const peer of peers) {
      if (peer === this) continue
      peer.onmessage?.({ data: message } as MessageEvent<LeaseMessage>)
    }
  }

  close() {
    const peers = FakeBroadcastChannel.channels.get(this.name)
    peers?.delete(this)
    if (peers && peers.size === 0) {
      FakeBroadcastChannel.channels.delete(this.name)
    }
  }

  static reset() {
    FakeBroadcastChannel.channels.clear()
  }
}

describe('createLeaderLeaseController', () => {
  it('在不支持 BroadcastChannel 时自动成为 leader', () => {
    const controller = createLeaderLeaseController({
      channelName: 'memory',
      BroadcastChannelCtor: undefined,
      createTabId: () => 'tab-a',
    })

    controller.start()
    expect(controller.isLeader.value).toBe(true)
    controller.stop()
  })

  it('多个 tab 中只保留字典序更小的 leader', async () => {
    vi.useFakeTimers()
    FakeBroadcastChannel.reset()

    const common = {
      channelName: 'memory',
      leaseMs: 100,
      heartbeatMs: 20,
      electionDelayMs: 0,
      randomBackoffMs: 0,
      random: () => 0,
      BroadcastChannelCtor: FakeBroadcastChannel,
    } as const

    const leader = createLeaderLeaseController({
      ...common,
      createTabId: () => 'tab-a',
    })
    const follower = createLeaderLeaseController({
      ...common,
      createTabId: () => 'tab-b',
    })

    leader.start()
    follower.start()
    await vi.advanceTimersByTimeAsync(1)

    expect(leader.isLeader.value).toBe(true)
    expect(follower.isLeader.value).toBe(false)

    leader.stop()
    follower.stop()
    vi.useRealTimers()
  })

  it('leader 停止后其他 tab 会自动接棒', async () => {
    vi.useFakeTimers()
    FakeBroadcastChannel.reset()

    const common = {
      channelName: 'memory',
      leaseMs: 120,
      heartbeatMs: 20,
      electionDelayMs: 0,
      randomBackoffMs: 0,
      random: () => 0,
      BroadcastChannelCtor: FakeBroadcastChannel,
    } as const

    const first = createLeaderLeaseController({
      ...common,
      createTabId: () => 'tab-a',
    })
    const second = createLeaderLeaseController({
      ...common,
      createTabId: () => 'tab-b',
    })

    first.start()
    second.start()
    await vi.advanceTimersByTimeAsync(1)
    expect(first.isLeader.value).toBe(true)
    expect(second.isLeader.value).toBe(false)

    first.stop()
    await vi.advanceTimersByTimeAsync(1)

    expect(second.isLeader.value).toBe(true)

    second.stop()
    vi.useRealTimers()
  })
})
