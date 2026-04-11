export function useTracking() {
  function trackEvent(name: string, data?: Record<string, unknown>) {
    window.umami?.track(name, data)
  }
  return { trackEvent }
}
