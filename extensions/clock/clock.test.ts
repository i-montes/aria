// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { nowLine } from './clock.ts'

describe('the [now] line', () => {
  const instant = new Date(Date.UTC(2026, 9, 20, 3, 15)) // 2026-10-20 03:15 UTC

  test('gives the local date, time, weekday, zone and offset', () => {
    expect(nowLine(instant, 'America/Bogota')).toBe('[now] 2026-10-19 22:15, Monday, America/Bogota (UTC-05:00)')
  })

  test('follows the zone across midnight and with a half-hour offset', () => {
    expect(nowLine(instant, 'Asia/Kolkata')).toBe('[now] 2026-10-20 08:45, Tuesday, Asia/Kolkata (UTC+05:30)')
    expect(nowLine(instant, 'UTC')).toBe('[now] 2026-10-20 03:15, Tuesday, UTC (UTC+00:00)')
  })

  test('includes daylight saving time', () => {
    expect(nowLine(new Date(Date.UTC(2026, 6, 1, 12, 0)), 'Europe/Madrid')).toBe('[now] 2026-07-01 14:00, Wednesday, Europe/Madrid (UTC+02:00)')
  })

  test('the default is now, in the machine\'s zone', () => {
    expect(nowLine()).toMatch(/^\[now\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}, [A-Z][a-z]+day, \S+ \(UTC[+-]\d{2}:\d{2}\)$/)
  })
})
