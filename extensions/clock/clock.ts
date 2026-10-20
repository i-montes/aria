// The line that tells the model when the user wrote: the local date and time, the weekday, the time zone and its UTC
// offset. Pure, so it can be tested with a fixed instant and zone.

const pad = (n: number) => String(n).padStart(2, '0')

/** `[now] 2026-10-19 22:15, Monday, America/Bogota (UTC-05:00)` for `date` seen in `timeZone`. */
export function nowLine(date: Date = new Date(), timeZone: string = Intl.DateTimeFormat().resolvedOptions().timeZone): string {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      weekday: 'long',
      hourCycle: 'h23',
    })
      .formatToParts(date)
      .map((p) => [p.type, p.value]),
  )
  // The offset is the zone's wall clock minus UTC, in minutes, for this instant (daylight saving included).
  const wall = Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day), Number(parts.hour), Number(parts.minute))
  const offset = Math.round((wall - Math.floor(date.getTime() / 60000) * 60000) / 60000)
  const sign = offset < 0 ? '-' : '+'
  const abs = Math.abs(offset)
  return `[now] ${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}, ${parts.weekday}, ${timeZone} (UTC${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)})`
}
