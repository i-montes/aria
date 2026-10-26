# Auditoría Aria — checklist completa

Todo lo que hay que revisar del proyecto Aria (dock, terminal, extensiones, puente del navegador, datos).
Ir punto por punto; marcar la casilla al cerrarlo. Numeración continua para referirse a cualquier punto.
También en Obsidian (bóveda Prototype): `Auditoría Aria - checklist.md`.

## SEGURIDAD — secretos y credenciales

- [ ] **1.** `~/.aria/secrets/aria-ext-key.pem` (clave privada de la extensión): permisos 600 y decidir respaldo en llavero o destrucción
- [ ] **2.** `auth.json` (OpenCode key + Anthropic OAuth): symlink a `/var/folders/.../aria-secrets-*` — comprobar que el temp no lo limpia y deja el link colgando
- [ ] **3.** Auditar que `auth.json` no quede con valores en claro (el diseño: claves fijas al llavero, archivos sellados)
- [ ] **4.** `mcp.json`: la clave de Tavily está en texto plano (permisos 600) — decidir si pasa al llavero
- [ ] **5.** `mcp-auth.json` (OAuth de Notion y Figma): comprobar que está sellado en el llavero
- [ ] **6.** `secrets.json` de `~/.aria`: qué guarda y con qué permisos
- [ ] **7.** Los 6 `.env` de `lsv` siguen con contraseñas en texto (WordPress, WooCommerce, OAuth) — migrarlos a `$ARIA_SECRET_*`
- [ ] **8.** Contraseña de WordPress duplicada en `qnq` y `quienai` (2 copias) — rotar si alguna vez se filtró
- [ ] **9.** `WC_CONSUMER_KEY`/`WC_CONSUMER_SECRET` de WooCommerce: permisos y quién más los usa
- [ ] **10.** `~/.aria/bridge.token` (puente del navegador): solo 127.0.0.1 y token exigido siempre
- [ ] **11.** `aria-host.js`: no debe escribir rutas arbitrarias; límite de tamaño de mensajes; manejo de chunks
- [ ] **12.** Extensión Chrome: permisos mínimos (`<all_urls>`, `scripting`, `nativeMessaging`, `tabs`)
- [ ] **13.** `NativeMessagingHosts/com.aria.browser.json`: `allowed_origins` solo con el ID de la extensión
- [ ] **14.** Sandbox (bubblewrap en Linux): qué queda fuera del box; definir equivalente en macOS
- [ ] **15.** Grants del sandbox por programa+subcomando: listar todos los concedidos
- [ ] **16.** `connectors`: toda escritura a MCP pide confirmación — verificar que no haya excepciones
- [ ] **17.** Socket de jobs (`run/jobs-*.sock`): permisos y quién puede enviarle comandos
- [ ] **18.** Socket de control del dock: 0700 y un solo dock por equipo
- [ ] **19.** Buscar secretos pegados en `logs/`, `briefs/*.err.log`, trazas `/debug` y sesiones de Pi
- [ ] **20.** Memoria de Engram: auditar que no guarde valores de credenciales (solo nombres)
- [ ] **21.** Repo `i-montes/aria-extension`: seguir privado; la clave privada nunca debe subirse
- [ ] **22.** GitHub token del credential helper: dónde vive y qué alcance tiene
- [ ] **23.** Backups (Time Machine): confirmar que la copia viva de credenciales NO se copia
- [ ] **24.** Permisos de `~/.aria` (0700), `attachments/` (0700) y cada archivo sensible
- [ ] **25.** Cadena de suministro: `npx rea-agents@6.7.0` fijado ✓; auditar paquetes npm/bun descargados al vuelo y dependencias cargo
- [ ] **26.** Binarios (`~/.local/bin/aria`, `~/.aria/bin/*`): permisos y firmas

## ESTABILIDAD — fallos, deuda y bordes

- [ ] **27.** Los 3 tests rotos preexistentes (`schedule_store`×2, `control`) — arreglar o documentar por qué fallan
- [ ] **28.** `view.rs` y otros cambios sin commitear del otro agente — decidir qué se commitea
- [ ] **29.** Coordinación con el otro agente (rename Aria→Mimo): no pisar los mismos archivos
- [ ] **30.** Aviso cuando un schedule falle (el brief estuvo 3 días roto sin que nadie supiera; mirar `lastResult`)
- [ ] **31.** Verificar que el brief corra a las 7:00 de verdad ("next not again" en la salida del scheduler)
- [ ] **32.** `logs/aria.log` sin rotación — crecimiento indefinido
- [ ] **33.** `activity.jsonl` del puente del navegador crece sin límite
- [ ] **34.** `sessions.json`, `jobs.json`, `schedules.json`: crecimiento, consistencia y limpieza de viejos
- [ ] **35.** `attachments/` acumulándose — política de retención
- [ ] **36.** Worktrees de subagentes (`.aria-worktrees/`) acumulados
- [ ] **37.** `briefs/` con err.logs viejos
- [ ] **38.** Whisper: liberar el modelo cuando no dicta (RAM)
- [ ] **39.** El dock "se reinicia al quedar oculto y sin nada corriendo" — ¿puede quedarse colgado?
- [ ] **40.** `aria-host.js`: carrera de `extConnected`, chunks incompletos, cierre limpio
- [ ] **41.** Timeout del host (20s) vs capturas lentas — margen suficiente
- [ ] **42.** Keepalive cada 20s: ¿despierta si Chrome mata el service worker?
- [ ] **43.** `aria-browse`: los eventos broadcast ya colaron una respuesta una vez — auditar el resto del CLI
- [ ] **44.** `bridge.json` al caerse el host — limpieza y re-escritura
- [ ] **45.** Terminal: sin truecolor, terminales pequeños, pegados enormes
- [ ] **46.** Sesiones JSONL de Pi creciendo (`PI_SESSION_FILE`) — retención
- [ ] **47.** `~/.claude/projects` (logs de Claude Code) acumulándose
- [ ] **48.** Pi pinned 1.3.0 y agent-browser 0.38.2 — plan de actualizaciones de seguridad
- [ ] **49.** Integridad/backup de la base de Engram (`~/.aria/engram`)
- [ ] **50.** `worktree.rs`, `watch.rs`: procesos hijos huérfanos si el dock muere

## EFICIENCIA — costo, cuota y recursos

- [ ] **51.** Cuota Go al 87% semanal / 93% mensual — cerca del tope; plan de consumo
- [ ] **52.** Compaction al 90% — se compacta tarde (más tokens por turno); valorar bajarlo
- [ ] **53.** Diario usa `opencode-go/deepseek-v4.1-flash` ✓ — auditar que Zen (de pago) no se cuele
- [ ] **54.** `jev-1.13-free` agotado y `jev-1.13` sin fondos — clasificador de respaldo o recargar
- [ ] **55.** `jev-router` consulta el clasificador en cada mensaje — latencia y cuota; ¿cachear?
- [ ] **56.** Tavily dio 429 (rate limit) — cachear resultados o espaciar llamadas
- [ ] **57.** `npx -y rea-agents` descarga en cada corrida — cachear
- [ ] **58.** El análisis de REA produjo 206MB en `/tmp` — correr con salida acotada
- [ ] **59.** El dock mantiene Pi vivo oculto — medir RAM constante
- [ ] **60.** `context` mide el contexto antes de cada corrida — costo vs valor
- [ ] **61.** Capturas del navegador en `shots/` sin límite de tamaño
- [ ] **62.** Builds: release vs debug; `target/` de Rust ocupando GBs
- [ ] **63.** `~/lsv` pesa 160GB — revisar qué hay ahí
- [ ] **64.** `node_modules` y `.venv` dispersos por los proyectos
- [ ] **65.** Tests de cargo (53s): ver si los lentos son solo los preexistentes

## MANTENIBILIDAD — código, tests y docs

- [ ] **66.** `.md` del schema de WooCommerce desactualizado vs la página de Notion
- [ ] **67.** `aria-extension` sin tests automatizados — decidir los mínimos
- [ ] **68.** `ARIA-SETUP.md`: verificar cada punto tras los cambios del picker/brief
- [ ] **69.** `CHANGELOG.md`/`RELEASES.md`: reflejar los fixes (/models, picker, brief)
- [ ] **70.** `docs/pending.md` e `ideas.md`: deuda registrada sin cerrar
- [ ] **71.** `docs/parity.md` y las 15 specs vs implementación real
- [ ] **72.** i18n es/en completo (`ui/i18n`, `texts.rs`)
- [ ] **73.** `bun test` de cada extensión Pi en verde hoy
- [ ] **74.** Evals: `bun eval.ts` (jev-router) y los del sandbox, periódicos
- [ ] **75.** Cobertura de tests de `aria-core` (vault, paths, auth) tras la migración al llavero
- [ ] **76.** Nombres/interfaces preparados para el rename Aria→Mimo (pendiente, lo hará otro agente)

## OPERACIÓN — continuidad

- [ ] **77.** Backup real de `~/.aria` (sesiones, memoria, settings, schedules) — ¿existe?
- [ ] **78.** Recuperación si se pierde la máquina (re-login, ARIA-SETUP §D)
- [ ] **79.** El Mac despierto a las 7:00 para el brief — `pmset` schedule
- [ ] **80.** Auto-arranque del dock al iniciar sesión — verificar
- [ ] **81.** Proceso de update de Aria (`release.sh`) — documentarlo (probado el 2026-10-25)
- [ ] **82.** Actualizar la extensión Chrome desde el repo cuando cambie (hoy es manual)
- [ ] **83.** Plan de lanzamiento público (los 50 sitios guardados) — cuándo y checklist de release
- [ ] **84.** Estado global de un vistazo: cómo ver "todo en verde" fácilmente

## PRIVACIDAD — datos que guardamos

- [ ] **85.** `activity.jsonl` = historial de navegación completo — retención o borrado periódico
- [ ] **86.** `shots/` = capturas de pantalla del navegador — retención
- [ ] **87.** Memoria de Engram con datos personales — qué retiene y cómo borrarlo
- [ ] **88.** Trazas `/debug` y sesiones JSONL con conversaciones completas — quién las lee
- [ ] **89.** `attachments/` con archivos privados
- [ ] **90.** Briefs de Notion copiando datos de trabajo a `~/.aria/briefs/` — sensible
