# Auditoría Aria — checklist completa

Todo lo que hay que revisar del proyecto Aria (dock, terminal, extensiones, puente del navegador, datos, hub, máquinas, instalador y releases).
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
- [ ] **26.** Binarios (`~/.local/bin/aria`, `~/.aria/bin/*`): permisos y firmas (ver también 108–109 de releases e instalador)

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

## HUB, MÁQUINAS E INSTALADOR — seguridad

- [ ] **91.** Llave del cliente `~/.ssh/aria-hub`: permisos 600, con passphrase y en `ssh-agent`; revocarla = quitar su línea de `authorized_keys` del hub (comprobar que la conexión abierta se corta al reiniciar el servicio)
- [ ] **92.** Línea de la llave del cliente en el hub: `restrict,command="…/aria client-stdio"` sin opciones de más (solo `from=`, `expiry-time=`, `verify-required`); revisar `authorized_keys` en busca de llaves de Aria sin restricción
- [ ] **93.** Una llave de cliente robada equivale a un shell como el usuario del hub (el cliente responde las preguntas del sandbox): decidir `sandbox.askToLeave=false` en el hub y revisar qué más se puede cerrar
- [ ] **94.** Usuario del hub: hoy es un usuario con `sudo` sin contraseña — crear un usuario dedicado sin privilegios para `aria serve` y volver a probar `hub init` con él
- [ ] **95.** Llave de la máquina `~/.ssh/aria-machine-<nombre>`: sin passphrase por diseño (el enlace arranca solo) — decidir si se acepta o se exige agente (`--machine-key`); permisos 600
- [ ] **96.** `aria hub-link machine <nombre>`: el nombre sale solo de la línea forzada de `authorized_keys`; probar con `SSH_ORIGINAL_COMMAND` manipulado, nombres inválidos y dos máquinas con el mismo nombre
- [ ] **97.** Lo que una máquina puede mandar al hub: listas de sesiones falsas, respuestas a clientes que no preguntaron, `machine`/`offline` falsificados — confirmar con una máquina maliciosa que el hub los descarta
- [ ] **98.** Topes del enlace de máquinas (línea 16 MiB, `link-sessions` 2 MiB, 500 sesiones, 32 clientes por enlace, 64 KiB de saludo, 10 s para identificarse): revisar que sean razonables y que una máquina que los pasa solo se cae a sí misma
- [ ] **99.** `hubUpdate` (el hub se actualiza solo): solo `X.Y.Z` exacta y mayor que la que corre, un repositorio fijo, hijo sin `sudo` y con límite de 15 min — revisar `ARIA_REPO` del servicio y que ninguna ruta del cliente llegue a un shell
- [ ] **100.** `unlink-machine` edita `authorized_keys` del hub: probar con líneas parecidas (mismo prefijo, otro nombre), con el archivo sin salto de línea final y comprobar el respaldo `authorized_keys.aria-backup` (permisos 600, borrarlo cuando no haga falta)
- [ ] **101.** `hub init` y `hub update` entran con el login de administrador de quien los ejecuta: confirmar que no guardan ni envían ese login a ninguna parte (solo `~/.ssh/config`, agente y `ARIA_SSH`)
- [ ] **102.** `hub init` envía credenciales al hub (`auth.json`, `mcp.json`, `mcp-auth.json`): revisar qué archivos salieron, que un hub que ya los tiene no los reemplaza sin `--credentials`, y que ningún valor aparece en logs ni en la pantalla
- [ ] **103.** Perfil de AppArmor `/etc/apparmor.d/aria-bwrap`: revisar que cubre solo `bwrap`, que el `sysctl` global sigue en 1 y cómo se quita (`apparmor_parser -R` y borrar el archivo)
- [ ] **104.** Puerto ssh público del hub (2222): cerrar cuando la red privada sea la única entrada (Tailscale) y confirmar `PasswordAuthentication no`; hasta entonces las llaves restringidas son la protección
- [ ] **105.** Credenciales en el hub: sin llavero (archivos 600 en `~/.aria`) — confirmar permisos, cifrado de disco del VPS y que los backups del VPS no las copian
- [ ] **106.** Un plan de Claude no funciona en un hub: decidir API key de Anthropic u OpenCode Go en el hub y comprobar que el aviso de `hub init` aparece

## INSTALADOR Y RELEASES — cadena de suministro

- [ ] **107.** `curl | sh` del instalador: confía solo en TLS y en GitHub; documentar la verificación manual (descargar, leer, ejecutar) y el checksum de cada binario
- [ ] **108.** Releases sin firma: el `.sha256` sale del mismo release que el binario — decidir firmas (minisign o cosign) o un checksum fijo en el repo
- [ ] **109.** Binarios publicados desde el portátil (`tools/release-local.sh`): antes de cada release comprobar que ningún binario contiene la ruta del home ni el usuario (el script lo exige), y decidir pasar a CI limpio cuando GitHub vuelva a ejecutar Actions
- [ ] **110.** GitHub Actions bloqueado por facturación (los jobs no arrancan en ningún repo de la cuenta): resolver el pago, y revisar de paso los repos privados con CI o despliegue que llevan semanas sin ejecutarse
- [ ] **111.** `.github/workflows/release.yml` no ha corrido nunca: probar `glslc`/`spirv-headers` en Ubuntu, el runner `ubuntu-24.04-arm` para el hub aarch64 y la compilación cruzada del Mac Intel
- [ ] **112.** El tag `v0.21.0` está en GitHub sin release (quedó atrás del 0.22.0): decidir borrarlo o publicarlo
- [ ] **113.** El release 0.22.0 no trae binarios de macOS: ejecutar `sh tools/release-local.sh vX.Y.Z` en el Mac (completa el release) y probar `curl | sh` en un Mac limpio
- [ ] **114.** El instalador de escritorio desde el release (`curl | sh`) no se ha probado de punta a punta; solo el de hub dentro de una actualización
- [ ] **115.** Instalador: `sudo` solo para paquetes, `loginctl enable-linger` y el perfil de AppArmor — verificar que no ejecuta nada más con privilegios y que `--yes` solo acepta lo que el plan anuncia
- [ ] **116.** Instalador: pantallas y preguntas probadas con una terminal simulada (archivos y stubs); probar a mano en terminal real (80 columnas, `NO_COLOR`, `TERM=dumb`, UTF-8 y no UTF-8), en macOS y con Ctrl+C en cada pregunta
- [ ] **117.** Instalar el dock actualiza los hubs enlazados (`aria hub update --all`): probar con dos hubs, uno caído y sin terminal; que un fallo no deje la instalación local a medias

## HUB Y MÁQUINAS — estabilidad y pruebas pendientes

- [ ] **118.** Auto-actualización del hub por la llave restringida (`hubUpdate`): no se ha ejecutado contra un hub real (el primero que la entienda es el 0.22.0); probar con un release más nuevo, y el reinicio del servicio a mitad de la actualización con systemd real (la conexión puede cortarse antes del resultado)
- [ ] **119.** Enlace de máquinas (`aria link`): probado solo con hubs y ssh falsos; probar con el VPS y el Omarchy reales, `BatchMode=yes` con una llave que pide agente, y la reconexión tras suspender y reanudar el portátil
- [ ] **120.** Dock ya abierto antes de `link.json`: tarda hasta 10 s en notarlo, y uno abierto con un binario viejo no enlaza hasta reiniciar — comprobar el aviso de `hub init` y el reinicio del dock
- [ ] **121.** El dock de macOS: el enlace compila (cfg) pero no se ha ejecutado; la ruta del runtime privado y el supervisor del hilo
- [ ] **122.** La lista de sesiones de la terminal solo conoce las máquinas por sus sesiones o por `machinesList`: probar un hub viejo que no contesta `machinesList` y una máquina que desaparece
- [ ] **123.** Un hub y un cliente de versiones distintas: la nota de versión en la terminal está probada en tests, no vista; probar hub viejo, hub nuevo y protocolo distinto
- [ ] **124.** `aria hub init`: panel de ratatui probado en tmux; probar en otras terminales, con redimensionado, passphrase real de `ssh-keygen`, `ssh -tt` con `sudo` real y un fallo a mitad (la terminal debe quedar limpia)
- [ ] **125.** `aria hub init --binary` desde un checkout puede subir una compilación vieja sin avisar (el plan no muestra su fecha): mostrar la fecha o comparar versiones
- [ ] **126.** Test inestable `pi::tests::with_saved_sessions_the_shim_gives_a_child_a_session_dir` (aparece cuando corre otra compilación a la vez, probable «text file busy»): reproducir y arreglar
- [ ] **127.** `cargo fmt --check` falla en `term/app.rs` y `term/picker.rs` (commits de otra máquina) y `clippy` deja 3 avisos preexistentes en `chat.rs` y `state.rs`: limpiar
- [ ] **128.** `aria connect` sigue siendo un segundo comando: el diseño de «un solo Aria» (`docs/superpowers/specs/2026-10-27-one-aria-design.md`) lo convierte en alias — piezas S3 (el cerebro desde el hub), S4 (un solo `aria`), S5 (adopción) y S6 (instalador) sin construir

## HUB — operación y privacidad

- [ ] **129.** VPS compartido con otros servicios: poner límites de memoria y CPU a `aria-serve.service` (`MemoryMax`, `CPUQuota`) y revisar que un Pi descontrolado no afecte al resto
- [ ] **130.** Logs del hub (`journalctl --user`, `~/.aria/logs`): rotación, tamaño y que no contengan secretos ni texto de conversaciones
- [ ] **131.** Copias de seguridad del hub: qué de `~/.aria` (sesiones, memoria de Engram, credenciales) se respalda, dónde y cifrado; probar una restauración
- [ ] **132.** Memoria dividida: el Engram del hub y el del computador son bases distintas hasta la pieza S3 — decidir qué se importa y cómo se verifica (conteo de entradas)
- [ ] **133.** Datos que viajan al hub: ajustes, credenciales y conectores se copian al VPS — documentar quién tiene acceso a ese servidor (otros usuarios, otros servicios) y qué país/proveedor lo aloja
- [ ] **134.** Servidor de prueba por la tailnet usado durante las pruebas (`python3 -m http.server` en la IP de Tailscale): comprobar que no queda ninguno corriendo
- [ ] **135.** Cambios hechos fuera del repositorio en el VPS y en esta máquina: mantener `CHANGELOG.md` al día (archivo, copia, cómo revertir) y revisar que cada fila siga siendo cierta
