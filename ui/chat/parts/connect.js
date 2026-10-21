// The "connect a provider" card (`/connect`): choose the provider, how to sign in, enter a key or finish the browser
// sign-in, then pick a model. It lives in the thread while it is open. Not in the design (the design has no command for
// this); it is drawn like the design's other cards.
//
// Props: { provider: 'anthropic' | 'opencode-go' | null, step, error?, url?, revealed?, keyDraft? }
//   step   pick | anthropic | key | waiting | installing | done
//   error  a code the dock answered (see TEXTS `chat.connect.error.*`)
//   url    the sign-in link, so the browser can be opened again
// ui: unused. t: translate(key, vars).
//
// Emits: data-chat="connect" with data-op:
//   provider (data-value: the provider id)   method (data-value: oauth | key)   back   save   reveal   open-url   models   close
// and the key field `data-chat-field="connect-key"` (its text is read on `input`).
import { esc } from '../../onboarding/components.js'

export const PROVIDERS = [
  { id: 'anthropic', label: 'Anthropic' },
  { id: 'opencode-go', label: 'OpenCode Go' },
]

const CODES = ['key-rejected', 'network-unavailable', 'auth-file-broken', 'keyring-failed', 'oauth-denied', 'oauth-timeout', 'oauth-state', 'oauth-failed', 'pi-missing', 'claude-code-missing', 'claude-bridge-install-failed', 'unknown-provider']

/** The text key for an error code (a generic one for a code nobody knows), or null when there is no error. */
export const errorKey = (code) => (code ? `chat.connect.error.${CODES.includes(code) ? code : 'generic'}` : null)

export const TEXTS = {
  en: {
    'chat.connect.title': 'Connect a provider',
    'chat.connect.pick': 'Which one do you want to connect?',
    'chat.connect.anthropic.desc': 'Claude: with your Pro or Max plan, or an API key.',
    'chat.connect.opencode-go.desc': 'Open models with a free tier. Needs an API key.',
    'chat.connect.method.title': 'How do you want to connect Anthropic?',
    'chat.connect.method.oauth': 'Sign in with Claude',
    'chat.connect.method.oauth.desc': 'Uses your Claude Pro or Max plan. Needs Claude Code installed, and downloads the Claude provider once (about 300 MB).',
    'chat.connect.method.key': 'Use an API key',
    'chat.connect.method.key.desc': 'Pay per token. Nothing else to install.',
    'chat.connect.key.title': 'Paste your {provider} API key',
    'chat.connect.key.placeholder': 'API key',
    'chat.connect.key.save': 'Save key',
    'chat.connect.key.show': 'Show',
    'chat.connect.key.hide': 'Hide',
    'chat.connect.waiting': 'Finish signing in in your browser. Aria continues by itself.',
    'chat.connect.openAgain': 'Open the link again',
    'chat.connect.installing': 'Installing the Claude provider… this happens once and can take a minute.',
    'chat.connect.done': '{provider} is connected.',
    'chat.connect.models': 'Choose a model',
    'chat.connect.back': 'Back',
    'chat.connect.close': 'Close',
    'chat.connect.error.generic': 'Something went wrong. Try again.',
    'chat.connect.error.key-rejected': 'That key was not accepted. Check it and try again.',
    'chat.connect.error.network-unavailable': 'Could not reach the provider to check the key. Check your connection.',
    'chat.connect.error.auth-file-broken': 'Aria could not write its credentials file (~/.aria/auth.json). Check its permissions.',
    'chat.connect.error.keyring-failed': 'Aria could not keep the key in the system keyring. Unlock it and try again (or run aria keyring off).',
    'chat.connect.error.oauth-denied': 'The sign-in was denied.',
    'chat.connect.error.oauth-timeout': 'The sign-in took too long. Try again.',
    'chat.connect.error.oauth-state': 'The sign-in did not match this request. Try again.',
    'chat.connect.error.oauth-failed': 'The sign-in failed. Try again.',
    'chat.connect.error.pi-missing': 'The engine (Pi) is not installed. Run the Aria installer again.',
    'chat.connect.error.claude-code-missing': 'Claude Code is not installed. Install it, run the Aria installer again, and retry.',
    'chat.connect.error.claude-bridge-install-failed': 'The Claude provider could not be installed. Check your connection and retry.',
    'chat.connect.error.unknown-provider': 'Aria does not know that provider.',
  },
  es: {
    'chat.connect.title': 'Conectar un proveedor',
    'chat.connect.pick': '¿Cuál quieres conectar?',
    'chat.connect.anthropic.desc': 'Claude: con tu plan Pro o Max, o con una API key.',
    'chat.connect.opencode-go.desc': 'Modelos abiertos con capa gratuita. Necesita una API key.',
    'chat.connect.method.title': '¿Cómo quieres conectar Anthropic?',
    'chat.connect.method.oauth': 'Iniciar sesión con Claude',
    'chat.connect.method.oauth.desc': 'Usa tu plan Claude Pro o Max. Necesita Claude Code instalado y descarga el proveedor de Claude una vez (unos 300 MB).',
    'chat.connect.method.key': 'Usar una API key',
    'chat.connect.method.key.desc': 'Se paga por token. No hay nada más que instalar.',
    'chat.connect.key.title': 'Pega tu API key de {provider}',
    'chat.connect.key.placeholder': 'API key',
    'chat.connect.key.save': 'Guardar clave',
    'chat.connect.key.show': 'Mostrar',
    'chat.connect.key.hide': 'Ocultar',
    'chat.connect.waiting': 'Termina de iniciar sesión en el navegador. Aria sigue sola.',
    'chat.connect.openAgain': 'Abrir el enlace otra vez',
    'chat.connect.installing': 'Instalando el proveedor de Claude… pasa una sola vez y puede tardar un minuto.',
    'chat.connect.done': '{provider} está conectado.',
    'chat.connect.models': 'Elegir un modelo',
    'chat.connect.back': 'Atrás',
    'chat.connect.close': 'Cerrar',
    'chat.connect.error.generic': 'Algo salió mal. Inténtalo otra vez.',
    'chat.connect.error.key-rejected': 'Esa clave no fue aceptada. Revísala e inténtalo otra vez.',
    'chat.connect.error.network-unavailable': 'No se pudo contactar al proveedor para comprobar la clave. Revisa tu conexión.',
    'chat.connect.error.auth-file-broken': 'Aria no pudo escribir su archivo de credenciales (~/.aria/auth.json). Revisa sus permisos.',
    'chat.connect.error.keyring-failed': 'Aria no pudo guardar la llave en el llavero del sistema. Desbloquéalo y vuelve a intentar (o ejecuta aria keyring off).',
    'chat.connect.error.oauth-denied': 'Se denegó el inicio de sesión.',
    'chat.connect.error.oauth-timeout': 'El inicio de sesión tardó demasiado. Inténtalo otra vez.',
    'chat.connect.error.oauth-state': 'El inicio de sesión no coincidió con esta solicitud. Inténtalo otra vez.',
    'chat.connect.error.oauth-failed': 'El inicio de sesión falló. Inténtalo otra vez.',
    'chat.connect.error.pi-missing': 'El motor (Pi) no está instalado. Vuelve a ejecutar el instalador de Aria.',
    'chat.connect.error.claude-code-missing': 'Claude Code no está instalado. Instálalo, vuelve a ejecutar el instalador de Aria y reintenta.',
    'chat.connect.error.claude-bridge-install-failed': 'No se pudo instalar el proveedor de Claude. Revisa tu conexión y reintenta.',
    'chat.connect.error.unknown-provider': 'Aria no conoce ese proveedor.',
  },
}

const labelOf = (id) => PROVIDERS.find((p) => p.id === id)?.label ?? String(id ?? '')
const op = (name, value) => `data-chat="connect" data-op="${name}"${value === undefined ? '' : ` data-value="${esc(value)}"`}`

const choice = (name, value, title, description) =>
  `<button type="button" class="p-connect-opt" ${op(name, value)}><span class="p-connect-opt-title">${esc(title)}</span><span class="p-connect-opt-desc">${esc(description)}</span></button>`

export function renderConnect({ provider, step = 'pick', error, url, revealed, keyDraft } = {}, ui, t) {
  const label = labelOf(provider)
  const failure = errorKey(error)
  const close = `<button type="button" class="p-connect-x" ${op('close')} aria-label="${esc(t('chat.connect.close'))}" title="${esc(t('chat.connect.close'))}">×</button>`
  const back = `<button type="button" class="p-connect-link" ${op('back')}>${esc(t('chat.connect.back'))}</button>`
  let body = ''
  let title = t('chat.connect.title')
  if (step === 'pick') {
    body = `<p class="p-connect-text">${esc(t('chat.connect.pick'))}</p><div class="p-connect-opts">${PROVIDERS.map((p) => choice('provider', p.id, p.label, t(`chat.connect.${p.id}.desc`))).join('')}</div>`
  } else if (step === 'anthropic') {
    title = label
    body = `<p class="p-connect-text">${esc(t('chat.connect.method.title'))}</p><div class="p-connect-opts">${choice('method', 'oauth', t('chat.connect.method.oauth'), t('chat.connect.method.oauth.desc'))}${choice('method', 'key', t('chat.connect.method.key'), t('chat.connect.method.key.desc'))}</div><div class="p-connect-actions">${back}</div>`
  } else if (step === 'key') {
    title = label
    body = `<p class="p-connect-text">${esc(t('chat.connect.key.title', { provider: label }))}</p>
      <div class="p-connect-field">
        <input type="${revealed ? 'text' : 'password'}" class="p-connect-input" data-chat-field="connect-key" value="${esc(keyDraft ?? '')}" placeholder="${esc(t('chat.connect.key.placeholder'))}" spellcheck="false" autocomplete="off">
        <button type="button" class="p-connect-link" ${op('reveal')}>${esc(t(revealed ? 'chat.connect.key.hide' : 'chat.connect.key.show'))}</button>
      </div>
      <div class="p-connect-actions">${provider === 'anthropic' ? back : ''}<button type="button" class="p-connect-primary" ${op('save')}>${esc(t('chat.connect.key.save'))}</button></div>`
  } else if (step === 'waiting') {
    title = label
    body = `<p class="p-connect-text" role="status">${esc(t('chat.connect.waiting'))}</p>${url ? `<div class="p-connect-actions"><button type="button" class="p-connect-link" ${op('open-url')} data-url="${esc(url)}">${esc(t('chat.connect.openAgain'))}</button></div>` : ''}`
  } else if (step === 'installing') {
    title = label
    body = `<p class="p-connect-text" role="status">${esc(t('chat.connect.installing'))}</p>`
  } else if (step === 'done') {
    title = label
    body = `<p class="p-connect-text" role="status">${esc(t('chat.connect.done', { provider: label }))}</p><div class="p-connect-actions"><button type="button" class="p-connect-primary" ${op('models')}>${esc(t('chat.connect.models'))}</button></div>`
  }
  return `<div class="p-connect" role="group" aria-label="${esc(t('chat.connect.title'))}">
    <div class="p-connect-head"><span class="p-connect-title">${esc(title)}</span>${close}</div>
    ${body}
    ${failure ? `<p class="p-connect-error" role="alert">${esc(t(failure))}</p>` : ''}
  </div>`
}
