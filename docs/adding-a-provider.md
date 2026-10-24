# Adding a provider

A provider is one `ProviderModule` in `crates/aria-core/src/providers.rs`. It declares what the provider supports and
the settings Aria applies when you choose it (`apply_provider`). Three providers exist today: `opencode-go`,
`anthropic-api` and `anthropic-oauth`. The settings and the setup read the module, but a few other places still name
the providers, so a new one touches more than one file. This is the whole list; [development](development.md) covers
building and testing.

1. **Declare it.** Add an entry to `PROVIDERS` in `providers.rs` (and bump the array length): `id`, `label`, `auth`
   (`ApiKey` or `OAuth`), `auth_entries` (the keys of `auth.json` its credential is stored under, named as Pi names
   them), its `models` and `free_models` (empty when it has no free tier), `quota` (`Some(QuotaDefaults)` only when
   the free-model fallback applies), `includes_jev` (its own account reaches Jev, so no separate Jev key is needed),
   and the daily and planning defaults. `apply_provider` turns the router on, with effort `auto` and Jev's review,
   when `includes_jev` is true, and off, with effort fixed at `medium`, when it is false.
2. **Validate its key.** Add a `KeyKind` and its probe in `crates/aria-core/src/validate.rs`: one cheap request that
   tells a valid key from a rejected one. Then map the provider id to that kind and to its `auth.json` entries in
   `key_target` in `crates/aria/src/handlers.rs`; an id it does not know answers `unknown-provider`.
3. **Show it in the setup and settings.** Add the provider to `PROVIDERS` in `ui/onboarding/steps.js` (same id, models
   and defaults; `applyProvider` there is the page's twin of the Rust function) and its credential name to `CREDENTIAL`
   in the same file. Add Pi's provider id to `PI_PROVIDER` in `ui/onboarding/view.js` (it picks the model list), and its
   strings (`ob.provider.<id>.desc`) to `ui/i18n/en.json` and `ui/i18n/es.json`.
4. **Show it in the chat.** The model picker names providers and their credentials in `ui/chat/parts/models.js`
   (`PROVIDERS`, `CREDENTIAL`), the terminal's picker does the same in `crates/aria/src/term/picker.rs`, and the
   `/connect` card lists a family in `ui/chat/parts/connect.js` and `KEY_PROVIDER` in `ui/chat/view.js`.
5. **Read its quota, if it has one.** The bars come from `source_for` and `request_for` in
   `crates/aria-core/src/quota.rs`, which match on Pi's provider name; without a match the provider shows no bars.
6. **Test it.** `apply_provider` should produce the settings you expect: add a test next to the existing ones in
   `providers.rs`. The tests also compare the settings after each provider with the files in
   `crates/aria-core/tests/golden/`; `cargo test -p aria-core regenerate_golden -- --ignored` rewrites them.

The free-model fallback is tied to OpenCode Go: the `quota-fallback` extension acts only on `opencode-go` models that
are not free and reads the Go key (`extensions/quota-fallback/fallback.ts`, `extensions/jev-router/usage.ts`). Leave
`quota` as `None` for any other provider; to give a new one a fallback, those extensions need to learn it too. With
`quota` as `None` Aria never switches models for that provider.
