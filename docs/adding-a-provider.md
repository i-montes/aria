# Adding a provider

A provider is one `ProviderModule` in `crates/aria-core/src/providers.rs`. It declares what the provider
supports; nothing else in Aria hard-codes a provider.

1. **Declare it.** Add an entry to `PROVIDERS` (and bump the array length): `id`, `label`, `auth`
   (`ApiKey` or `OAuth`), `auth_entries` (the keys of `auth.json` its credential is stored under, named
   as Pi names them), its `models`, `free_models` (empty when it has no free tier), `quota`
   (`Some(QuotaDefaults)` only when the quota can be read), `includes_jev`, and the daily and planning
   defaults.
2. **Validate its key.** Add a `KeyKind` and its probe in `crates/aria-core/src/validate.rs`: one cheap
   request that tells a valid key from a rejected one.
3. **Show it.** Add the provider to `PROVIDERS` in `ui/onboarding/steps.js` (same id, models and defaults)
   and its strings to `ui/i18n/en.json` and `ui/i18n/es.json`.
4. **Test it.** `apply_provider` should produce the settings you expect; add a test next to the existing ones.

Only providers with a free tier get fallback models: leave `quota` as `None` for the rest and Aria shows
and warns about usage without switching models.
