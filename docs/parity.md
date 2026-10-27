# Parity matrix

Phase 2's rule: Aria works at least as well as the maintainer's Pi. This page walks `ARIA-SETUP.md` (the inventory of
that Pi) item by item against Aria, with the check that proves each one and its last result. An item with several
parts has one row per part. `bun test tools/parity` fails when an item has no row, or when a row names a file or a
test that no longer exists. How to rerun the checks is in `tools/parity/README.md`.

The checks prove the parts. Parity itself is accepted by the maintainer using Aria for real work, not by this page.

Check kinds:
- `test`: an offline test, named as `file::test`. Shell tests are named by their message.
- `eval`: a real-model eval.
- `script`: a script to run.
- `manual`: a short step by hand.
- `difference`: a deliberate difference, with who decided it.
- `out-of-scope`: left out of phase 2 by the maintainer.
- `n/a`: does not apply to Aria, with the reason.

| Item | What | Check | Where | Result |
|---|---|---|---|---|
| 1 | Pi, pinned in `pi-version` | test | `crates/aria-core/src/pi.rs::the_pinned_version_comes_from_the_repo_file` | pass 2026-10-19 |
| 1 | The installer installs the pinned Pi | test | `install/install.test.sh::install dry run installs the pinned Pi` | pass 2026-10-19 |
| 2 | bubblewrap and iproute2 | test | `install/install.test.sh::bubblewrap iproute2` | pass 2026-10-19 |
| 2 | bun | n/a | Only the checks run on bun; Aria and its Pi do not need it. | n/a |
| 2 | Node via mise for `rea` and `pi-acp` | n/a | `rea` and OpenPencil are out of scope; Pi's own Node is checked (`install/install.test.sh`, Node 22.19 or newer). | n/a |
| 3 | Engram, pinned in `engram-version` | test | `install/install.test.sh::a verified Engram goes into ~/.aria/bin` | pass 2026-10-19 |
| 4 | Claude Code, for the Claude provider | test | `install/install.test.sh::the installer links Claude Code into ~/.aria/bin` | pass 2026-10-19 |
| 4 | No Claude Code, no Claude provider | test | `crates/aria/src/handlers.rs::without_claude_code_nothing_is_downloaded` | pass 2026-10-19 |
| 5 | Pi packages pinned from the repo | test | `crates/aria-core/src/agent.rs::the_required_packages_come_from_the_repo_file_and_are_pinned` | pass 2026-10-19 |
| 5 | superpowers filtered to `systematic-debugging` | test | `crates/aria-core/src/agent.rs::prepare_gives_the_superpowers_package_the_owners_filter` | pass 2026-10-23 |
| 5 | `ponytail`, pinned, with its two skills | test | `crates/aria-core/src/agent.rs::prepare_gives_the_ponytail_package_its_two_skills` | pass 2026-10-23 |
| 5 | `pi-claude-agent-sdk`, pinned in `agent.rs` (`CLAUDE_BRIDGE_PACKAGE`), the `claude-bridge` provider | test | `crates/aria-core/src/agent.rs::the_install_command_runs_the_pinned_pi_in_aria_s_agent_dir_with_the_shims_first` | pass 2026-10-19 |
| 5 | The local `meeting` package | out-of-scope | The maintainer's local package, left out of phase 2. | out of scope |
| 6 | OpenPencil | out-of-scope | The maintainer's integration, left out of phase 2. | out of scope |
| 7 | Daily model `opencode-go/mimo-v2.6-flash` | test | `crates/aria-core/src/providers.rs::apply_opencode_go` | pass 2026-10-21 |
| 7 | The daily model is Pi's default | test | `crates/aria-core/src/agent.rs::sets_default_provider_and_model_from_the_daily_model` | pass 2026-10-21 |
| 7 | Never `opencode/*` unless allowed | test | `ui/chat/parts/models.test.js::are hidden unless allowed` | pass 2026-10-21 |
| 8 | Classifier `opencode/jev-1.13-free`, exact id | test | `extensions/jev-router/route.test.ts::the classifier is the free Jev` | pass 2026-10-19 |
| 9 | `/plan` on `mimo-v2.6-pro` at `high` | test | `extensions/jev-router/modes.test.ts::without a config the plan model is mimo-v2.6-pro at high, switching at 90%` | pass 2026-10-21 |
| 10 | Free fallback `cat-jam-free` from 98% | test | `extensions/quota-fallback/fallback.test.ts::without a config: cat-jam-free at 98%, five minutes` | pass 2026-10-19 |
| 10 | Always asking first | test | `extensions/lib/config.test.ts::without aria.json every value is today` | pass 2026-10-19 |
| 10 | Only OpenCode Go offers fallback models | test | `crates/aria-core/src/agent_config.rs::a_provider_without_a_free_tier_has_no_quota_section` | pass 2026-10-19 |
| 11 | Claude Max through `claude-bridge` | test | `crates/aria-core/src/agent.rs::the_claude_provider_is_installed_when_settings_list_it_and_its_files_exist` | pass 2026-10-19 |
| 11 | It answers on the Max plan | manual | With a `claude-bridge` model picked in the dock, `hola` answers and the Claude quota bars move. | manual — maintainer |
| 12 | Compaction 40% by default | test | `extensions/autocompact/autocompact.test.ts::percent defaults to 40 and round-trips` | pass 2026-10-19 |
| 12 | The maintainer's 90% comes from the settings | test | `extensions/autocompact/autocompact.test.ts::the percent comes from the config` | pass 2026-10-19 |
| 13 | Effort auto by default | test | `extensions/lib/config.test.ts::without aria.json every value is today` | pass 2026-10-19 |
| 14 | `trust.json` trusts the project | n/a | Pi asks for project trust only in interactive mode (read in Pi's own code, `dist/core/trust-manager.js`, when the row was written, 2026-10-19; not in this repository); the dock runs Pi in RPC mode. | n/a |
| 15 | Persona, word for word with the maintainer's settings | test | `crates/aria-core/src/persona.rs::the_owners_settings_render_his_system_md_word_for_word` | pass 2026-10-19 |
| 15 | The date comes in a hidden `[now]` line, not from `date` | difference | The maintainer asked for it (2026-10-19): the `clock` extension sends the date and time with each message, and the persona (and its golden copy) says to use it. | difference |
| 16 | Engram with `--project` and 7 direct tools | test | `crates/aria-core/src/mcp.rs::memory_on_writes_arias_engram_entry_with_the_project_and_its_own_database` | pass 2026-10-19 |
| 16 | The maintainer's 53 memories | difference | Aria's Engram has its own database in `~/.aria/engram` (the maintainer's decision, 2026-10-19), so the memories in `~/.engram` are not in Aria's memory. | difference |
| 17 | Tavily, with its key in `mcp.json` (mode 600; see the notes below) | test | `crates/aria-core/src/mcp.rs::tavily_is_written_in_pis_mcp_format_with_mode_600` | pass 2026-10-19 |
| 17 | Notion, OAuth, without a secret in `mcp.json` | test | `crates/aria-core/src/mcp.rs::notion_is_listed_without_any_secret` | pass 2026-10-19 |
| 17 | Notion and Tavily answer | manual | Connect both in the setup's services step, then ask the dock for a Notion task and a web search. | manual — maintainer |
| 17 | rea, openpencil, figma | out-of-scope | The maintainer's integrations, left out of phase 2. | out of scope |
| 18 | The router's routes and the hidden hint | eval | `extensions/jev-router/eval.ts` | 93/94 2026-10-19 (one borderline expert case, see [pending](pending.md)) |
| 18 | The route line the dock shows | test | `extensions/jev-router/route.test.ts::the flags become what Aria does` | pass 2026-10-19 |
| 18 | `pkill`, `killall` and `fuser -k` blocked | test | `extensions/jev-router/process-guard.test.ts::fuser -k -n tcp 5174` | pass 2026-10-19 |
| 18 | Expert assignment | eval | `extensions/jev-router/eval.ts` | 93/94 2026-10-19 (one borderline expert case, see [pending](pending.md)) |
| 19 | Auto effort follows the route; free models at their top | test | `extensions/jev-router/effort.test.ts::a free model runs at its top level` | pass 2026-10-19 |
| 19 | A level picked by hand stays until `/effort auto` | test | `extensions/jev-router/effort.test.ts::a level fixes it in the settings; auto goes back` | pass 2026-10-19 |
| 20 | `/plan` writes only `docs/plans`, `.gitignore` and `.git` of repos under the projects folder | test | `extensions/jev-router/modes.test.ts::source code is refused` | pass 2026-10-23 |
| 20 | `/execute` requires a plan with phases | test | `extensions/jev-router/slice.test.ts::a file without phases is refused` | pass 2026-10-23 |
| 21 | The experts' tools: two writers, three read-only | test | `crates/aria-core/src/persona.rs::a_read_only_expert_gets_the_read_only_tools_line_and_a_writer_loses_it` | pass 2026-10-19 |
| 21 | The maintainer's experts, word for word | test | `crates/aria-core/src/agent.rs::writes_the_owners_persona_and_experts` | pass 2026-10-19 |
| 22 | No turn limit by default | test | `extensions/subagent-mode/limit.test.ts::no limit by default` | pass 2026-10-19 |
| 22 | No saved sessions by default | test | `crates/aria-core/src/pi.rs::the_shim_keeps_no_session_by_default` | pass 2026-10-19 |
| 22 | `SUBAGENT.md`, memory read-only, `## Learnings` | manual | Ask the dock for a parallel task; each subagent's answer ends with `## Learnings` and no memory is saved by it. | manual — maintainer |
| 22 | Watching subagents | test | `crates/aria/src/watch.rs::a_child_that_connects_is_an_active_agent_and_the_page_is_told` | pass 2026-10-19 |
| 22 | `/agents` | difference | The dock opens the subagents panel with Alt+A; `/agents` (Pi's own overlay, which exists only in Pi's terminal UI, `extensions/subagent-watch/index.ts`) does nothing there ([pending](pending.md)). | difference |
| 23 | `ask_user`: 1-4 questions, recommended option first | test | `extensions/ask-user/questions.test.ts::the recommended option goes first, marked` | pass 2026-10-19 |
| 23 | An answer lifts read-only mode for the turn | manual | In a read-only turn, after the model asks with `ask_user` and the answer arrives, its next write goes through (`extensions/jev-router/index.ts`, the `ask_user` handler). | manual — maintainer |
| 24 | Writes only to the project, `/tmp` and caches | test | `extensions/sandbox/policy.test.ts::project, /tmp and existing caches` | pass 2026-10-19 |
| 24 | Credentials hidden, Aria's included | test | `extensions/sandbox/policy.test.ts::aria\'s credentials too` | pass 2026-10-19 |
| 24 | Credentials hidden in the real box, links included | test | `extensions/sandbox/policy.test.ts::neither the link nor its target can be read` | pass 2026-10-19 |
| 24 | Leaving asks, with Jev's review | test | `extensions/sandbox/route.test.ts::outside after asking, when the model asks for it` | pass 2026-10-19 |
| 24 | Jev's review | eval | `extensions/sandbox/review-eval.ts` | 10/10 2026-10-19 |
| 24 | Session grants by program and subcommand | test | `extensions/sandbox/grants.test.ts::program and subcommand` | pass 2026-10-19 |
| 24 | The sandbox with a real model | script | `extensions/sandbox/accept.sh` | 7/7 2026-10-19 |
| 25 | One quota question, shared by the chat, `/plan` and subagents | test | `extensions/quota-fallback/fallback.test.ts::the decision is shared through a file, for its window only` | pass 2026-10-19 |
| 26 | `/autocompact <0-100>` | test | `extensions/autocompact/autocompact.test.ts::writes compaction.percent in Aria` | pass 2026-10-19 |
| 27 | Code and prompts in English, the conversation in the user's language | test | `crates/aria-core/src/persona.rs::the_name_user_name_language_and_style_change_only_their_lines` | pass 2026-10-19 |
| 28 | `PI_SANDBOX=0`, `PI_QUOTA_FALLBACK=0`, `PI_SUBAGENT_LEAN=0`, `CLAUDE_BRIDGE_DEBUG=1` | manual | The extensions read them unchanged (`extensions/*/index.ts`); under Aria they reach Pi only through the dock's environment (on Linux `systemctl --user set-environment`, then restart the dock; no macOS step is written down). | manual — maintainer |
| 29 | The OpenCode Go key and the Anthropic sign-in in `auth.json` (see the notes below) | test | `crates/aria-core/src/auth.rs::set_writes_pi_format_with_0600` | pass 2026-10-19 |
| 29 | Never `ANTHROPIC_*` variables | test | `crates/aria-core/src/pi.rs::anthropic_variables_are_unset_for_pi` | pass 2026-10-19 |
| 30 | The Tavily key stays out of Aria's own config | test | `crates/aria-core/src/agent_config.rs::the_config_never_holds_a_secret` | pass 2026-10-19 |
| 31 | Notion's OAuth in `mcp-auth.json` | test | `crates/aria-core/src/mcp.rs::an_oauth_record_is_stored_under_pis_key_and_other_servers_stay` | pass 2026-10-19 |
| 32 | The `engram` path is Aria's own | test | `crates/aria-core/src/mcp.rs::memory_on_writes_arias_engram_entry_with_the_project_and_its_own_database` | pass 2026-10-19 |
| 32 | The `rea` path | out-of-scope | `rea` is left out of phase 2. | out of scope |
| 33 | The projects folder is the working folder | test | `crates/aria-core/src/pi.rs::session_env_lists_what_pi_needs` | pass 2026-10-19 |
| 33 | The extensions read it | test | `extensions/lib/agent-dir.test.ts::projectsDir uses ARIA_PROJECTS_DIR` | pass 2026-10-19 |
| 34 | The eval's hardcoded project paths | n/a | The ported eval names a neutral project built from the projects folder (`extensions/jev-router/eval.ts`). | n/a |
| 35 | Every test suite | script | `tools/parity/suites.sh` | pass 2026-10-19 |
| 36 | The router eval | eval | `extensions/jev-router/eval.ts` | 93/94 2026-10-19 (one borderline expert case, see [pending](pending.md)) |
| 36 | The sandbox review eval | eval | `extensions/sandbox/review-eval.ts` | 10/10 2026-10-19 |
| 36 | The sandbox acceptance | script | `extensions/sandbox/accept.sh` | 7/7 2026-10-19 |
| 37 | `pi -p "hola"` answers short and in Spanish | script | `tools/parity/smoke.sh` | pass 2026-10-19 |
| 37 | With a `claude-bridge` model it answers on the Max plan | manual | The smoke test with `ARIA_SMOKE_MODEL` set to a `claude-bridge` model, once the provider is connected. | manual — maintainer |

Notes:
- Rows 17, 29 and 31 check the plain-file format (`auth.json`, `mcp.json`, `mcp-auth.json` at mode 600). With
  `credentials.keyring` on, which is the default, the keys are sealed in the system keyring and the service works from a
  live copy outside `~/.aria`; see [safety](guide/safety.md) and [data and paths](reference/data-and-paths.md).
- The sandbox rows (24, 36) run on Linux with `bwrap`. macOS has no sandbox: every command asks first.
- The results are dated and were not rerun for this page. `tools/parity/suites.sh` has grown since row 35 passed: it also
  builds and checks the headless service and the hub's acceptance on Linux, and runs the release tests.
