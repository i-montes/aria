//! The owner's persona (`SYSTEM.md`) and experts (`agents/*.md`) as Aria writes them for its Pi: his text word for word,
//! with only the parts a setting varies filled in (spec: "The owner's prose becomes templates").
//!
//! The template is `agent-files/SYSTEM.md`: `{{name}}` placeholders, and blocks that stay or go between
//! `<!-- if X -->` and `<!-- end X -->` lines (the marker lines themselves never reach the file).

use crate::settings::{Expert, Settings};

/// The "Keep answers…" line for each answer style of the setup (`assistant.style`); `very-short` is the owner's own.
fn style_line(style: &str) -> &'static str {
    match style {
        "brief" => "- Keep answers brief: the answer first, in a short paragraph, with a short list when it helps. No tables or headings unless the user asks. Don't repeat the question, recap your steps or close with a summary or an offer.",
        "detailed" => "- Answer in detail: the answer first, then the explanation that makes it clear, with short sections or lists when they help. Don't pad it or repeat the question.",
        "technical" => "- Be technical and precise: the answer first, with the exact code, commands and file paths that decide it. No filler and no recap at the end.",
        "teaching" => "- Teach step by step: the answer first, then how it works, one step at a time with a small example. Explain any term the user may not know. No recap at the end.",
        _ => "- Keep answers very short and direct: the answer first, in 2-4 sentences and under about 80 words. No tables, no headings and no long lists (a short list only for a few separate items, one line each). Give only what decides the question, not every detail you know. Don't repeat the question, recap your steps or close with a summary, a \"practical takeaway\" or an offer. Go longer only when the user asks for detail.",
    }
}

/// The **Git** bullet from the four git settings; the defaults give the owner's own text.
fn git_line(settings: &Settings) -> String {
    let git = &settings.git;
    let mut line = String::from("- **Git:** working on your own branch is fine.");
    if git.ask_before_others_changes {
        line += " If at the start there are uncommitted changes you didn't make in this conversation, ask before committing, stashing (`stash`) or discarding them.";
    }
    if git.commit_without_asking {
        line += " Commit your work at the end of each task without asking.";
    }
    if git.push_or_merge_without_asking {
        line += " You may `push` and `merge` without asking; never delete branches without asking.";
    } else {
        line += " Never `push`, `merge` or delete branches without asking.";
    }
    line
}

/// "`a` and `b`" (or "`a`, `b` and `c`").
fn code_list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.len() {
        0 => String::new(),
        1 => quoted[0].clone(),
        n => format!("{} and {}", quoted[..n - 1].join(", "), quoted[n - 1]),
    }
}

/// Keeps or drops each `<!-- if X -->` … `<!-- end X -->` block; the marker lines never reach the file.
fn blocks(template: &str, on: &dyn Fn(&str) -> bool) -> String {
    let mut out = String::with_capacity(template.len());
    let mut dropping: Option<String> = None;
    for line in template.split_inclusive('\n') {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("<!-- if ").and_then(|r| r.strip_suffix(" -->")) {
            if dropping.is_none() && !on(name) {
                dropping = Some(name.to_string());
            }
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("<!-- end ").and_then(|r| r.strip_suffix(" -->")) {
            if dropping.as_deref() == Some(name) {
                dropping = None;
            }
            continue;
        }
        if dropping.is_none() {
            out.push_str(line);
        }
    }
    out
}

/// The conversation language as the persona names it: the setup's codes by name, anything else (typed by the user,
/// such as "French") as it was written.
fn language_name(language: &str) -> String {
    match language.trim() {
        "es" => "Spanish".into(),
        "en" => "English".into(),
        "pt" => "Portuguese".into(),
        "" => "Spanish".into(),
        other => other.to_string(),
    }
}

pub fn render_system(template: &str, settings: &Settings) -> String {
    let assistant = &settings.assistant;
    let name = if assistant.name.trim().is_empty() { "Aria" } else { assistant.name.trim() };
    let user = assistant.user_name.trim();
    let kept = blocks(template, &|block| match block {
        "user" => !user.is_empty(),
        "proposals" => settings.git.unasked_as_proposals,
        "memory" => settings.memory.enabled,
        "plan" => settings.subagents.plan_execute,
        _ => true,
    });
    let experts: Vec<String> =
        settings.subagents.experts.iter().filter(|e| e.enabled).map(|e| e.name.clone()).collect();
    let skills = &settings.subagents.auto_skills;
    let auto_skills = match skills.len() {
        0 => String::new(),
        1 => format!(", except {}, which does apply whenever relevant", code_list(skills)),
        _ => format!(", except {}, which do apply whenever relevant", code_list(skills)),
    };
    let project = if settings.memory.project.trim().is_empty() { "aria" } else { settings.memory.project.trim() };
    kept.replace("{{name}}", name)
        .replace("{{user_name}}", user)
        .replace("{{language}}", &language_name(&assistant.language))
        .replace("{{style}}", style_line(&assistant.style))
        .replace("{{git}}", &git_line(settings))
        .replace("{{memory_project}}", project)
        .replace("{{experts}}", &if experts.is_empty() { "none enabled".to_string() } else { experts.join(", ") })
        .replace("{{auto_skills}}", &auto_skills)
        .replace(
            "{{memory_learnings}}",
            if settings.memory.enabled {
                " Save to memory (with `mcp__engram__mem_save`) the \"Learnings\" worth keeping, with the same rules as your own memories."
            } else {
                ""
            },
        )
}

/// The read-only experts' tool list (the owner's front matter for security, reviewer and researcher).
const READ_ONLY_TOOLS: &str = "tools: read, bash, codemode";

/// An expert file with its front matter's `tools` line following `read_only`; the rest is the owner's text.
pub fn render_expert(source: &str, expert: &Expert) -> String {
    let Some(rest) = source.strip_prefix("---\n") else { return source.to_string() };
    let Some(end) = rest.find("\n---\n") else { return source.to_string() };
    let front: Vec<&str> = rest[..end].lines().filter(|l| !l.starts_with("tools:")).collect();
    let mut front = front.join("\n");
    if expert.read_only {
        front += &format!("\n{READ_ONLY_TOOLS}");
    }
    format!("---\n{front}{}", &rest[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = include_str!("../../../agent-files/SYSTEM.md");
    const OWNER: &str = include_str!("../tests/golden/SYSTEM.owner.md");

    fn owners() -> Settings {
        let mut settings: Settings =
            serde_json::from_str(include_str!("../tests/golden/settings-provider-opencode-go.json")).unwrap();
        settings.assistant.language = "es".into();
        settings
    }

    #[test]
    fn the_owners_settings_render_his_system_md_word_for_word() {
        assert_eq!(render_system(TEMPLATE, &owners()), OWNER);
    }

    #[test]
    fn the_name_user_name_language_and_style_change_only_their_lines() {
        let mut settings = owners();
        settings.assistant.name = "Nova".into();
        settings.assistant.user_name = "Iván".into();
        settings.assistant.language = "en".into();
        settings.assistant.style = "detailed".into();
        let text = render_system(TEMPLATE, &settings);
        assert!(text.starts_with("# Nova\n\nYou are Nova, a personal assistant"));
        assert!(text.contains("The user's name is Iván."));
        assert!(text.contains("in the user's language: English unless they write to you in another one"));
        assert!(!text.contains("under about 80 words"), "the very-short line is replaced");
        let changed: Vec<_> = OWNER.lines().filter(|l| !text.lines().any(|t| t == *l)).collect();
        assert!(changed.len() <= 4, "only the varying lines change: {changed:#?}");
    }

    #[test]
    fn every_style_has_its_own_line() {
        let mut lines = std::collections::HashSet::new();
        for style in ["very-short", "brief", "detailed", "technical", "teaching"] {
            let mut settings = owners();
            settings.assistant.style = style.into();
            let text = render_system(TEMPLATE, &settings);
            lines.insert(
                text.lines()
                    .find(|l| {
                        l.starts_with("- Keep")
                            || l.starts_with("- Answer")
                            || l.starts_with("- Be ")
                            || l.starts_with("- Teach")
                    })
                    .unwrap()
                    .to_string(),
            );
        }
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn memory_off_drops_the_memory_section_and_the_project_is_the_setting() {
        let mut settings = owners();
        settings.memory.project = "casa".into();
        let text = render_system(TEMPLATE, &settings);
        assert!(text.contains("project: \"casa\"") && !text.contains("project: \"aria\""));
        settings.memory.enabled = false;
        let text = render_system(TEMPLATE, &settings);
        assert!(!text.contains("# Memory (Engram)") && !text.contains("mcp__engram__mem_save"));
    }

    #[test]
    fn the_git_rules_follow_the_settings() {
        let mut settings = owners();
        settings.git.commit_without_asking = true;
        settings.git.push_or_merge_without_asking = true;
        settings.git.ask_before_others_changes = false;
        settings.git.unasked_as_proposals = false;
        let text = render_system(TEMPLATE, &settings);
        assert!(!text.contains("Never `push`, `merge`"));
        assert!(text.contains("Commit your work"));
        assert!(!text.contains("ask before committing, stashing"));
        assert!(!text.contains("- **Scope:**"));
    }

    #[test]
    fn only_enabled_experts_are_listed() {
        let mut settings = owners();
        settings.subagents.experts.retain(|e| e.name != "security");
        settings.subagents.experts.iter_mut().for_each(|e| e.enabled = e.name != "backend");
        let text = render_system(TEMPLATE, &settings);
        assert!(
            text.contains("There are experts (frontend, reviewer, researcher)"),
            "{}",
            text.lines().find(|l| l.contains("There are experts")).unwrap_or("")
        );
    }

    #[test]
    fn plan_off_drops_slice_and_the_skills_come_from_the_settings() {
        let mut settings = owners();
        assert!(render_system(TEMPLATE, &settings).contains("# Slice"));
        settings.subagents.auto_skills = vec!["brainstorming".into()];
        let text = render_system(TEMPLATE, &settings);
        assert!(text.contains("except `brainstorming`, which does apply whenever relevant"));
        settings.subagents.plan_execute = false;
        let text = render_system(TEMPLATE, &settings);
        assert!(!text.contains("# Slice") && !text.contains("/plan <idea>") && !text.contains("`plan=offer`"));
    }

    #[test]
    fn every_language_is_named_as_chosen() {
        for (code, name) in [("es", "Spanish"), ("en", "English"), ("pt", "Portuguese"), ("Français", "Français")] {
            let mut settings = owners();
            settings.assistant.language = code.into();
            let text = render_system(TEMPLATE, &settings);
            assert!(text.contains(&format!("in the user's language: {name} unless")), "{code}");
        }
    }

    #[test]
    fn no_placeholder_or_marker_survives() {
        for language in ["es", "en"] {
            let mut settings = owners();
            settings.assistant.language = language.into();
            settings.memory.enabled = false;
            settings.subagents.plan_execute = false;
            let text = render_system(TEMPLATE, &settings);
            assert!(!text.contains("{{") && !text.contains("<!--"), "{text}");
        }
    }

    #[test]
    fn a_read_only_expert_gets_the_read_only_tools_line_and_a_writer_loses_it() {
        let frontend = include_str!("../../../agent-files/agents/frontend.md");
        let security = include_str!("../../../agent-files/agents/security.md");
        let ro = |name: &str, read_only: bool| Expert { name: name.into(), enabled: true, read_only };
        assert_eq!(render_expert(frontend, &ro("frontend", false)), frontend);
        assert_eq!(render_expert(security, &ro("security", true)), security);
        assert!(render_expert(frontend, &ro("frontend", true)).contains("\ntools: read, bash, codemode\n---"));
        assert!(!render_expert(security, &ro("security", false)).contains("tools:"));
    }
}
