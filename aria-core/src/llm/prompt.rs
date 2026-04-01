use crate::agent::definition::AgentDefinition;
use crate::whiteboard::schema::ContextSnapshot;
use crate::tools::tool_trait::Tool;
use std::sync::Arc;

pub fn build_system_prompt(
    definition: &AgentDefinition,
    context: &ContextSnapshot,
    tools: &[Arc<dyn Tool>]
) -> String {
    let mut prompt = String::new();

    // Soul y Rol
    prompt.push_str(&format!("{}\n\n", definition.soul));
    prompt.push_str(&format!("## Tu Rol\n{}\n\n", definition.role));

    // Reglas de Trabajo
    prompt.push_str("## Reglas de Trabajo\n");
    for rule in &definition.workflow_rules {
        prompt.push_str(&format!("- {}\n", rule));
    }
    prompt.push_str("\n");

    // Contexto de la Tarea (YAML)
    let context_yaml = serde_yaml::to_string(context).unwrap_or_default();
    prompt.push_str(&format!("## Contexto de la Tarea\n```yaml\n{}```\n\n", context_yaml));

    // Herramientas Disponibles
    prompt.push_str("## Herramientas Disponibles\n");
    for tool in tools {
        prompt.push_str(&format!("- **{}**: {}\n", tool.name(), tool.description()));
    }
    prompt.push_str("\n");

    // Instrucciones de Comunicación
    prompt.push_str("## Instrucciones de Comunicación\n");
    prompt.push_str("- Actualiza tu tarea en el tablero después de cada acción significativa.\n");
    prompt.push_str("- Si encuentras un bloqueador, márcalo antes de detener el trabajo.\n");
    prompt.push_str("- Tu result_summary final debe ser autónomo: quien lo lea sin contexto debe entenderlo.\n");

    prompt
}
