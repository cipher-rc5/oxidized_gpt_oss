// file: src/tokenizer.rs
// description: Loads tokenizer.json and applies a minimal Harmony chat template and response parsing helpers.
// author: cipher-rc5
// created: 2026-02-21
// modified: 2026-02-21

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", path.display()))
}

pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

pub fn decode(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    tok.decode(ids, false)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
}

pub fn apply_harmony_chat_template(
    system: Option<&str>,
    user: &str,
    reasoning_effort: ReasoningEffort,
) -> String {
    let sys = system.unwrap_or("You are ChatGPT, a large language model trained by OpenAI.");
    format!(
        "<|start|>system<|message|>{sys}\n\nReasoning: {}<|end|><|start|>user<|message|>{user}<|end|><|start|>assistant",
        reasoning_effort.as_str()
    )
}

pub fn strip_harmony_reasoning(text: &str, show_thinking: bool) -> String {
    if show_thinking {
        return text.to_string();
    }
    let mut out = text.to_string();
    for token in [
        "<|channel|>analysis",
        "<|channel|>commentary",
        "<|channel|>final",
        "<|return|>",
        "<|call|>",
        "<|end|>",
        "<|message|>",
    ] {
        out = out.replace(token, "");
    }
    out
}

pub fn harmony_stop_token_ids(tok: &Tokenizer) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for token in ["<|return|>", "<|call|>", "<|end|>"] {
        if let Some(id) = tok.token_to_id(token) {
            out.push(id);
        }
    }
    anyhow::ensure!(!out.is_empty(), "failed to resolve harmony stop tokens");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_contains_harmony_markers() {
        let t = apply_harmony_chat_template(None, "hello", ReasoningEffort::Low);
        assert!(t.contains("<|start|>system<|message|>"));
        assert!(t.contains("<|start|>user<|message|>hello<|end|>"));
        assert!(t.contains("<|start|>assistant"));
    }
}
