//! Night Agent — a minimal overnight autonomous agent harness for local models.
//! Uses an OpenAI-compatible chat completion endpoint.

use anyhow::{anyhow, Result};
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;

// ------------------------- Data Structures -------------------------

#[derive(Clone, Serialize, Deserialize)]
struct Message {
    role: String, // "system", "user", "assistant"
    content: String,
}

#[derive(Debug, Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
enum Event {
    Message(Message),
    Compaction { summary: String },
}

// ------------------------- Model Client -------------------------

#[derive(Clone)]
struct Model {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f32,
}

impl Model {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": false
            }))
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;

        if !status.is_success() {
            return Err(anyhow!("HTTP {status}: {body}"));
        }

        let val: serde_json::Value = serde_json::from_str(&body)?;
        let content = val["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    async fn chat_with_retry(&self, messages: &[Message]) -> String {
        let mut backoff = 1;
        loop {
            match self.chat(messages).await {
                Ok(r) => return r,
                Err(e) => {
                    eprintln!("model error: {e}; retrying in {backoff}s");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(60);
                }
            }
        }
    }
}

// ------------------------- Tool Call Extraction -------------------------

fn extract_tool_calls(text: &str) -> Result<Vec<ToolCall>, String> {
    let re = Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap();
    let mut calls = vec![];

    for cap in re.captures_iter(text) {
        let raw = cap[1].trim();

        #[derive(Deserialize)]
        struct RawToolCall {
            name: String,
            arguments: serde_json::Value,
            id: Option<String>,
        }

        match serde_json::from_str::<RawToolCall>(raw) {
            Ok(r) => calls.push(ToolCall {
                id: r.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
                name: r.name,
                arguments: r.arguments,
            }),
            Err(e) => {
                return Err(format!(
                    "malformed tool call JSON: {e}\nRaw: {raw}"
                ))
            }
        }
    }

    Ok(calls)
}

// ------------------------- Tool Execution -------------------------

async fn execute_tool(call: &ToolCall, workdir: &Path) -> Result<String> {
    match call.name.as_str() {
        "read_file" => {
            let p = call.arguments["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
            let full = safe_path(workdir, p)?;
            Ok(tokio::fs::read_to_string(&full).await?)
        }
        "write_file" => {
            let p = call.arguments["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
            let content = call.arguments["content"].as_str().ok_or_else(|| anyhow!("missing content"))?;
            let full = safe_path(workdir, p)?;
            if let Some(parent) = full.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&full, content).await?;
            Ok(format!("wrote {}", full.display()))
        }
        "list_dir" => {
            let p = call.arguments["path"].as_str().unwrap_or(".");
            let full = safe_path(workdir, p)?;
            let mut out = String::new();
            let mut entries = tokio::fs::read_dir(&full).await?;
            while let Some(entry) = entries.next_entry().await? {
                out.push_str(&format!("{}\n", entry.file_name().to_string_lossy()));
            }
            Ok(truncate(&out, 4000))
        }
        "run_command" => {
            let cmd = call.arguments["command"].as_str().ok_or_else(|| anyhow!("missing command"))?;
            let output = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(workdir)
                .output()
                .await?;

            let mut result = String::new();
            if !output.stdout.is_empty() {
                result.push_str(&format!("stdout:\n{}", truncate(&String::from_utf8_lossy(&output.stdout), 3000)));
            }
            if !output.stderr.is_empty() {
                result.push_str(&format!("stderr:\n{}", truncate(&String::from_utf8_lossy(&output.stderr), 3000)));
            }
            result.push_str(&format!("\nexit code: {}", output.status.code().unwrap_or(-1)));
            Ok(result)
        }
        "finish" => {
            let reason = call.arguments["reason"].as_str().unwrap_or("done");
            Ok(format!("finish: {reason}"))
        }
        other => Err(anyhow!("unknown tool {other}")),
    }
}

fn safe_path(workdir: &Path, p: &str) -> Result<PathBuf> {
    let path = Path::new(p);
    if path.is_absolute() {
        return Err(anyhow!("absolute paths not allowed"));
    }
    let full = workdir.join(path);
    let canonical_workdir = workdir.canonicalize()?;
    let canonical_full = full.canonicalize().unwrap_or_else(|_| full.clone());
    if !canonical_full.starts_with(&canonical_workdir) {
        return Err(anyhow!("path escapes workdir"));
    }
    Ok(full)
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut out = s.to_string();
    if out.chars().count() > max_chars {
        out = out.chars().take(max_chars).collect();
        out.push_str("\n...[truncated]");
    }
    out
}

// ------------------------- Session Management -------------------------

struct Session {
    id: String,
    goal: String,
    scratchpad: String,
    messages: Vec<Message>, // recent messages, excluding system
    transcript_path: PathBuf,
    workdir: PathBuf,
    context_tokens: usize,
}

impl Session {
    async fn append_event(&self, event: &Event) -> Result<()> {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.transcript_path)
            .await?;
        let line = serde_json::to_string(event)?;
        file.write_all(format!("{line}\n").as_bytes()).await?;
        Ok(())
    }

    async fn append_message(&self, m: &Message) -> Result<()> {
        self.append_event(&Event::Message(m.clone())).await
    }

    async fn append_compaction(&self, summary: &str) -> Result<()> {
        self.append_event(&Event::Compaction {
            summary: summary.to_string(),
        })
        .await
    }
}

fn session_dir(id: &str) -> PathBuf {
    let base = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".night_agent")
        .join("sessions")
        .join(id);
    base
}

async fn create_session(
    id: &str,
    goal: &str,
    workdir: PathBuf,
    context_tokens: usize,
) -> Result<Session> {
    let dir = session_dir(id);
    tokio::fs::create_dir_all(&dir).await?;
    tokio::fs::write(dir.join("goal.txt"), goal).await?;
    let transcript_path = dir.join("transcript.jsonl");
    // Create empty transcript if not exists
    if !transcript_path.exists() {
        tokio::fs::write(&transcript_path, "").await?;
    }
    Ok(Session {
        id: id.to_string(),
        goal: goal.to_string(),
        scratchpad: String::new(),
        messages: vec![],
        transcript_path,
        workdir,
        context_tokens,
    })
}

async fn load_session(
    id: &str,
    workdir: PathBuf,
    context_tokens: usize,
) -> Result<Session> {
    let dir = session_dir(id);
    let transcript_path = dir.join("transcript.jsonl");

    if !dir.exists() {
        return Err(anyhow!("session {id} does not exist"));
    }

    let goal = tokio::fs::read_to_string(dir.join("goal.txt")).await?;
    let mut scratchpad = String::new();
    let mut messages = vec![];

    let data = tokio::fs::read_to_string(&transcript_path).await?;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(line)?;
        match event {
            Event::Message(m) => messages.push(m),
            Event::Compaction { summary } => scratchpad = summary,
        }
    }

    Ok(Session {
        id: id.to_string(),
        goal,
        scratchpad,
        messages,
        transcript_path,
        workdir,
        context_tokens,
    })
}

// ------------------------- Context Management -------------------------

fn estimate_tokens(messages: &[Message], scratchpad: &str) -> usize {
    let mut chars = scratchpad.len();
    for m in messages {
        chars += m.content.len();
    }
    chars / 4 // rough estimate
}

fn system_prompt(goal: &str, scratchpad: &str) -> String {
    format!(
        r#"You are an autonomous agent. Goal: {goal}

Current scratchpad:
{scratchpad}

Available tools:
- read_file(path)
- write_file(path, content)
- list_dir(path)
- run_command(command)
- finish(reason)

Always output tool calls as:
<tool_call>
{{"id":"...","name":"tool_name","arguments":{{...}}}}
</tool_call>

Rules:
- Never ask for permission.
- Never stop until the goal is verified.
- If a tool fails, read the error and try a different approach.
- You are operating unsupervised. Do not ask questions.
"#
    )
}

async fn maybe_compact(session: &mut Session, model: &Model) -> Result<()> {
    let threshold = session.context_tokens * 70 / 100;
    if estimate_tokens(&session.messages, &session.scratchpad) < threshold {
        return Ok(());
    }

    let keep = 12;
    let split = session.messages.len().saturating_sub(keep);
    if split == 0 {
        return Ok(());
    }

    let old: Vec<_> = session.messages.drain(..split).collect();
    let old_text = old
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let summary_messages = vec![
        Message {
            role: "system".into(),
            content: "You are a summarizer. Preserve goal, completed work, important findings, current plan, and next actions. Be concise.".into(),
        },
        Message {
            role: "user".into(),
            content: format!("Summarize:\n{old_text}"),
        },
    ];

    let summary = model.chat(&summary_messages).await?;
    session.scratchpad = summary.clone();
    session.append_compaction(&summary).await?;

    Ok(())
}

// ------------------------- Main Agent Loop -------------------------

async fn run_agent(
    config: &Config,
    model: &Model,
    session: &mut Session,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(config.max_wall_secs);
    let mut iterations = 0;
    let mut malformed_streak = 0;

    println!(
        "Starting agent with goal: {}\nSession: {}",
        session.goal, session.id
    );

    while iterations < config.max_iterations && Instant::now() < deadline {
        iterations += 1;

        maybe_compact(session, model).await?;

        let mut messages = vec![Message {
            role: "system".into(),
            content: system_prompt(&session.goal, &session.scratchpad),
        }];
        messages.extend(session.messages.iter().cloned());

        let response = tokio::time::timeout(
            Duration::from_secs(120),
            model.chat_with_retry(&messages),
        )
        .await;

        let response = match response {
            Ok(r) => r,
            Err(_) => {
                let msg = Message {
                    role: "user".into(),
                    content: "Model call timed out. Continue where you left off.".into(),
                };
                session.messages.push(msg.clone());
                session.append_message(&msg).await?;
                continue;
            }
        };

        let assistant_msg = Message {
            role: "assistant".into(),
            content: response.clone(),
        };
        session.messages.push(assistant_msg.clone());
        session.append_message(&assistant_msg).await?;

        match extract_tool_calls(&response) {
            Ok(calls) if !calls.is_empty() => {
                malformed_streak = 0;

                for call in calls {
                    if call.name == "finish" {
                        let reason = call.arguments["reason"].as_str().unwrap_or("done");
                        println!("Agent finished: {reason}");
                        println!("Iterations: {iterations}");
                        return Ok(());
                    }

                    println!("Executing tool: {}", call.name);
                    let result = execute_tool(&call, &session.workdir)
                        .await
                        .unwrap_or_else(|e| format!("ERROR: {e}"));

                    println!("Result: {}", truncate(&result, 200));
                    let tool_msg = Message {
                        role: "user".into(),
                        content: format!(
                            "Tool result for {}: {}",
                            call.name,
                            truncate(&result, 4000)
                        ),
                    };

                    session.messages.push(tool_msg.clone());
                    session.append_message(&tool_msg).await?;
                }
            }
            Ok(_) => {
                // Plain text, no tool call. Nudge.
                malformed_streak += 1;
                let nudge = if malformed_streak > 3 {
                    "You appear to be stuck. Re-read the goal, update your plan, and take a concrete action using a tool call."
                } else {
                    "Continue working autonomously. Output a tool call next."
                };
                println!("No tool call, nudging: {nudge}");
                let msg = Message {
                    role: "user".into(),
                    content: nudge.into(),
                };
                session.messages.push(msg.clone());
                session.append_message(&msg).await?;
            }
            Err(err) => {
                malformed_streak += 1;
                let correction = format!(
                    "Your previous tool call was malformed: {err}\nPlease output a valid tool call inside <tool_call> tags."
                );
                println!("Malformed tool call: {err}");
                let msg = Message {
                    role: "user".into(),
                    content: correction,
                };
                session.messages.push(msg.clone());
                session.append_message(&msg).await?;
            }
        }
    }

    if iterations >= config.max_iterations {
        println!("Hit max iterations");
    } else if Instant::now() >= deadline {
        println!("Hit wall-clock limit");
    }

    Ok(())
}

// ------------------------- CLI & Main -------------------------

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// Base URL of the OpenAI-compatible endpoint (without trailing /v1, e.g. http://localhost:8080/v1)
    #[clap(long)]
    base_url: String,

    /// Model name
    #[clap(long)]
    model: String,

    /// Goal description
    #[clap(long)]
    goal: String,

    /// Working directory (sandbox root)
    #[clap(long)]
    workdir: PathBuf,

    /// Session ID (optional, will be generated if not provided)
    #[clap(long)]
    session: Option<String>,

    /// Context token budget
    #[clap(long, default_value_t = 8192)]
    context_tokens: usize,

    /// Max iterations
    #[clap(long, default_value_t = 200)]
    max_iterations: usize,

    /// Max wall-clock seconds
    #[clap(long, default_value_t = 28800)]
    max_wall_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();

    // Create model client
    let client = reqwest::Client::new();
    let model = Model {
        client,
        base_url: config.base_url.clone(),
        model: config.model.clone(),
        temperature: 0.7,
    };

    // Determine session ID
    let session_id = match config.session.clone() {
        Some(id) => id,
        None => Uuid::new_v4().to_string(),
    };

    // Load or create session
    let mut session = if let Ok(s) = load_session(&session_id, config.workdir.clone(), config.context_tokens).await {
        println!("Resuming session {session_id}");
        s
    } else {
        println!("Creating new session {session_id}");
        create_session(&session_id, &config.goal, config.workdir.clone(), config.context_tokens).await?
    };

    // Ensure the workdir exists
    tokio::fs::create_dir_all(&session.workdir).await?;

    // Run agent
    run_agent(&config, &model, &mut session).await?;

    Ok(())
}