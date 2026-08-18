//! Night Agent — an overnight autonomous agent harness with TUI,
//! scrollable transcript, editable todo panel, web search, interactive
//! control, context compaction, and session persistence.

use anyhow::{anyhow, Result};
use arboard::Clipboard;
use clap::Parser;
use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste,
        EnableMouseCapture, Event as CEvent, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

// ============================================================
// Data Structures
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
enum Event {
    Message(Message),
    Compaction { summary: String },
}

#[derive(Debug)]
enum UiEvent {
    Log(String),

    Status {
        iteration: usize,
        tokens: usize,
        context_tokens: usize,
        goal: String,
        todo: String,
        elapsed: Duration,
    },

    Reasoning {
        content: String,
    },

    AgentFinished {
        reason: String,
    },

    AgentError {
        error: String,
    },

    Quit,
}

#[derive(Debug)]
enum AgentCommand {
    Pause,
    Resume,
    UpdateGoal(String),
    AddInstruction(String),
    UpdateTodo(String),
    Quit,
}

// ============================================================
// Model Client
// ============================================================

#[derive(Clone)]
struct Model {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f32,
}

impl Model {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        let url = format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        );

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
        let mut backoff = 1u64;

        loop {
            match self.chat(messages).await {
                Ok(response) => return response,

                Err(e) => {
                    eprintln!("model error: {e}; retrying in {backoff}s");

                    tokio::time::sleep(Duration::from_secs(backoff)).await;

                    backoff = (backoff * 2).min(60);
                }
            }
        }
    }
}

// ============================================================
// Tool Call Extraction
// ============================================================

fn extract_tool_calls(text: &str) -> Result<Vec<ToolCall>, String> {
    let re = Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap();

    let mut calls = Vec::new();

    for cap in re.captures_iter(text) {
        let raw = cap[1].trim();

        #[derive(Deserialize)]
        struct RawToolCall {
            name: String,
            arguments: serde_json::Value,
        }

        match serde_json::from_str::<RawToolCall>(raw) {
            Ok(parsed) => {
                calls.push(ToolCall {
                    name: parsed.name,
                    arguments: parsed.arguments,
                });
            }

            Err(e) => {
                return Err(format!("malformed tool call JSON: {e}\nRaw: {raw}"));
            }
        }
    }

    Ok(calls)
}

// ============================================================
// Tool Execution
// ============================================================

async fn execute_tool(call: &ToolCall, workdir: &Path, todo_path: &Path) -> Result<String> {
    match call.name.as_str() {
        "read_file" => {
            let p = call.arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow!("missing path"))?;

            let full = safe_path(workdir, p)?;

            Ok(tokio::fs::read_to_string(&full).await?)
        }

        "write_file" => {
            let p = call.arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow!("missing path"))?;

            let content = call.arguments["content"]
                .as_str()
                .ok_or_else(|| anyhow!("missing content"))?;

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
            let cmd = call.arguments["command"]
                .as_str()
                .ok_or_else(|| anyhow!("missing command"))?;

            // NOTE:
            // This preserves your original behavior.
            // The command must eventually exit.
            //
            // On Windows, your current implementation uses `sh`,
            // so this expects a shell such as Git Bash to be available.
            let output = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(workdir)
                .output()
                .await?;

            let mut result = String::new();

            if !output.stdout.is_empty() {
                result.push_str(&format!(
                    "stdout:\n{}",
                    truncate(&String::from_utf8_lossy(&output.stdout), 3000)
                ));
            }

            if !output.stderr.is_empty() {
                result.push_str(&format!(
                    "stderr:\n{}",
                    truncate(&String::from_utf8_lossy(&output.stderr), 3000)
                ));
            }

            result.push_str(&format!(
                "\nexit code: {}",
                output.status.code().unwrap_or(-1)
            ));

            Ok(result)
        }

        "update_todo" => {
            let content = call.arguments["content"]
                .as_str()
                .ok_or_else(|| anyhow!("missing content"))?;

            tokio::fs::write(todo_path, content).await?;

            Ok(format!("todo list updated:\n{}", content))
        }

        "get_todo" => {
            match tokio::fs::read_to_string(todo_path).await {
                Ok(content) => Ok(content),
                Err(_) => Ok("No todo list yet.".to_string()),
            }
        }

        "search_web" => {
            let query = call.arguments["query"]
                .as_str()
                .ok_or_else(|| anyhow!("missing query"))?;

            search_web(query).await
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

fn truncate_display(s: &str, max_chars: usize) -> String {
    if s.chars().count() > max_chars {
        let mut t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        t.push('…');
        t
    } else {
        s.to_string()
    }
}

// ============================================================
// Web Search
// ============================================================

async fn search_web(query: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://html.duckduckgo.com/html/")?;

    url.query_pairs_mut().append_pair("q", query);

    let client = reqwest::Client::new();

    let resp = client
        .get(url)
        .header(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        .send()
        .await?;

    let body = resp.text().await?;

    let re = Regex::new(
        r#"<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>"#,
    )
    .unwrap();

    let mut results = String::new();
    let mut count = 0;

    for cap in re.captures_iter(&body) {
        if count >= 5 {
            break;
        }

        let url = cap[1].to_string();
        let title = strip_html(&cap[2]);
        let snippet = strip_html(&cap[3]);

        results.push_str(&format!(
            "{}. {}\nURL: {}\n{}\n\n",
            count + 1,
            title,
            url,
            snippet
        ));

        count += 1;
    }

    if results.is_empty() {
        return Ok("No search results found.".to_string());
    }

    Ok(truncate(&results, 4000))
}

fn strip_html(s: &str) -> String {
    let re = Regex::new(r"<[^>]*>").unwrap();
    re.replace_all(s, "").to_string()
}

// ============================================================
// Session Management
// ============================================================

struct Session {
    id: String,
    goal: String,
    scratchpad: String,
    messages: Vec<Message>,
    transcript_path: PathBuf,
    todo_path: PathBuf,
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

    async fn append_message(&self, message: &Message) -> Result<()> {
        self.append_event(&Event::Message(message.clone())).await
    }

    async fn append_compaction(&self, summary: &str) -> Result<()> {
        self.append_event(&Event::Compaction {
            summary: summary.to_string(),
        })
        .await
    }
}

fn session_dir(id: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".night_agent")
        .join("sessions")
        .join(id)
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

    if !transcript_path.exists() {
        tokio::fs::write(&transcript_path, "").await?;
    }

    let todo_path = dir.join("todo.md");

    if !todo_path.exists() {
        tokio::fs::write(&todo_path, "").await?;
    }

    Ok(Session {
        id: id.to_string(),
        goal: goal.to_string(),
        scratchpad: String::new(),
        messages: Vec::new(),
        transcript_path,
        todo_path,
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

    let todo_path = dir.join("todo.md");

    if !dir.exists() {
        return Err(anyhow!("session {id} does not exist"));
    }

    let goal = tokio::fs::read_to_string(dir.join("goal.txt")).await?;

    let mut scratchpad = String::new();
    let mut messages = Vec::new();

    let data = tokio::fs::read_to_string(&transcript_path).await?;

    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let event: Event = serde_json::from_str(line)?;

        match event {
            Event::Message(message) => {
                messages.push(message);
            }

            Event::Compaction { summary } => {
                scratchpad = summary;
            }
        }
    }

    Ok(Session {
        id: id.to_string(),
        goal,
        scratchpad,
        messages,
        transcript_path,
        todo_path,
        workdir,
        context_tokens,
    })
}

// ============================================================
// Context Management
// ============================================================

fn estimate_tokens(messages: &[Message], scratchpad: &str) -> usize {
    let mut chars = scratchpad.len();

    for message in messages {
        chars += message.content.len();
    }

    chars / 4
}

fn system_prompt(goal: &str, scratchpad: &str) -> String {
    format!(
        r#"You are an autonomous coding agent.

Goal:
{goal}

Current scratchpad:
{scratchpad}

Available tools:
- read_file(path)
- write_file(path, content)
- list_dir(path)
- run_command(command)
- update_todo(content)
- get_todo()
- search_web(query)
- finish(reason)

Always output tool calls exactly as:

<tool_call>
{{"name":"tool_name","arguments":{{...}}}}
</tool_call>

Rules:
- Never ask for permission.
- Never stop until the goal is verified.
- If a tool fails, read the error and try another approach.
- You are operating unsupervised.
- Maintain a todo list.
- Update the todo list whenever you start or finish meaningful work.
- Prefer actually testing changes instead of assuming they work.
"#,
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
            content:
                "You are a summarizer. Preserve the goal, completed work, important findings, current plan, and next actions. Be concise."
                    .into(),
        },
        Message {
            role: "user".into(),
            content: format!("Summarize:\n{}", old_text),
        },
    ];

    let summary = model.chat(&summary_messages).await?;

    session.scratchpad = summary.clone();

    session.append_compaction(&summary).await?;

    Ok(())
}

// ============================================================
// Auto Context Detection
// ============================================================

async fn detect_context_size(base_url: &str) -> Option<usize> {
    let client = reqwest::Client::new();

    let url = format!("{}/props", base_url.trim_end_matches('/'));

    let response = client.get(&url).send().await.ok()?;

    if !response.status().is_success() {
        return None;
    }

    let value: serde_json::Value = response.json().await.ok()?;

    value["default_generation_settings"]["n_ctx"]
        .as_u64()
        .map(|v| v as usize)
}

// ============================================================
// Auto-generate initial todo from goal
// ============================================================

async fn generate_initial_todo(model: &Model, goal: &str) -> Result<String> {
    let messages = vec![
        Message {
            role: "system".into(),
            content: "You are a planning assistant. Create a concise markdown todo list for the goal. Break it into small manageable tasks. Use '- [ ]' checkboxes. Do not include anything except the markdown list.".into(),
        },
        Message {
            role: "user".into(),
            content: format!("Goal: {goal}"),
        },
    ];

    Ok(model.chat(&messages).await?)
}

// ============================================================
// Agent Loop
// ============================================================

async fn run_agent(
    config: &Config,
    model: &Model,
    session: &mut Session,
    tx_ui: mpsc::UnboundedSender<UiEvent>,
    mut rx_cmd: mpsc::UnboundedReceiver<AgentCommand>,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(config.max_wall_secs);

    let mut iterations = 0usize;
    let mut malformed_streak = 0usize;
    let mut paused = false;
    let mut finished = false;

    let send = |event: UiEvent| {
        let _ = tx_ui.send(event);
    };

    send(UiEvent::Log(format!(
        "Starting agent with goal: {}",
        session.goal
    )));

    send(UiEvent::Log(format!("Session: {}", session.id)));

    send(UiEvent::Log(format!(
        "Context limit: {}",
        session.context_tokens
    )));

    // Auto-generate initial todo if none exists
    let existing_todo = tokio::fs::read_to_string(&session.todo_path)
        .await
        .unwrap_or_default();

    if existing_todo.trim().is_empty() {
        send(UiEvent::Log(
            "Generating initial todo list from goal...".into(),
        ));

        match generate_initial_todo(model, &session.goal).await {
            Ok(todo) => {
                tokio::fs::write(&session.todo_path, &todo).await?;
                send(UiEvent::Log(format!("Initial todo:\n{}", todo)));
            }
            Err(e) => {
                send(UiEvent::Log(format!(
                    "Failed to generate initial todo: {e}"
                )));
            }
        }
    }

    loop {
        while let Ok(command) = rx_cmd.try_recv() {
            match command {
                AgentCommand::Pause => {
                    paused = true;
                    send(UiEvent::Log("Paused by user.".into()));
                }

                AgentCommand::Resume => {
                    paused = false;
                    finished = false;
                    send(UiEvent::Log("Resumed by user.".into()));
                }

                AgentCommand::UpdateGoal(new_goal) => {
                    session.goal = new_goal.clone();
                    let dir = session_dir(&session.id);
                    if let Err(e) = tokio::fs::write(dir.join("goal.txt"), &new_goal).await {
                        send(UiEvent::Log(format!("Failed to save new goal: {e}")));
                    } else {
                        send(UiEvent::Log(format!("Goal updated to: {}", new_goal)));
                    }
                    finished = false;
                }

                AgentCommand::AddInstruction(instruction) => {
                    let message = Message {
                        role: "user".into(),
                        content: format!("New instruction from user: {}", instruction),
                    };
                    session.messages.push(message.clone());
                    session.append_message(&message).await?;
                    send(UiEvent::Log(format!("Instruction added: {}", instruction)));
                    finished = false;
                }

                AgentCommand::UpdateTodo(content) => {
                    if let Err(e) = tokio::fs::write(&session.todo_path, &content).await {
                        send(UiEvent::Log(format!("Failed to save todo: {e}")));
                    } else {
                        send(UiEvent::Log("Todo updated by user.".into()));
                    }
                }

                AgentCommand::Quit => {
                    send(UiEvent::Log("Quit command received, stopping agent.".into()));
                    send(UiEvent::Quit);
                    return Ok(());
                }
            }
        }

        if paused || finished {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        }

        if iterations >= config.max_iterations || Instant::now() >= deadline {
            if iterations >= config.max_iterations {
                send(UiEvent::Log("Hit max iterations.".into()));
            } else {
                send(UiEvent::Log("Hit wall-clock limit.".into()));
            }
            send(UiEvent::AgentFinished {
                reason: "limits reached".into(),
            });
            finished = true;
            continue;
        }

        iterations += 1;

        maybe_compact(session, model).await?;

        let mut messages = vec![Message {
            role: "system".into(),
            content: system_prompt(&session.goal, &session.scratchpad),
        }];

        let mut history = session.messages.clone();

        // Some APIs reject consecutive assistant messages.
        // If the transcript ends with assistant, add a synthetic user turn.
        if history
            .last()
            .map(|m| m.role == "assistant")
            .unwrap_or(false)
        {
            history.push(Message {
                role: "user".into(),
                content: "Continue working on the goal.".into(),
            });
        }

        messages.extend(history);

        let response = tokio::time::timeout(
            Duration::from_secs(120),
            model.chat_with_retry(&messages),
        )
        .await;

        let response = match response {
            Ok(response) => response,
            Err(_) => {
                let message = Message {
                    role: "user".into(),
                    content: "Model call timed out. Continue where you left off.".into(),
                };
                session.messages.push(message.clone());
                session.append_message(&message).await?;
                send(UiEvent::Log("Model timed out, nudging to continue.".into()));
                continue;
            }
        };

        send(UiEvent::Reasoning {
            content: response.clone(),
        });

        let assistant_message = Message {
            role: "assistant".into(),
            content: response.clone(),
        };
        session.messages.push(assistant_message.clone());
        session.append_message(&assistant_message).await?;

        match extract_tool_calls(&response) {
            Ok(calls) if !calls.is_empty() => {
                malformed_streak = 0;

                for call in calls {
                    if call.name == "finish" {
                        let reason = call.arguments["reason"].as_str().unwrap_or("done");
                        send(UiEvent::Log(format!("Agent finished: {reason}")));
                        send(UiEvent::Log(format!("Iterations: {iterations}")));
                        send(UiEvent::AgentFinished {
                            reason: reason.to_string(),
                        });
                        finished = true;
                        break;
                    }

                    send(UiEvent::Log(format!("Executing tool: {}", call.name)));
                    let result = execute_tool(&call, &session.workdir, &session.todo_path)
                        .await
                        .unwrap_or_else(|e| format!("ERROR: {e}"));

                    send(UiEvent::Log(format!("Result: {}", truncate(&result, 200))));

                    let tool_message = Message {
                        role: "user".into(),
                        content: format!(
                            "Tool result for {}: {}",
                            call.name,
                            truncate(&result, 4000)
                        ),
                    };
                    session.messages.push(tool_message.clone());
                    session.append_message(&tool_message).await?;
                }
            }
            Ok(_) => {
                malformed_streak += 1;
                let nudge = if malformed_streak > 3 {
                    "You appear to be stuck. Re-read the goal, update your plan, and take a concrete action using a tool call."
                } else {
                    "Continue working autonomously. Output a tool call next."
                };
                send(UiEvent::Log(format!("No tool call, nudging: {nudge}")));
                let message = Message {
                    role: "user".into(),
                    content: nudge.into(),
                };
                session.messages.push(message.clone());
                session.append_message(&message).await?;
            }
            Err(error) => {
                malformed_streak += 1;
                let correction = format!(
                    "Your previous tool call was malformed: {error}\nPlease output a valid tool call inside <tool_call> tags."
                );
                send(UiEvent::Log(format!("Malformed tool call: {error}")));
                let message = Message {
                    role: "user".into(),
                    content: correction,
                };
                session.messages.push(message.clone());
                session.append_message(&message).await?;
            }
        }

        let todo = tokio::fs::read_to_string(&session.todo_path)
            .await
            .unwrap_or_else(|_| "No todo".into());

        send(UiEvent::Status {
            iteration: iterations,
            tokens: estimate_tokens(&session.messages, &session.scratchpad),
            context_tokens: session.context_tokens,
            goal: session.goal.clone(),
            todo,
            elapsed: Instant::now().duration_since(
                deadline - Duration::from_secs(config.max_wall_secs),
            ),
        });
    }
}

// ============================================================
// TUI State
// ============================================================

#[derive(Debug, Clone)]
enum EntryKind {
    Log,
    Reasoning,
    Error,
}

#[derive(Debug, Clone)]
struct TranscriptEntry {
    kind: EntryKind,
    text: String,
}

#[derive(Debug, Clone)]
struct StatusInfo {
    iteration: usize,
    tokens: usize,
    context_tokens: usize,
    goal: String,
    elapsed: Duration,
}

struct TuiState {
    status: Option<StatusInfo>,
    transcript: Vec<TranscriptEntry>,
    scroll_offset: usize,
    todo_text: String,

    input_mode: Option<InputMode>,
    input_buffer: String,
    cursor_position: usize,

    agent_finished: bool,

    // Mouse drag-selection state (app-drawn, since raw mode disables the
    // terminal's own native selection on Windows).
    mouse_selecting: bool,
    selection_start: Option<(u16, u16)>,
    selection_end: Option<(u16, u16)>,

    // Snapshot of exactly what's on screen after the last draw, indexed
    // [row][col], used to pull text out of a selected rectangle.
    screen_text: Vec<Vec<char>>,
}

#[derive(Debug, Clone, Copy)]
enum InputMode {
    EditingGoal,
    AddingInstruction,
    EditingTodo,
}

impl TuiState {
    fn input_active(&self) -> bool {
        self.input_mode.is_some()
    }

    fn input_label(&self) -> &'static str {
        match self.input_mode {
            Some(InputMode::EditingGoal) => "Goal",
            Some(InputMode::AddingInstruction) => "Instruction",
            Some(InputMode::EditingTodo) => "Todo",
            None => "",
        }
    }

    fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.cursor_position = 0;
    }

    fn insert_text(&mut self, text: &str) {
        self.input_buffer.insert_str(self.cursor_position, text);
        self.cursor_position += text.len();
    }

    fn insert_char(&mut self, c: char) {
        self.input_buffer.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor_position == 0 {
            return;
        }
        let previous = self.input_buffer[..self.cursor_position]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.input_buffer.drain(previous..self.cursor_position);
        self.cursor_position = previous;
    }

    fn delete(&mut self) {
        if self.cursor_position >= self.input_buffer.len() {
            return;
        }
        let next = self.input_buffer[self.cursor_position..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.cursor_position + index)
            .unwrap_or(self.input_buffer.len());
        self.input_buffer.drain(self.cursor_position..next);
    }

    fn move_left(&mut self) {
        if self.cursor_position == 0 {
            return;
        }
        self.cursor_position = self.input_buffer[..self.cursor_position]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    fn move_right(&mut self) {
        if self.cursor_position >= self.input_buffer.len() {
            return;
        }
        if let Some((index, _ch)) = self.input_buffer[self.cursor_position..]
            .char_indices()
            .nth(1)
        {
            self.cursor_position += index;
        } else {
            self.cursor_position = self.input_buffer.len();
        }
    }

    fn move_home(&mut self) {
        self.cursor_position = 0;
    }

    fn move_end(&mut self) {
        self.cursor_position = self.input_buffer.len();
    }

    fn clear_selection(&mut self) {
        self.selection_start = None;
        self.selection_end = None;
        self.mouse_selecting = false;
    }
}

fn push_entry(state: &mut TuiState, kind: EntryKind, text: &str) {
    for raw_line in text.split('\n') {
        state.transcript.push(TranscriptEntry {
            kind: kind.clone(),
            text: raw_line.to_string(),
        });
    }

    const MAX_ENTRIES: usize = 5000;
    if state.transcript.len() > MAX_ENTRIES {
        let excess = state.transcript.len() - MAX_ENTRIES;
        state.transcript.drain(0..excess);
    }
}

// ============================================================
// Clipboard Helpers
// ============================================================

fn read_clipboard() -> Result<String> {
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
    clipboard
        .get_text()
        .map_err(|e| anyhow!("failed to read clipboard: {e}"))
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
    clipboard
        .set_text(text.to_string())
        .map_err(|e| anyhow!("failed to write clipboard: {e}"))
}

// ============================================================
// Mouse Drag-Selection
// ============================================================

fn normalize_selection(a: (u16, u16), b: (u16, u16)) -> (u16, u16, u16, u16) {
    // Returns (start_x, start_y, end_x, end_y) in reading order (top-to-bottom,
    // left-to-right), regardless of which direction the user dragged.
    if (a.1, a.0) <= (b.1, b.0) {
        (a.0, a.1, b.0, b.1)
    } else {
        (b.0, b.1, a.0, a.1)
    }
}

fn extract_selected_text(screen: &[Vec<char>], start: (u16, u16), end: (u16, u16)) -> String {
    let (sx, sy, ex, ey) = normalize_selection(start, end);
    let (sx, sy, ex) = (sx as usize, sy as usize, ex as usize);

    if sy >= screen.len() {
        return String::new();
    }

    let ey = (ey as usize).min(screen.len() - 1);

    let mut lines = Vec::new();

    for y in sy..=ey {
        let row = &screen[y];

        if row.is_empty() {
            lines.push(String::new());
            continue;
        }

        let row_max = row.len() - 1;

        // Single-line selection: just the highlighted span.
        // Multi-line: first row runs to end-of-line, middle rows are taken
        // in full, and the last row runs from the start of the line.
        let (from, to) = if sy == ey {
            (sx.min(row_max), ex.min(row_max))
        } else if y == sy {
            (sx.min(row_max), row_max)
        } else if y == ey {
            (0, ex.min(row_max))
        } else {
            (0, row_max)
        };

        if from > to {
            lines.push(String::new());
            continue;
        }

        let mut line: String = row[from..=to].iter().collect();
        while line.ends_with(' ') {
            line.pop();
        }
        lines.push(line);
    }

    lines.join("\n")
}

fn handle_mouse_event(mouse: MouseEvent, state: &mut TuiState) {
    if state.input_active() {
        return;
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            state.mouse_selecting = true;
            state.selection_start = Some((mouse.column, mouse.row));
            state.selection_end = Some((mouse.column, mouse.row));
        }

        MouseEventKind::Drag(MouseButton::Left) => {
            if state.mouse_selecting {
                state.selection_end = Some((mouse.column, mouse.row));
            }
        }

        MouseEventKind::Up(MouseButton::Left) => {
            if state.mouse_selecting {
                state.mouse_selecting = false;

                if let (Some(start), Some(end)) = (state.selection_start, state.selection_end) {
                    let text = extract_selected_text(&state.screen_text, start, end);
                    if !text.is_empty() {
                        match copy_to_clipboard(&text) {
                            Ok(()) => push_entry(
                                state,
                                EntryKind::Log,
                                &format!("Copied {} chars to clipboard.", text.len()),
                            ),
                            Err(e) => push_entry(
                                state,
                                EntryKind::Error,
                                &format!("Copy failed: {e}"),
                            ),
                        }
                    }
                }

                state.clear_selection();
            }
        }

        MouseEventKind::ScrollUp => {
            state.scroll_offset = state.scroll_offset.saturating_add(3);
        }

        MouseEventKind::ScrollDown => {
            state.scroll_offset = state.scroll_offset.saturating_sub(3);
        }

        _ => {}
    }
}

// ============================================================
// Text Wrapping
// ============================================================

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![line.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for word in line.split(' ') {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }

    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }

    lines
}

// ============================================================
// Drawing
// ============================================================

fn draw_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
) -> Result<()> {
    terminal.draw(|frame| {
        let area = frame.size();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),
                Constraint::Length(8),
                Constraint::Length(1),
                Constraint::Length(3),
            ])
            .split(area);

        // ---- Transcript (scrollable) ----
        let transcript_area = chunks[0];
        let inner_width = transcript_area.width.saturating_sub(2).max(1) as usize;
        let visible_height = transcript_area.height.saturating_sub(2).max(1) as usize;

        let mut wrapped: Vec<(EntryKind, String)> = Vec::new();
        for entry in &state.transcript {
            if entry.text.is_empty() {
                wrapped.push((entry.kind.clone(), String::new()));
                continue;
            }
            for w in wrap_line(&entry.text, inner_width) {
                wrapped.push((entry.kind.clone(), w));
            }
        }

        let total = wrapped.len();
        let max_offset = total.saturating_sub(visible_height);
        if state.scroll_offset > max_offset {
            state.scroll_offset = max_offset;
        }
        let end = total.saturating_sub(state.scroll_offset);
        let start = end.saturating_sub(visible_height);

        let lines: Vec<Line> = wrapped[start..end]
            .iter()
            .map(|(kind, text)| {
                let style = match kind {
                    EntryKind::Log => Style::default().fg(Color::White),
                    EntryKind::Reasoning => Style::default().fg(Color::Cyan),
                    EntryKind::Error => Style::default().fg(Color::Red),
                };
                Line::from(Span::styled(text.clone(), style))
            })
            .collect();

        let transcript_title = if state.scroll_offset > 0 {
            format!(
                "Transcript (scrolled, {} lines back — End to jump to latest)",
                state.scroll_offset
            )
        } else {
            "Transcript".to_string()
        };

        let transcript = Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).title(transcript_title),
        );
        frame.render_widget(transcript, transcript_area);

        // ---- Todo panel (persistent, editable) ----
        let editing_todo = matches!(state.input_mode, Some(InputMode::EditingTodo));

        let todo_title = if editing_todo {
            "Todo — editing (Enter: newline, Ctrl+S: save, Esc: cancel)"
        } else {
            "Todo (t to edit)"
        };

        let todo_display = if editing_todo {
            state.input_buffer.clone()
        } else if state.todo_text.trim().is_empty() {
            "No todo yet.".to_string()
        } else {
            state.todo_text.clone()
        };

        let todo = Paragraph::new(todo_display)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(todo_title));
        frame.render_widget(todo, chunks[1]);

        // ---- Status line ----
        let status_text = match &state.status {
            Some(s) => {
                let pct = if s.context_tokens > 0 {
                    s.tokens * 100 / s.context_tokens
                } else {
                    0
                };
                format!(
                    "Goal: {} | iter {} | ctx {}/{} ({}%) | elapsed {:?} | agent: {}",
                    truncate_display(&s.goal, 40),
                    s.iteration,
                    s.tokens,
                    s.context_tokens,
                    pct,
                    s.elapsed,
                    if state.agent_finished { "finished" } else { "running" }
                )
            }
            None => "waiting for status...".to_string(),
        };
        frame.render_widget(
            Paragraph::new(status_text).style(Style::default().fg(Color::DarkGray)),
            chunks[2],
        );

        // ---- Input / controls ----
        if state.input_active() {
            let label = state.input_label();
            let input = Paragraph::new(state.input_buffer.as_str())
                .wrap(Wrap { trim: false })
                .style(Style::default().add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL).title(label));
            frame.render_widget(input, chunks[3]);

            // Precise cursor placement is only shown for single-line inputs;
            // the todo editor can wrap/contain newlines so we skip it there.
            if !editing_todo {
                let cursor_x = chunks[3].x
                    + 1
                    + state.input_buffer[..state.cursor_position].chars().count() as u16;
                let max_x = chunks[3].x + chunks[3].width.saturating_sub(1);
                let cursor_y = chunks[3].y + 1;
                frame.set_cursor(cursor_x.min(max_x), cursor_y);
            }
        } else {
            let controls =
                "p:pause r:resume i:instruction g:goal t:todo q:quit  ↑↓/PgUp/PgDn/wheel: scroll  drag: select+copy";
            let help = Paragraph::new(controls).block(
                Block::default().borders(Borders::ALL).title("Controls"),
            );
            frame.render_widget(help, chunks[3]);
        }

        // ---- Selection highlight (drawn last, on top of everything) ----
        if let (Some(start), Some(end)) = (state.selection_start, state.selection_end) {
            let (sx, sy, ex, ey) = normalize_selection(start, end);

            let sx = sx.min(area.width.saturating_sub(1));
            let ex = ex.min(area.width.saturating_sub(1));
            let sy = sy.min(area.height.saturating_sub(1));
            let ey = ey.min(area.height.saturating_sub(1));

            let highlight_style = Style::default()
                .bg(Color::White)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD);

            for y in sy..=ey {
                let row_max = area.width.saturating_sub(1);

                let (from, to) = if sy == ey {
                    (sx.min(row_max), ex.min(row_max))
                } else if y == sy {
                    // First line: from start column to end of line
                    (sx.min(row_max), row_max)
                } else if y == ey {
                    // Last line: from start of line to end column
                    (0, ex.min(row_max))
                } else {
                    // Middle lines: whole line
                    (0, row_max)
                };

                if from <= to {
                    for x in from..=to {
                        let cell = frame.buffer_mut().get_mut(x, y);
                        cell.set_style(highlight_style);
                    }
                }
            }
        }

        // ---- Capture the fully rendered screen for selection extraction ----
        let mut screen = vec![vec![' '; area.width as usize]; area.height as usize];
        for y in 0..area.height {
            for x in 0..area.width {
                let cell = frame.buffer_mut().get(x, y);
                let symbol = cell.symbol();
                screen[y as usize][x as usize] = symbol.chars().next().unwrap_or(' ');
            }
        }
        state.screen_text = screen;
    })?;

    Ok(())
}

// ============================================================
// Input Handling
// ============================================================

fn start_input(state: &mut TuiState, mode: InputMode, prefill: &str) {
    state.input_mode = Some(mode);
    state.input_buffer = prefill.to_string();
    state.cursor_position = state.input_buffer.len();
    state.clear_selection();
}

fn finish_input(state: &mut TuiState, tx_cmd: &mpsc::UnboundedSender<AgentCommand>) {
    let input = state.input_buffer.clone();

    if input.is_empty() {
        state.input_mode = None;
        state.clear_input();
        return;
    }

    match state.input_mode {
        Some(InputMode::EditingGoal) => {
            let _ = tx_cmd.send(AgentCommand::UpdateGoal(input.clone()));
            push_entry(state, EntryKind::Log, &format!("Goal updated to: {}", input));
        }
        Some(InputMode::AddingInstruction) => {
            let _ = tx_cmd.send(AgentCommand::AddInstruction(input.clone()));
            push_entry(state, EntryKind::Log, &format!("Instruction added: {}", input));
        }
        Some(InputMode::EditingTodo) => {
            let _ = tx_cmd.send(AgentCommand::UpdateTodo(input.clone()));
            state.todo_text = input.clone();
            push_entry(state, EntryKind::Log, "Todo updated.");
        }
        None => {}
    }

    state.input_mode = None;
    state.clear_input();
}

fn cancel_input(state: &mut TuiState) {
    state.input_mode = None;
    state.clear_input();
    push_entry(state, EntryKind::Log, "Input cancelled.");
}

fn handle_input_key(
    key: KeyEvent,
    state: &mut TuiState,
    tx_cmd: &mpsc::UnboundedSender<AgentCommand>,
) -> Result<()> {
    if key.kind != KeyEventKind::Press {
        return Ok(());
    }

    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let multiline = matches!(state.input_mode, Some(InputMode::EditingTodo));

    match key.code {
        KeyCode::Enter if multiline => state.insert_char('\n'),
        KeyCode::Enter => finish_input(state, tx_cmd),
        KeyCode::Char('s') if ctrl => finish_input(state, tx_cmd),
        KeyCode::Esc => cancel_input(state),
        KeyCode::Char('c') if ctrl => cancel_input(state),
        KeyCode::Char('v') if ctrl => match read_clipboard() {
            Ok(text) => state.insert_text(&text),
            Err(error) => push_entry(state, EntryKind::Error, &format!("Paste failed: {error}")),
        },
        KeyCode::Char('V') if ctrl && shift => match read_clipboard() {
            Ok(text) => state.insert_text(&text),
            Err(error) => push_entry(state, EntryKind::Error, &format!("Paste failed: {error}")),
        },
        KeyCode::Backspace => state.backspace(),
        KeyCode::Delete => state.delete(),
        KeyCode::Left => state.move_left(),
        KeyCode::Right => state.move_right(),
        KeyCode::Home => state.move_home(),
        KeyCode::End => state.move_end(),
        KeyCode::Char(c) if !ctrl => state.insert_char(c),
        _ => {}
    }

    Ok(())
}

// ============================================================
// TUI
// ============================================================

async fn run_tui(
    mut rx_ui: mpsc::UnboundedReceiver<UiEvent>,
    tx_cmd: mpsc::UnboundedSender<AgentCommand>,
) -> Result<()> {
    enable_raw_mode()?;

    let mut stdout = io::stdout();

    // Mouse capture is enabled so the app can draw its own drag-selection
    // and copy to the clipboard directly. This is necessary on Windows,
    // where enable_raw_mode() disables QuickEdit Mode (the console feature
    // that normally powers native click-drag copy), so the terminal can't
    // do it for us.
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture,
        Hide
    )?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut state = TuiState {
        status: None,
        transcript: Vec::new(),
        scroll_offset: 0,
        todo_text: String::new(),
        input_mode: None,
        input_buffer: String::new(),
        cursor_position: 0,
        agent_finished: false,
        mouse_selecting: false,
        selection_start: None,
        selection_end: None,
        screen_text: Vec::new(),
    };

    let result = async {
        loop {
            while let Ok(ui_event) = rx_ui.try_recv() {
                match ui_event {
                    UiEvent::Log(line) => {
                        push_entry(&mut state, EntryKind::Log, &line);
                    }
                    UiEvent::Reasoning { content } => {
                        push_entry(
                            &mut state,
                            EntryKind::Reasoning,
                            &format!("── model output ──\n{}", content),
                        );
                    }
                    UiEvent::Status {
                        iteration,
                        tokens,
                        context_tokens,
                        goal,
                        todo,
                        elapsed,
                    } => {
                        state.status = Some(StatusInfo {
                            iteration,
                            tokens,
                            context_tokens,
                            goal,
                            elapsed,
                        });
                        // Don't clobber the todo panel while the user is
                        // actively editing it.
                        if !matches!(state.input_mode, Some(InputMode::EditingTodo)) {
                            state.todo_text = todo;
                        }
                    }
                    UiEvent::AgentFinished { reason } => {
                        push_entry(
                            &mut state,
                            EntryKind::Log,
                            &format!("Agent finished: {reason}"),
                        );
                        state.agent_finished = true;
                    }
                    UiEvent::AgentError { error } => {
                        push_entry(&mut state, EntryKind::Error, &format!("Agent error: {error}"));
                        state.agent_finished = true;
                    }
                    UiEvent::Quit => {
                        return Ok(());
                    }
                }
            }

            draw_ui(&mut terminal, &mut state)?;

            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    CEvent::Paste(text) => {
                        if state.input_active() {
                            state.insert_text(&text);
                        }
                    }

                    CEvent::Key(key) => {
                        if state.input_active() {
                            handle_input_key(key, &mut state, &tx_cmd)?;
                        } else {
                            if key.kind != KeyEventKind::Press {
                                continue;
                            }

                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                            let shift = key.modifiers.contains(KeyModifiers::SHIFT);

                            match key.code {
                                KeyCode::Char('c') if ctrl => {
                                    let _ = tx_cmd.send(AgentCommand::Quit);
                                    break;
                                }
                                KeyCode::Char('p') if !ctrl && !shift => {
                                    let _ = tx_cmd.send(AgentCommand::Pause);
                                }
                                KeyCode::Char('r') if !ctrl && !shift => {
                                    let _ = tx_cmd.send(AgentCommand::Resume);
                                }
                                KeyCode::Char('i') if !ctrl && !shift => {
                                    start_input(&mut state, InputMode::AddingInstruction, "");
                                }
                                KeyCode::Char('g') if !ctrl && !shift => {
                                    start_input(&mut state, InputMode::EditingGoal, "");
                                }
                                KeyCode::Char('t') if !ctrl && !shift => {
                                    let prefill = state.todo_text.clone();
                                    start_input(&mut state, InputMode::EditingTodo, &prefill);
                                }
                                KeyCode::Char('q') if !ctrl && !shift => {
                                    let _ = tx_cmd.send(AgentCommand::Quit);
                                    break;
                                }
                                KeyCode::Up => {
                                    state.scroll_offset = state.scroll_offset.saturating_add(1);
                                }
                                KeyCode::Down => {
                                    state.scroll_offset = state.scroll_offset.saturating_sub(1);
                                }
                                KeyCode::PageUp => {
                                    state.scroll_offset = state.scroll_offset.saturating_add(10);
                                }
                                KeyCode::PageDown => {
                                    state.scroll_offset = state.scroll_offset.saturating_sub(10);
                                }
                                KeyCode::Home => {
                                    state.scroll_offset = usize::MAX;
                                }
                                KeyCode::End => {
                                    state.scroll_offset = 0;
                                }
                                _ => {}
                            }
                        }
                    }

                    CEvent::Mouse(mouse_event) => {
                        handle_mouse_event(mouse_event, &mut state);
                    }

                    CEvent::Resize(_, _) => {}
                    CEvent::FocusGained | CEvent::FocusLost => {}
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    }
    .await;

    disable_raw_mode()?;

    execute!(
        terminal.backend_mut(),
        Show,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;

    terminal.show_cursor()?;
    io::stdout().flush()?;

    result
}

// ============================================================
// CLI
// ============================================================

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// Base URL of OpenAI-compatible endpoint.
    /// Example: http://localhost:8081/v1
    #[clap(long)]
    base_url: String,

    /// Model name.
    #[clap(long)]
    model: String,

    /// Agent goal.
    #[clap(long)]
    goal: String,

    /// Working directory / sandbox root.
    #[clap(long)]
    workdir: PathBuf,

    /// Existing session to resume.
    #[clap(long)]
    session: Option<String>,

    /// Context token budget.
    /// 0 = automatically detect from server.
    #[clap(long, default_value_t = 0)]
    context_tokens: usize,

    /// Maximum iterations.
    #[clap(long, default_value_t = 200)]
    max_iterations: usize,

    /// Maximum wall-clock runtime.
    #[clap(long, default_value_t = 28800)]
    max_wall_secs: u64,

    /// Disable TUI and use plain logging.
    #[clap(long)]
    no_tui: bool,
}

// ============================================================
// Main
// ============================================================

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();

    let context_tokens = if config.context_tokens > 0 {
        config.context_tokens
    } else {
        detect_context_size(&config.base_url).await.unwrap_or(8192)
    };

    println!("Using context size: {}", context_tokens);

    let client = reqwest::Client::new();
    let model = Model {
        client,
        base_url: config.base_url.clone(),
        model: config.model.clone(),
        temperature: 0.7,
    };

    let session_id = match config.session.clone() {
        Some(id) => id,
        None => Uuid::new_v4().to_string(),
    };

    let mut session = match load_session(&session_id, config.workdir.clone(), context_tokens).await {
        Ok(session) => {
            println!("Resuming session {}", session_id);
            session
        }
        Err(_) => {
            println!("Creating new session {}", session_id);
            create_session(&session_id, &config.goal, config.workdir.clone(), context_tokens).await?
        }
    };

    tokio::fs::create_dir_all(&session.workdir).await?;

    if config.no_tui {
        let (tx_ui, _rx_ui) = mpsc::unbounded_channel();
        let (_tx_cmd, rx_cmd) = mpsc::unbounded_channel();
        run_agent(&config, &model, &mut session, tx_ui, rx_cmd).await?;
    } else {
        let (tx_ui, rx_ui) = mpsc::unbounded_channel();
        let (tx_cmd, rx_cmd) = mpsc::unbounded_channel();

        let agent_config = config.clone();
        let agent_model = model.clone();
        let mut agent_session = session;

        let agent_handle = tokio::spawn(async move {
            if let Err(error) = run_agent(
                &agent_config,
                &agent_model,
                &mut agent_session,
                tx_ui.clone(),
                rx_cmd,
            )
            .await
            {
                let _ = tx_ui.send(UiEvent::AgentError {
                    error: error.to_string(),
                });
            }
        });

        let tui_result = run_tui(rx_ui, tx_cmd).await;

        if let Err(error) = tui_result {
            eprintln!("TUI error: {error}");
            let _ = agent_handle.await;
            return Err(error);
        }

        let _ = agent_handle.await;
    }

    Ok(())
}