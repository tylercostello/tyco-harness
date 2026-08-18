//! Night Agent — an overnight autonomous agent harness with TUI,
//! todo list, web search, interactive control, context compaction,
//! session persistence, and full mouse selection/copy.

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
    widgets::{Block, Borders, Paragraph},
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
        messages.extend(session.messages.iter().cloned());

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

struct TuiState {
    logs: Vec<String>,
    status: Option<UiEvent>,
    last_reasoning: String,

    input_mode: Option<InputMode>,
    input_buffer: String,
    cursor_position: usize,

    agent_finished: bool,

    // Mouse selection state
    mouse_selecting: bool,
    selection_start: Option<(u16, u16)>,
    selection_end: Option<(u16, u16)>,

    // Rendered text buffer: screen_text[row][col]
    screen_text: Vec<Vec<char>>,
}

#[derive(Debug, Clone, Copy)]
enum InputMode {
    EditingGoal,
    AddingInstruction,
}

impl TuiState {
    fn input_active(&self) -> bool {
        self.input_mode.is_some()
    }

    fn input_label(&self) -> &'static str {
        match self.input_mode {
            Some(InputMode::EditingGoal) => "Goal",
            Some(InputMode::AddingInstruction) => "Instruction",
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

// ============================================================
// Clipboard Helpers
// ============================================================

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
    clipboard
        .set_text(text.to_string())
        .map_err(|e| anyhow!("failed to write clipboard: {e}"))
}

fn read_clipboard() -> Result<String> {
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
    clipboard
        .get_text()
        .map_err(|e| anyhow!("failed to read clipboard: {e}"))
}

// ============================================================
// Selection Helpers
// ============================================================

fn extract_selected_text(
    screen: &[Vec<char>],
    start: (u16, u16),
    end: (u16, u16),
) -> Result<String> {
    let (x1, y1) = start;
    let (x2, y2) = end;

    let x_min = x1.min(x2) as usize;
    let x_max = x1.max(x2) as usize;
    let y_min = y1.min(y2) as usize;
    let y_max = y1.max(y2) as usize;

    if y_min >= screen.len() {
        return Ok(String::new());
    }

    let mut lines = Vec::new();

    for y in y_min..=y_max.min(screen.len() - 1) {
        let row = &screen[y];
        if x_min >= row.len() {
            continue;
        }
        let end_x = x_max.min(row.len() - 1);
        let mut line: String = row[x_min..=end_x].iter().collect();
        // Trim trailing whitespace
        while line.ends_with(' ') {
            line.pop();
        }
        lines.push(line);
    }

    Ok(lines.join("\n"))
}

fn apply_selection_highlight(state: &TuiState, screen: &mut Vec<Vec<char>>) {
    if let (Some(start), Some(end)) = (state.selection_start, state.selection_end) {
        let x_min = start.0.min(end.0) as usize;
        let x_max = start.0.max(end.0) as usize;
        let y_min = start.1.min(end.1) as usize;
        let y_max = start.1.max(end.1) as usize;

        for y in y_min..=y_max.min(screen.len() - 1) {
            if y >= screen.len() {
                break;
            }
            for x in x_min..=x_max.min(screen[y].len() - 1) {
                if x >= screen[y].len() {
                    break;
                }
                // We'll highlight by using a special character? For now we don't change content.
                // The actual visual highlight is done in draw_tui via buffer styles.
                // This function is no longer needed.
            }
        }
    }
}

// ============================================================
// Mouse Event Handling
// ============================================================

fn handle_mouse_event(mouse: MouseEvent, state: &mut TuiState) -> Result<()> {
    if state.input_active() {
        return Ok(());
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
                    let text = extract_selected_text(&state.screen_text, start, end)?;
                    if !text.is_empty() {
                        copy_to_clipboard(&text)?;
                        state.logs.push(format!("Copied {} chars", text.len()));
                    }
                }

                state.clear_selection();
            }
        }

        _ => {}
    }

    Ok(())
}

// ============================================================
// Drawing
// ============================================================

fn draw_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
) -> Result<()> {
    terminal.draw(|frame| {
        let area = frame.size();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(22),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
                Constraint::Percentage(12),
            ])
            .split(area);

        // Status
        let status_text = match &state.status {
            Some(UiEvent::Status {
                iteration,
                tokens,
                context_tokens,
                goal,
                todo,
                elapsed,
            }) => format!(
                "Goal: {}\nIteration: {}\nContext: {}/{} tokens ({}%)\nElapsed: {:?}\n\nTodo:\n{}",
                goal,
                iteration,
                tokens,
                context_tokens,
                if *context_tokens > 0 {
                    tokens * 100 / context_tokens
                } else {
                    0
                },
                elapsed,
                todo
            ),
            _ => "Waiting for status...".to_string(),
        };
        let status = Paragraph::new(status_text).block(
            Block::default().borders(Borders::ALL).title("Status"),
        );
        frame.render_widget(status, chunks[0]);

        // Reasoning
        let reasoning_lines: Vec<Line> = state
            .last_reasoning
            .lines()
            .map(|line| Line::from(Span::raw(line.to_string())))
            .collect();
        let reasoning = Paragraph::new(reasoning_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Latest Model Output"),
        );
        frame.render_widget(reasoning, chunks[1]);

        // Logs
        let log_lines: Vec<Line> = state
            .logs
            .iter()
            .rev()
            .take(100)
            .rev()
            .map(|line| Line::from(Span::raw(line.clone())))
            .collect();
        let logs = Paragraph::new(log_lines).block(
            Block::default().borders(Borders::ALL).title("Log"),
        );
        frame.render_widget(logs, chunks[2]);

        // Input / controls
        if state.input_active() {
            let input_text = format!("{}: {}", state.input_label(), state.input_buffer);
            let input = Paragraph::new(input_text)
                .style(Style::default().add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL).title("Input"));
            frame.render_widget(input, chunks[3]);

            let prefix_len = state.input_label().chars().count() + 2;
            let cursor_x = chunks[3].x
                + 1
                + (prefix_len + state.input_buffer[..state.cursor_position].chars().count())
                    as u16;
            let cursor_y = chunks[3].y + 1;
            frame.set_cursor(cursor_x, cursor_y);
        } else {
            let controls = format!(
                "p: pause | r: resume | i: instruction | g: goal | q: quit | Agent: {}",
                if state.agent_finished { "finished" } else { "running" }
            );
            let help = Paragraph::new(controls).block(
                Block::default().borders(Borders::ALL).title("Controls"),
            );
            frame.render_widget(help, chunks[3]);
        }

        // Apply selection highlight
        if let (Some(start), Some(end)) = (state.selection_start, state.selection_end) {
            let x_min = start.0.min(end.0);
            let x_max = start.0.max(end.0).min(area.width.saturating_sub(1));
            let y_min = start.1.min(end.1);
            let y_max = start.1.max(end.1).min(area.height.saturating_sub(1));

            let highlight_style = Style::default()
                .bg(Color::White)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD);

            if x_min <= x_max && y_min <= y_max {
                for y in y_min..=y_max {
                    for x in x_min..=x_max {
                        let cell = frame.buffer_mut().get_mut(x, y);
                        cell.set_style(highlight_style);
                    }
                }
            }
        }

        // Capture rendered screen text after all widgets are drawn.
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

fn start_input(state: &mut TuiState, mode: InputMode) {
    state.input_mode = Some(mode);
    state.clear_input();
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
            state.logs.push(format!("Goal updated to: {}", input));
        }
        Some(InputMode::AddingInstruction) => {
            let _ = tx_cmd.send(AgentCommand::AddInstruction(input.clone()));
            state.logs.push(format!("Instruction added: {}", input));
        }
        None => {}
    }

    state.input_mode = None;
    state.clear_input();
}

fn cancel_input(state: &mut TuiState) {
    state.input_mode = None;
    state.clear_input();
    state.logs.push("Input cancelled.".into());
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

    match key.code {
        KeyCode::Enter => {
            finish_input(state, tx_cmd);
        }
        KeyCode::Esc => {
            cancel_input(state);
        }
        KeyCode::Char('c') if ctrl => {
            cancel_input(state);
        }
        KeyCode::Char('v') if ctrl => {
            match read_clipboard() {
                Ok(text) => state.insert_text(&text),
                Err(error) => state.logs.push(format!("Paste failed: {error}")),
            }
        }
        KeyCode::Char('V') if ctrl && shift => {
            match read_clipboard() {
                Ok(text) => state.insert_text(&text),
                Err(error) => state.logs.push(format!("Paste failed: {error}")),
            }
        }
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
        logs: Vec::new(),
        status: None,
        last_reasoning: String::new(),
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
            draw_tui(&mut terminal, &mut state)?;

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
                                    start_input(&mut state, InputMode::AddingInstruction);
                                }
                                KeyCode::Char('g') if !ctrl && !shift => {
                                    start_input(&mut state, InputMode::EditingGoal);
                                }
                                KeyCode::Char('q') if !ctrl && !shift => {
                                    let _ = tx_cmd.send(AgentCommand::Quit);
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    CEvent::Mouse(mouse_event) => {
                        handle_mouse_event(mouse_event, &mut state)?;
                    }

                    CEvent::Resize(_, _) => {}

                    CEvent::FocusGained => {}
                    CEvent::FocusLost => {}
                }
            }

            while let Ok(ui_event) = rx_ui.try_recv() {
                match ui_event {
                    UiEvent::Log(line) => {
                        state.logs.push(line);
                        if state.logs.len() > 2000 {
                            let excess = state.logs.len() - 2000;
                            state.logs.drain(0..excess);
                        }
                    }
                    UiEvent::Status { .. } => {
                        state.status = Some(ui_event);
                    }
                    UiEvent::Reasoning { content } => {
                        state.last_reasoning = content;
                    }
                    UiEvent::AgentFinished { reason } => {
                        state.logs.push(format!("Agent finished: {}", reason));
                        state.agent_finished = true;
                    }
                    UiEvent::AgentError { error } => {
                        state.logs.push(format!("Agent error: {}", error));
                        state.agent_finished = true;
                    }
                    UiEvent::Quit => {
                        return Ok(());
                    }
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