//! Night Agent — an overnight autonomous agent harness with TUI, todo list, web search,
//! interactive control, context auto-detection, and native text selection (no alternate screen).

use anyhow::{anyhow, Result};
use clap::Parser;
use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
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
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

// ------------------------- Data Structures -------------------------

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
    AgentFinished { reason: String },
    AgentError { error: String },
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
        }

        match serde_json::from_str::<RawToolCall>(raw) {
            Ok(r) => calls.push(ToolCall {
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

async fn execute_tool(call: &ToolCall, workdir: &Path, todo_path: &Path) -> Result<String> {
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
        "update_todo" => {
            let content = call.arguments["content"].as_str().ok_or_else(|| anyhow!("missing content"))?;
            tokio::fs::write(todo_path, content).await?;
            Ok(format!("todo list updated:\n{}", content))
        }
        "get_todo" => {
            let content = match tokio::fs::read_to_string(todo_path).await {
                Ok(s) => s,
                Err(_) => "No todo list yet.".to_string(),
            };
            Ok(content)
        }
        "search_web" => {
            let query = call.arguments["query"].as_str().ok_or_else(|| anyhow!("missing query"))?;
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

// ------------------------- Web Search -------------------------

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

    let re = Regex::new(r#"<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>"#).unwrap();
    let mut results = String::new();
    let mut count = 0;
    for cap in re.captures_iter(&body) {
        if count >= 5 {
            break;
        }
        let url = cap[1].to_string();
        let title = strip_html(&cap[2]);
        let snippet = strip_html(&cap[3]);
        results.push_str(&format!("{}. {}\nURL: {}\n{}\n\n", count + 1, title, url, snippet));
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

// ------------------------- Session Management -------------------------

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
        messages: vec![],
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
        todo_path,
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
    chars / 4
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
- update_todo(content)   # Update the todo list (overwrite with new markdown)
- get_todo()              # Read the current todo list
- search_web(query)       # Search the web for information
- finish(reason)

Always output tool calls as:
<tool_call>
{{"name":"tool_name","arguments":{{...}}}}
</tool_call>

Rules:
- Never ask for permission.
- Never stop until the goal is verified.
- If a tool fails, read the error and try a different approach.
- You are operating unsupervised. Do not ask questions.
- Maintain a todo list to track progress. Update it whenever you start or finish a task.
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

// ------------------------- Auto Context Detection -------------------------

async fn detect_context_size(base_url: &str) -> Option<usize> {
    let client = reqwest::Client::new();
    let url = format!("{}/props", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let val: serde_json::Value = resp.json().await.ok()?;
    val["default_generation_settings"]["n_ctx"]
        .as_u64()
        .map(|v| v as usize)
}

// ------------------------- Agent Loop -------------------------

async fn run_agent(
    config: &Config,
    model: &Model,
    session: &mut Session,
    mut tx_ui: mpsc::UnboundedSender<UiEvent>,
    mut rx_cmd: mpsc::UnboundedReceiver<AgentCommand>,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(config.max_wall_secs);
    let mut iterations = 0;
    let mut malformed_streak = 0;
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
    send(UiEvent::Log(format!("Context limit: {}", session.context_tokens)));

    loop {
        // Process commands
        while let Ok(cmd) = rx_cmd.try_recv() {
            match cmd {
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
                    let msg = Message {
                        role: "user".into(),
                        content: format!("New instruction from user: {}", instruction),
                    };
                    session.messages.push(msg.clone());
                    session.append_message(&msg).await?;
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
                send(UiEvent::Log("Hit max iterations".into()));
            } else {
                send(UiEvent::Log("Hit wall-clock limit".into()));
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
            Ok(r) => r,
            Err(_) => {
                let msg = Message {
                    role: "user".into(),
                    content: "Model call timed out. Continue where you left off.".into(),
                };
                session.messages.push(msg.clone());
                session.append_message(&msg).await?;
                send(UiEvent::Log("Model timed out, nudging to continue.".into()));
                continue;
            }
        };

        send(UiEvent::Reasoning {
            content: response.clone(),
        });

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

                    send(UiEvent::Log(format!(
                        "Result: {}",
                        truncate(&result, 200)
                    )));

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
                malformed_streak += 1;
                let nudge = if malformed_streak > 3 {
                    "You appear to be stuck. Re-read the goal, update your plan, and take a concrete action using a tool call."
                } else {
                    "Continue working autonomously. Output a tool call next."
                };
                send(UiEvent::Log(format!("No tool call, nudging: {nudge}")));
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
                send(UiEvent::Log(format!("Malformed tool call: {err}")));
                let msg = Message {
                    role: "user".into(),
                    content: correction,
                };
                session.messages.push(msg.clone());
                session.append_message(&msg).await?;
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
            elapsed: Instant::now().duration_since(deadline - Duration::from_secs(config.max_wall_secs)),
        });
    }
}

// ------------------------- TUI -------------------------

struct TuiState {
    logs: Vec<String>,
    status: Option<UiEvent>,
    last_reasoning: String,
    input_mode: Option<InputMode>,
    input_buffer: String,
    agent_finished: bool,
}

enum InputMode {
    EditingGoal,
    AddingInstruction,
}

async fn run_tui(
    mut rx_ui: mpsc::UnboundedReceiver<UiEvent>,
    tx_cmd: mpsc::UnboundedSender<AgentCommand>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = TuiState {
        logs: Vec::new(),
        status: None,
        last_reasoning: String::new(),
        input_mode: None,
        input_buffer: String::new(),
        agent_finished: false,
    };

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(25), // Status
                    Constraint::Percentage(35), // Reasoning
                    Constraint::Percentage(35), // Logs
                    Constraint::Percentage(5),  // Help
                ])
                .split(f.size());

            let status_text = match &state.status {
                Some(UiEvent::Status { iteration, tokens, context_tokens, goal, todo, elapsed }) => {
                    format!(
                        "Goal: {}\nIteration: {}\nContext: {}/{} tokens ({}%)\nElapsed: {:?}\n\nTodo:\n{}",
                        goal,
                        iteration,
                        tokens,
                        context_tokens,
                        if *context_tokens > 0 { tokens * 100 / context_tokens } else { 0 },
                        elapsed,
                        todo
                    )
                }
                _ => "Waiting for status...".to_string(),
            };
            let status_paragraph = Paragraph::new(status_text)
                .block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(status_paragraph, chunks[0]);

            let reasoning_text: Vec<Line> = state
                .last_reasoning
                .lines()
                .map(|l| Line::from(Span::raw(l.to_string())))
                .collect();
            let reasoning_paragraph = Paragraph::new(reasoning_text)
                .block(Block::default().borders(Borders::ALL).title("Latest Model Output"));
            f.render_widget(reasoning_paragraph, chunks[1]);

            let log_text: Vec<Line> = state
                .logs
                .iter()
                .rev()
                .take(100)
                .rev()
                .map(|l| Line::from(Span::raw(l.clone())))
                .collect();
            let log_paragraph = Paragraph::new(log_text)
                .block(Block::default().borders(Borders::ALL).title("Log"));
            f.render_widget(log_paragraph, chunks[2]);

            let help_text = if let Some(mode) = &state.input_mode {
                match mode {
                    InputMode::EditingGoal => format!("Enter new goal: {}", state.input_buffer),
                    InputMode::AddingInstruction => format!("Enter instruction: {}", state.input_buffer),
                }
            } else {
                format!(
                    "p: pause | r: resume | i: instruction | g: edit goal | q: quit | Agent: {}",
                    if state.agent_finished { "finished" } else { "running" }
                )
            };
            let help_paragraph = Paragraph::new(help_text)
                .block(Block::default().borders(Borders::ALL).title("Controls"));
            f.render_widget(help_paragraph, chunks[3]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let CEvent::Key(key) = event::read()? {
                if let Some(mode) = &state.input_mode {
                    match key.code {
                        KeyCode::Enter => {
                            let input = state.input_buffer.clone();
                            match mode {
                                InputMode::EditingGoal => {
                                    let _ = tx_cmd.send(AgentCommand::UpdateGoal(input.clone()));
                                    state.logs.push(format!("Goal updated to: {}", input));
                                }
                                InputMode::AddingInstruction => {
                                    let _ = tx_cmd.send(AgentCommand::AddInstruction(input.clone()));
                                    state.logs.push(format!("Instruction added: {}", input));
                                }
                            }
                            state.input_mode = None;
                            state.input_buffer.clear();
                        }
                        KeyCode::Esc => {
                            state.input_mode = None;
                            state.input_buffer.clear();
                        }
                        KeyCode::Char(c) => {
                            state.input_buffer.push(c);
                        }
                        KeyCode::Backspace => {
                            state.input_buffer.pop();
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Char('p') => {
                            let _ = tx_cmd.send(AgentCommand::Pause);
                        }
                        KeyCode::Char('r') => {
                            let _ = tx_cmd.send(AgentCommand::Resume);
                        }
                        KeyCode::Char('i') => {
                            state.input_mode = Some(InputMode::AddingInstruction);
                            state.input_buffer.clear();
                        }
                        KeyCode::Char('g') => {
                            state.input_mode = Some(InputMode::EditingGoal);
                            state.input_buffer.clear();
                        }
                        KeyCode::Char('q') => {
                            let _ = tx_cmd.send(AgentCommand::Quit);
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        while let Ok(event) = rx_ui.try_recv() {
            match event {
                UiEvent::Log(line) => state.logs.push(line),
                UiEvent::Status { .. } => {
                    state.status = Some(event);
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
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    terminal.show_cursor()?;
    Ok(())
}

// ------------------------- CLI & Main -------------------------

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// Base URL of the OpenAI-compatible endpoint (e.g. http://localhost:8081/v1)
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

    /// Context token budget (0 = auto-detect from server)
    #[clap(long, default_value_t = 0)]
    context_tokens: usize,

    /// Max iterations
    #[clap(long, default_value_t = 200)]
    max_iterations: usize,

    /// Max wall-clock seconds
    #[clap(long, default_value_t = 28800)]
    max_wall_secs: u64,

    /// Disable the TUI and use plain logging
    #[clap(long)]
    no_tui: bool,
}

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

    let mut session = if let Ok(s) = load_session(&session_id, config.workdir.clone(), context_tokens).await {
        println!("Resuming session {session_id}");
        s
    } else {
        println!("Creating new session {session_id}");
        create_session(&session_id, &config.goal, config.workdir.clone(), context_tokens).await?
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
            if let Err(e) = run_agent(&agent_config, &agent_model, &mut agent_session, tx_ui, rx_cmd).await {
                eprintln!("Agent error: {e}");
            }
        });

        run_tui(rx_ui, tx_cmd).await?;

        let _ = agent_handle.await;
    }

    Ok(())
}