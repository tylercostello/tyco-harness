//! Night Agent — an overnight autonomous agent harness with TUI,
//! scrollable transcript, editable todo panel, web search, interactive
//! control, context compaction, and session persistence.
//!
//! SECURITY NOTE: the `run_command` tool executes arbitrary shell commands
//! produced by a language model with the full privileges of this process.
//! `--workdir` is only the *default* directory; it is NOT a sandbox. Run this
//! inside a container/VM/jail if you care about isolation.

use anyhow::{anyhow, Context, Result};
use arboard::Clipboard;
use clap::{Parser, ValueEnum};
use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
        MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use once_cell::sync::Lazy;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use uuid::Uuid;

// ============================================================
// Data Structures
// ============================================================

static SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

#[derive(Clone, Default, Serialize, Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: String,
    /// Native (structured) calls made by the assistant in this turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<NativeToolCall>,
    /// Set only on `role: "tool"` results, linking back to a call above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl Message {
    fn new(role: &str, content: impl Into<String>) -> Self {
        Message {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: "tool".into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Renders native-shaped history back into the text protocol, so a session
/// recorded against a native model can be resumed against a text-mode one.
fn flatten_for_text_mode(messages: &[Message]) -> Vec<Message> {
    let mut names: HashMap<String, String> = HashMap::new();
    let mut out = Vec::with_capacity(messages.len());

    for message in messages {
        if message.role == "tool" {
            let name = message
                .tool_call_id
                .as_ref()
                .and_then(|id| names.get(id))
                .cloned()
                .unwrap_or_else(|| "tool".to_string());
            out.push(Message::new(
                "user",
                format!("Tool result for {}: {}", name, message.content),
            ));
            continue;
        }

        if message.tool_calls.is_empty() {
            out.push(Message::new(&message.role, message.content.clone()));
            continue;
        }

        let mut rendered = message.content.clone();
        for call in &message.tool_calls {
            names.insert(call.id.clone(), call.function.name.clone());
            let args = if call.function.arguments.trim().is_empty() {
                "{}".to_string()
            } else {
                call.function.arguments.clone()
            };
            if !rendered.is_empty() {
                rendered.push('\n');
            }
            rendered.push_str(&format!(
                "<tool_call>\n{{\"name\":\"{}\",\"arguments\":{}}}\n</tool_call>",
                call.function.name, args
            ));
        }
        out.push(Message::new(&message.role, rendered));
    }

    out
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NativeToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: NativeFunctionCall,
}

/// `arguments` stays a JSON *string* so it round-trips to the API byte-for-byte;
/// it is only parsed once the stream has fully assembled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NativeFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
enum Event {
    Message(Message),
    /// `retained` holds the messages that remained in context *after* the
    /// compaction. Older transcripts may not have it; those are legacy.
    Compaction {
        summary: String,
        #[serde(default)]
        retained: Option<Vec<Message>>,
    },
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
    Running(bool),
    AgentFinished {
        reason: String,
    },
    AgentError {
        error: String,
    },
    /// Signals that the agent has started waiting for a model response.
    ModelRequestStart,
    /// Signals that the agent has finished waiting (response received or interrupted).
    ModelRequestEnd,
    /// Clears the streaming buffer before a new model attempt starts.
    ReasoningReset,
    /// A single streamed token/delta from the model.
    ReasoningChunk {
        delta: String,
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
    CompactNow,
    SwitchSession(String),
    Quit,
}

/// Thin wrapper so the agent never has to reach for `eprintln!` (which would
/// corrupt the alternate screen).
#[derive(Clone)]
struct UiLogger {
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl UiLogger {
    fn new(tx: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self { tx }
    }
    fn log(&self, text: impl Into<String>) {
        let _ = self.tx.send(UiEvent::Log(text.into()));
    }
    fn send(&self, event: UiEvent) {
        let _ = self.tx.send(event);
    }

    /// Sends are already best-effort, so dropping the receiver simply discards
    /// every event.
    #[cfg(test)]
    fn disabled() -> Self {
        Self {
            tx: mpsc::unbounded_channel().0,
        }
    }
}

// ============================================================
// Model Client
// ============================================================

#[derive(Debug)]
enum ChatError {
    /// Retrying will not help (bad key, bad model, malformed schema).
    Fatal(String),
    /// Worth retrying (network blip, 429, 5xx, timeout).
    Transient(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Fatal(m) => write!(f, "{m}"),
            ChatError::Transient(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ChatError {}

fn join_url(base: &str, suffix: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

#[derive(Clone)]
struct Model {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f32,
    api_key: Option<String>,
    request_timeout_secs: u64,
    reasoning_effort: Option<String>,
    tool_mode: ToolMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ToolMode {
    /// Model emits `<tool_call>{...}</tool_call>` as text. Qwen's native format.
    Text,
    /// Model uses the OpenAI `tools` / `tool_calls` fields. GPT-class models.
    Native,
}

/// Only agent turns may carry tools; summarization and todo generation must
/// stay tool-free or the model answers them with a tool call instead of prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    PlainText,
    Agent,
}

#[derive(Debug, Default, Clone)]
struct ChatResponse {
    text: String,
    tool_calls: Vec<NativeToolCall>,
    finish_reason: Option<String>,
}

fn tool_definitions() -> serde_json::Value {
    fn spec(
        name: &str,
        description: &str,
        props: serde_json::Value,
        required: &[&str],
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            }
        })
    }

    let string_prop = |desc: &str| serde_json::json!({ "type": "string", "description": desc });

    serde_json::json!([
        spec(
            "read_file",
            "Read a UTF-8 text file relative to the working directory.",
            serde_json::json!({ "path": string_prop("Path to read.") }),
            &["path"]
        ),
        spec(
            "write_file",
            "Create or overwrite a file relative to the working directory.",
            serde_json::json!({
                "path": string_prop("Path to write."),
                "content": string_prop("Full file contents."),
            }),
            &["path", "content"]
        ),
        spec(
            "list_dir",
            "List the entries of a directory.",
            serde_json::json!({ "path": string_prop("Directory to list.") }),
            &["path"]
        ),
        spec(
            "run_command",
            "Run a shell command in the working directory and return its output.",
            serde_json::json!({ "command": string_prop("Shell command to run.") }),
            &["command"]
        ),
        spec(
            "update_todo",
            "Replace the todo list with new markdown checklist content.",
            serde_json::json!({ "content": string_prop("Full markdown checklist.") }),
            &["content"]
        ),
        spec(
            "get_todo",
            "Read the current todo list.",
            serde_json::json!({}),
            &[]
        ),
        spec(
            "search_web",
            "Search the web for a query.",
            serde_json::json!({ "query": string_prop("Search query.") }),
            &["query"]
        ),
        spec(
            "finish",
            "Declare the goal complete. Call this alone, only when verified.",
            serde_json::json!({ "reason": string_prop("Why the goal is complete.") }),
            &["reason"]
        ),
    ])
}

#[derive(Debug, PartialEq)]
enum StreamLine {
    Ignore,
    Done,
    Record(StreamRecord),
}

/// One SSE delta. `content` and `tool_calls` can both be present in a chunk,
/// so this is a record rather than an enum of alternatives.
#[derive(Debug, Default, PartialEq)]
struct StreamRecord {
    /// Text shown to the user and, in text mode, scanned for tool calls.
    content: Option<String>,
    /// Thinking tokens. Displayed only — never scanned for tool calls.
    reasoning: Option<String>,
    tool_calls: Vec<ToolCallDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, PartialEq)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
}

/// Reassembles a completion from SSE deltas. Tool-call fragments arrive keyed
/// by `index`, with `function.arguments` split across arbitrarily many chunks.
#[derive(Default)]
struct StreamAccumulator {
    text: String,
    saw_reasoning: bool,
    calls: BTreeMap<usize, NativeToolCall>,
    finish_reason: Option<String>,
}

impl StreamAccumulator {
    fn push(&mut self, record: StreamRecord, ui: &UiLogger) {
        if let Some(reason) = record.finish_reason {
            self.finish_reason = Some(reason);
        }

        // Reasoning is surfaced live but deliberately kept out of `text`, so it
        // can never be mistaken for a tool call by the text-mode extractor.
        if let Some(reasoning) = record.reasoning {
            self.saw_reasoning = true;
            ui.send(UiEvent::ReasoningChunk { delta: reasoning });
        }

        if let Some(content) = record.content {
            self.text.push_str(&content);
            ui.send(UiEvent::ReasoningChunk { delta: content });
        }

        for delta in record.tool_calls {
            let call = self.calls.entry(delta.index).or_default();
            if let Some(id) = delta.id {
                call.id = id;
            }
            if let Some(name) = delta.name {
                call.function.name.push_str(&name);
            }
            if let Some(arguments) = delta.arguments {
                call.function.arguments.push_str(&arguments);
            }
            if call.kind.is_empty() {
                call.kind = "function".into();
            }
        }
    }

    fn finish(self) -> Result<ChatResponse, ChatError> {
        let tool_calls: Vec<NativeToolCall> = self.calls.into_values().collect();
        if self.text.is_empty() && tool_calls.is_empty() && !self.saw_reasoning {
            return Err(ChatError::Fatal("empty streamed response".into()));
        }
        Ok(ChatResponse {
            text: self.text,
            tool_calls,
            finish_reason: self.finish_reason,
        })
    }
}

/// Pops one complete SSE line from the byte buffer, or `None` if no newline
/// has arrived yet. Operating on bytes (not `str`) is what keeps a multi-byte
/// character split across network chunks from being decoded as two U+FFFD.
fn take_sse_line(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pos = buf.iter().position(|&byte| byte == b'\n')?;
    let mut line: Vec<u8> = buf.drain(..=pos).collect();
    line.pop();
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Some(line)
}

fn parse_stream_line(line: &[u8]) -> Result<StreamLine, ChatError> {
    let line = std::str::from_utf8(line)
        .map_err(|e| ChatError::Transient(format!("invalid UTF-8 in model stream: {e}")))?
        .trim();
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(StreamLine::Ignore);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(StreamLine::Done);
    }

    // SSE streams may contain comments, keep-alives, or provider-specific
    // data records. Preserve the previous tolerant behavior for unknown data.
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return Ok(StreamLine::Ignore);
    };
    if let Some(err) = value.get("error") {
        if !err.is_null() {
            return Err(ChatError::Fatal(format!("api error: {err}")));
        }
    }

    let text = |path: &str| {
        value
            .pointer(path)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    };

    // Providers disagree on the field name for thinking tokens.
    let reasoning =
        text("/choices/0/delta/reasoning_content").or_else(|| text("/choices/0/delta/reasoning"));

    let mut tool_calls = Vec::new();
    if let Some(items) = value
        .pointer("/choices/0/delta/tool_calls")
        .and_then(|v| v.as_array())
    {
        for item in items {
            tool_calls.push(ToolCallDelta {
                index: item.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                id: item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                name: item
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                arguments: item
                    .pointer("/function/arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            });
        }
    }

    let record = StreamRecord {
        content: text("/choices/0/delta/content"),
        reasoning,
        tool_calls,
        finish_reason: value
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    };

    if record == StreamRecord::default() {
        return Ok(StreamLine::Ignore);
    }
    Ok(StreamLine::Record(record))
}

impl Model {
    fn request_body(
        &self,
        messages: &[Message],
        stream: bool,
        kind: RequestKind,
    ) -> serde_json::Value {
        let native = self.tool_mode == ToolMode::Native;
        let wire = if native {
            messages.to_vec()
        } else {
            flatten_for_text_mode(messages)
        };

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": wire,
            "temperature": self.temperature,
            "stream": stream
        });
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning_effort"] = serde_json::Value::String(effort.clone());
        }
        if native && kind == RequestKind::Agent {
            body["tools"] = tool_definitions();
            // Forcing a call is what stops these models from narrating
            // ("I'll finish now") instead of acting.
            body["tool_choice"] = serde_json::Value::String("required".into());
        }
        body
    }

    #[allow(dead_code)]
    async fn chat(&self, messages: &[Message], kind: RequestKind) -> Result<String, ChatError> {
        let url = join_url(&self.base_url, "chat/completions");
        let mut req = self
            .client
            .post(&url)
            .json(&self.request_body(messages, false, kind));
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ChatError::Transient(format!("request failed: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| ChatError::Transient(format!("failed to read body: {e}")))?;

        if !status.is_success() {
            let msg = format!("HTTP {status}: {}", truncate(&body, 400));
            return if matches!(status.as_u16(), 408 | 425 | 429) || status.is_server_error() {
                Err(ChatError::Transient(msg))
            } else {
                Err(ChatError::Fatal(msg))
            };
        }

        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ChatError::Transient(format!("invalid JSON from server: {e}")))?;

        // Some servers return errors with a 200 status.
        if let Some(err) = value.get("error") {
            if !err.is_null() {
                return Err(ChatError::Fatal(format!("api error: {err}")));
            }
        }

        let message = value.pointer("/choices/0/message").ok_or_else(|| {
            ChatError::Fatal(format!(
                "unexpected response shape (no choices[0].message): {}",
                truncate(&body, 400)
            ))
        })?;

        extract_message_content(message).ok_or_else(|| {
            ChatError::Fatal(format!(
                "could not extract text content from message: {}",
                truncate(&message.to_string(), 400)
            ))
        })
    }

    /// SSE streaming variant. Sends `ReasoningChunk` events as tokens arrive.
    async fn chat_stream(
        &self,
        messages: &[Message],
        ui: &UiLogger,
        kind: RequestKind,
    ) -> Result<ChatResponse, ChatError> {
        use futures_util::StreamExt;

        let url = join_url(&self.base_url, "chat/completions");
        let mut req = self
            .client
            .post(&url)
            .json(&self.request_body(messages, true, kind));
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ChatError::Transient(format!("request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let msg = format!("HTTP {status}: {}", truncate(&body, 400));
            return if matches!(status.as_u16(), 408 | 425 | 429) || status.is_server_error() {
                Err(ChatError::Transient(msg))
            } else {
                Err(ChatError::Fatal(msg))
            };
        }

        let mut stream = resp.bytes_stream();
        // Keep raw bytes until complete lines are available. Converting each
        // network chunk independently with from_utf8_lossy corrupts Unicode
        // code points when a UTF-8 sequence is split across chunks.
        let mut buf: Vec<u8> = Vec::new();
        let mut acc = StreamAccumulator::default();
        let mut done = false;
        const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;

        'stream: while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| ChatError::Transient(format!("stream read error: {e}")))?;
            buf.extend_from_slice(&chunk);

            while let Some(line) = take_sse_line(&mut buf) {
                match parse_stream_line(&line)? {
                    StreamLine::Ignore => {}
                    StreamLine::Done => {
                        done = true;
                        break 'stream;
                    }
                    StreamLine::Record(record) => acc.push(record, ui),
                }
            }

            if buf.len() > MAX_SSE_LINE_BYTES {
                return Err(ChatError::Transient(format!(
                    "model stream line exceeded {MAX_SSE_LINE_BYTES} bytes"
                )));
            }
        }

        // Be tolerant of a final SSE record without a trailing newline.
        if !done && !buf.is_empty() {
            match parse_stream_line(&buf)? {
                StreamLine::Ignore | StreamLine::Done => {}
                StreamLine::Record(record) => acc.push(record, ui),
            }
        }

        acc.finish()
    }

    /// Retries transient failures with exponential backoff; gives up
    /// immediately on fatal errors so we never spin forever on a bad API key.
    async fn chat_with_retry(
        &self,
        messages: &[Message],
        ui: &UiLogger,
        max_attempts: usize,
        kind: RequestKind,
    ) -> Result<ChatResponse, ChatError> {
        let mut backoff = 1u64;
        let mut last: ChatError = ChatError::Transient("no attempts made".into());
        for attempt in 1..=max_attempts.max(1) {
            ui.send(UiEvent::ReasoningReset);
            let result = tokio::time::timeout(
                Duration::from_secs(self.request_timeout_secs),
                self.chat_stream(messages, ui, kind),
            )
            .await;
            match result {
                Ok(Ok(content)) => return Ok(content),
                Ok(Err(ChatError::Fatal(m))) => return Err(ChatError::Fatal(m)),
                Ok(Err(ChatError::Transient(m))) => {
                    ui.log(format!(
                        "model error (attempt {attempt}/{max_attempts}): {m}; retrying in {backoff}s"
                    ));
                    last = ChatError::Transient(m);
                }
                Err(_) => {
                    let m = format!("model call timed out after {}s", self.request_timeout_secs);
                    ui.log(format!(
                        "model error (attempt {attempt}/{max_attempts}): {m}; retrying in {backoff}s"
                    ));
                    last = ChatError::Transient(m);
                }
            }
            if attempt < max_attempts {
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
            }
        }
        Err(last)
    }
}

#[allow(dead_code)]
fn extract_message_content(message: &serde_json::Value) -> Option<String> {
    match message.get("content") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                } else if let Some(t) = part.as_str() {
                    out.push_str(t);
                }
            }
            Some(out)
        }
        // Some servers put reasoning-only replies here.
        Some(serde_json::Value::Null) | None => message
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

// ============================================================
// Tool Call Extraction
// ============================================================

static TOOL_CALL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap());

type IdentifiedCall = (Option<String>, ToolCall);
type IdentifiedError = (Option<String>, String);

/// Parses assembled native calls, pairing each with its `tool_call_id` so a
/// result (or an error) can be attributed back to it.
fn split_native_calls(calls: &[NativeToolCall]) -> (Vec<IdentifiedCall>, Vec<IdentifiedError>) {
    let mut parsed = Vec::new();
    let mut errors = Vec::new();

    for call in calls {
        let id = Some(call.id.clone());
        let raw = call.function.arguments.trim();
        let arguments = if raw.is_empty() {
            Ok(serde_json::json!({}))
        } else {
            serde_json::from_str::<serde_json::Value>(raw)
        };

        match arguments {
            Ok(arguments) => parsed.push((
                id,
                ToolCall {
                    name: call.function.name.clone(),
                    arguments,
                },
            )),
            Err(e) => errors.push((
                id,
                format!(
                    "invalid arguments for {}: {e}\nRaw: {}",
                    call.function.name,
                    truncate(raw, 400)
                ),
            )),
        }
    }

    (parsed, errors)
}

struct ExtractedCalls {
    calls: Vec<ToolCall>,
    errors: Vec<String>,
}

/// A malformed block no longer discards the well-formed ones.
fn extract_tool_calls(text: &str) -> ExtractedCalls {
    #[derive(Deserialize)]
    struct RawToolCall {
        #[serde(alias = "function name")]
        name: Option<String>,
        #[serde(default)]
        arguments: serde_json::Value,
    }

    let mut calls = Vec::new();
    let mut errors = Vec::new();

    for cap in TOOL_CALL_RE.captures_iter(text) {
        let raw = cap[1].trim();
        match serde_json::from_str::<RawToolCall>(raw) {
            Ok(parsed) => {
                let Some(name) = parsed.name else {
                    errors.push(format!(
                        "tool call is missing 'name' or 'function name'\nRaw: {}",
                        truncate(raw, 400)
                    ));
                    continue;
                };

                calls.push(ToolCall {
                    name,
                    arguments: if parsed.arguments.is_null() {
                        serde_json::json!({})
                    } else {
                        parsed.arguments
                    },
                });
            }
            Err(e) => errors.push(format!(
                "malformed tool call JSON: {e}\nRaw: {}",
                truncate(raw, 400)
            )),
        }
    }

    ExtractedCalls { calls, errors }
}

// ============================================================
// Tool Execution
// ============================================================

async fn execute_tool(
    call: &ToolCall,
    workdir: &Path,
    todo_path: &Path,
    command_timeout_secs: u64,
) -> Result<String> {
    match call.name.as_str() {
        "read_file" => {
            let p = call.arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow!("missing path"))?;
            let full = safe_path(workdir, p)?;
            let content = tokio::fs::read_to_string(&full).await?;
            Ok(truncate(&content, 16000))
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
            Ok(format!(
                "wrote {} ({} bytes)",
                full.display(),
                content.len()
            ))
        }
        "list_dir" => {
            let p = call.arguments["path"].as_str().unwrap_or(".");
            let full = safe_path(workdir, p)?;
            let mut out = String::new();
            let mut entries = tokio::fs::read_dir(&full).await?;
            while let Some(entry) = entries.next_entry().await? {
                let is_dir = entry
                    .file_type()
                    .await
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false);
                out.push_str(&format!(
                    "{}{}\n",
                    entry.file_name().to_string_lossy(),
                    if is_dir { "/" } else { "" }
                ));
            }
            if out.is_empty() {
                out.push_str("(empty directory)\n");
            }
            Ok(truncate(&out, 4000))
        }
        "run_command" => {
            let cmd = call.arguments["command"]
                .as_str()
                .ok_or_else(|| anyhow!("missing command"))?;
            run_shell_command(cmd, workdir, command_timeout_secs).await
        }
        "update_todo" => {
            let content = call.arguments["content"]
                .as_str()
                .ok_or_else(|| anyhow!("missing content"))?;
            tokio::fs::write(todo_path, content).await?;
            Ok(format!("todo list updated:\n{content}"))
        }
        "get_todo" => match tokio::fs::read_to_string(todo_path).await {
            Ok(content) if !content.trim().is_empty() => Ok(content),
            _ => Ok("No todo list yet.".to_string()),
        },
        "search_web" => {
            let query = call.arguments["query"]
                .as_str()
                .ok_or_else(|| anyhow!("missing query"))?;
            search_web(query).await
        }
        // Normally intercepted by the agent loop before dispatch.
        "finish" => {
            let reason = call.arguments["reason"].as_str().unwrap_or("done");
            Ok(format!("finish: {reason}"))
        }
        other => Err(anyhow!("unknown tool {other}")),
    }
}

/// Reads a pipe to EOF while retaining at most `limit` bytes. Continuing to
/// drain after the limit prevents a verbose child from blocking on a full OS
/// pipe without allowing its output to consume unbounded memory.
async fn read_bounded<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
) -> io::Result<(Vec<u8>, bool)> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        let available = limit.saturating_sub(retained.len());
        let keep = count.min(available);
        retained.extend_from_slice(&buffer[..keep]);
        truncated |= keep < count;
    }
    Ok((retained, truncated))
}

/// Runs a shell command with a hard timeout, no stdin, and bounded output.
///
/// Note: only the direct child is killed; a command that daemonises
/// grandchildren may leave them running. Use a container if that matters.
async fn run_shell_command(cmd: &str, workdir: &Path, timeout_secs: u64) -> Result<String> {
    #[cfg(windows)]
    let (shell, command_flag) = ("cmd", "/C");
    #[cfg(not(windows))]
    let (shell, command_flag) = ("sh", "-c");

    let mut child = Command::new(shell)
        .arg(command_flag)
        .arg(cmd)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to spawn shell for command: {cmd}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;

    const MAX_CAPTURE_BYTES: usize = 64 * 1024;
    let execution = async {
        tokio::try_join!(
            child.wait(),
            read_bounded(stdout, MAX_CAPTURE_BYTES),
            read_bounded(stderr, MAX_CAPTURE_BYTES)
        )
    };
    let (status, (stdout, stdout_limited), (stderr, stderr_limited)) =
        match tokio::time::timeout(Duration::from_secs(timeout_secs.max(1)), execution).await {
            Ok(result) => result?,
            Err(_) => {
                // kill_on_drop is the backstop, but explicitly killing and
                // reaping avoids leaving the direct child as a zombie.
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Ok(format!(
                    "ERROR: command timed out after {}s and was killed:\n{cmd}",
                    timeout_secs.max(1)
                ));
            }
        };

    let mut result = String::new();
    if !stdout.is_empty() {
        let label = if stdout_limited {
            "stdout (capture limited)"
        } else {
            "stdout"
        };
        result.push_str(&format!(
            "{label}:\n{}\n",
            truncate(&String::from_utf8_lossy(&stdout), 3000)
        ));
    }
    if !stderr.is_empty() {
        let label = if stderr_limited {
            "stderr (capture limited)"
        } else {
            "stderr"
        };
        result.push_str(&format!(
            "{label}:\n{}\n",
            truncate(&String::from_utf8_lossy(&stderr), 3000)
        ));
    }
    match status.code() {
        Some(code) => result.push_str(&format!("exit code: {code}")),
        None => result.push_str("exit code: terminated by signal"),
    }
    Ok(result)
}

/// Resolves `p` relative to `workdir`, rejecting absolute paths, escaping
/// `..`, and — crucially — symlinked ancestors for paths that do not yet
/// exist (the classic write-through-a-symlink escape).
fn safe_path(workdir: &Path, p: &str) -> Result<PathBuf> {
    let path = Path::new(p);
    if path.is_absolute() {
        return Err(anyhow!("absolute paths not allowed"));
    }

    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(os) => relative.push(os),
            Component::CurDir => {}
            Component::ParentDir => {
                if !relative.pop() {
                    return Err(anyhow!("path escapes workdir"));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(anyhow!("invalid path component"));
            }
        }
    }

    let canonical_workdir = workdir
        .canonicalize()
        .with_context(|| format!("workdir {} is not accessible", workdir.display()))?;
    let full = canonical_workdir.join(&relative);

    // Walk up to the nearest existing ancestor and canonicalize *that*.
    let mut existing = full.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        match existing.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                if !existing.pop() {
                    break;
                }
            }
            None => break,
        }
    }

    let canonical_existing = existing
        .canonicalize()
        .with_context(|| format!("cannot resolve {}", existing.display()))?;
    if !canonical_existing.starts_with(&canonical_workdir) {
        return Err(anyhow!("path escapes workdir (symlink or traversal)"));
    }

    let mut resolved = canonical_existing;
    for name in tail.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() > max_chars {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push_str("\n...[truncated]");
        out
    } else {
        s.to_string()
    }
}

fn truncate_display(s: &str, max_chars: usize) -> String {
    let single_line = s.replace(['\n', '\r'], " ");
    if single_line.chars().count() > max_chars {
        let mut t: String = single_line
            .chars()
            .take(max_chars.saturating_sub(1))
            .collect();
        t.push('…');
        t
    } else {
        single_line
    }
}

// ============================================================
// Web Search
// ============================================================

static HTTP: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(10))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .build()
        .expect("failed to build HTTP client")
});

static RESULT_A_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<a([^>]*class="[^"]*result__a[^"]*"[^>]*)>(.*?)</a>"#).unwrap());
static SNIPPET_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>"#).unwrap()
});
static HREF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"href="([^"]*)""#).unwrap());
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<[^>]*>").unwrap());

async fn search_web(query: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://html.duckduckgo.com/html/")?;
    url.query_pairs_mut().append_pair("q", query);

    let resp = HTTP.get(url).send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Ok(format!("search failed: HTTP {status}"));
    }

    let snippets: Vec<String> = SNIPPET_RE
        .captures_iter(&body)
        .map(|c| strip_html(&c[1]))
        .collect();

    let mut results = String::new();
    let mut count = 0usize;
    for cap in RESULT_A_RE.captures_iter(&body) {
        if count >= 5 {
            break;
        }
        let attrs = &cap[1];
        let href = match HREF_RE.captures(attrs) {
            Some(h) => resolve_ddg_url(&decode_entities(&h[1])),
            None => continue,
        };
        let title = strip_html(&cap[2]);
        let snippet = snippets.get(count).cloned().unwrap_or_default();
        results.push_str(&format!(
            "{}. {}\nURL: {}\n{}\n\n",
            count + 1,
            title,
            href,
            snippet
        ));
        count += 1;
    }

    if results.is_empty() {
        return Ok(
            "No search results found (DuckDuckGo may have changed its HTML or blocked the request)."
                .to_string(),
        );
    }
    Ok(truncate(&results, 4000))
}

/// DuckDuckGo wraps outbound links in `//duckduckgo.com/l/?uddg=<encoded>`.
fn resolve_ddg_url(raw: &str) -> String {
    let candidate = if raw.starts_with("//") {
        format!("https:{raw}")
    } else {
        raw.to_string()
    };
    if let Ok(url) = reqwest::Url::parse(&candidate) {
        if url.path().starts_with("/l/") {
            if let Some((_, value)) = url.query_pairs().find(|(k, _)| k == "uddg") {
                return value.into_owned();
            }
        }
        return url.to_string();
    }
    candidate
}

fn strip_html(s: &str) -> String {
    decode_entities(&TAG_RE.replace_all(s, ""))
        .trim()
        .to_string()
}

fn decode_entities(s: &str) -> String {
    // &amp; must be last so "&amp;lt;" does not become "<".
    s.replace("&nbsp;", " ")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

// ============================================================
// Session Management
// ============================================================

#[derive(Debug, Serialize, Deserialize)]
struct SessionInfo {
    id: String,
    goal: String,
    summary: String,
    created: u64,
    last_modified: u64,
    #[serde(default)]
    workdir: Option<String>,
}

struct Session {
    id: String,
    goal: String,
    scratchpad: String,
    messages: Vec<Message>,
    transcript_path: PathBuf,
    todo_path: PathBuf,
    workdir: PathBuf,
    context_tokens: usize,
    compaction_threshold: usize,
    todo_cache: String,
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
        file.flush().await?;
        Ok(())
    }

    async fn push_message(&mut self, message: Message) -> Result<()> {
        self.append_event(&Event::Message(message.clone())).await?;
        self.messages.push(message);
        Ok(())
    }

    async fn append_compaction(&self, summary: &str, retained: &[Message]) -> Result<()> {
        self.append_event(&Event::Compaction {
            summary: summary.to_string(),
            retained: Some(retained.to_vec()),
        })
        .await
    }

    async fn refresh_todo(&mut self) -> String {
        self.todo_cache = tokio::fs::read_to_string(&self.todo_path)
            .await
            .unwrap_or_default();
        self.todo_cache.clone()
    }

    /// Estimates the size of the *entire* request we will send, including the
    /// system prompt, goal, todo list, roles and protocol overhead.
    fn estimate_tokens(&self) -> usize {
        estimate_tokens_for(
            &self.messages,
            &self.scratchpad,
            &self.goal,
            &self.todo_cache,
        )
    }
}

/// Static text in `system_prompt` is roughly this many characters.
const SYSTEM_PROMPT_OVERHEAD_CHARS: usize = 900;
/// Rough per-message wire overhead (role, JSON punctuation, chat template).
const PER_MESSAGE_OVERHEAD_CHARS: usize = 16;

fn estimate_tokens_for(messages: &[Message], scratchpad: &str, goal: &str, todo: &str) -> usize {
    let mut chars = SYSTEM_PROMPT_OVERHEAD_CHARS
        + scratchpad.chars().count()
        + goal.chars().count()
        + todo.chars().count();
    for message in messages {
        chars += message.content.chars().count()
            + message.role.chars().count()
            + PER_MESSAGE_OVERHEAD_CHARS;
    }
    chars / 3
}

fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 128 {
        return Err(anyhow!("session id must be 1-128 characters"));
    }
    if id == "." || id == ".." {
        return Err(anyhow!("invalid session id"));
    }
    if id.chars().any(|c| {
        c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
    }) {
        return Err(anyhow!("session id contains illegal characters"));
    }
    let mut components = Path::new(id).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(anyhow!("session id must be a single path component")),
    }
}

fn sessions_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".night_agent")
        .join("sessions")
}

fn session_dir(id: &str) -> PathBuf {
    // Callers must have validated the id already; belt and braces here.
    sessions_root().join(id)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn read_session_info(id: &str) -> Option<SessionInfo> {
    let path = session_dir(id).join("session.json");
    let text = tokio::fs::read_to_string(&path).await.ok()?;
    serde_json::from_str::<SessionInfo>(&text).ok()
}

/// Never fails on a corrupt `session.json`; it just rebuilds it.
async fn update_session_info(
    session_id: &str,
    goal: Option<&str>,
    summary: Option<&str>,
    workdir: Option<&Path>,
) -> Result<()> {
    let dir = session_dir(session_id);
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join("session.json");
    let now = now_secs();

    let mut info = match tokio::fs::read_to_string(&path).await {
        Ok(s) => serde_json::from_str::<SessionInfo>(&s).unwrap_or(SessionInfo {
            id: session_id.to_string(),
            goal: goal.unwrap_or("Unknown").to_string(),
            summary: summary.unwrap_or("").to_string(),
            created: now,
            last_modified: now,
            workdir: workdir.map(|w| w.display().to_string()),
        }),
        Err(_) => SessionInfo {
            id: session_id.to_string(),
            goal: goal.unwrap_or("Unknown").to_string(),
            summary: summary.unwrap_or("").to_string(),
            created: now,
            last_modified: now,
            workdir: workdir.map(|w| w.display().to_string()),
        },
    };

    if let Some(g) = goal {
        info.goal = g.to_string();
    }
    if let Some(s) = summary {
        info.summary = truncate_display(s, 200);
    }
    if info.summary.is_empty() {
        info.summary = truncate_display(&info.goal, 120);
    }
    if let Some(w) = workdir {
        info.workdir = Some(w.display().to_string());
    }
    info.last_modified = now;

    tokio::fs::write(&path, serde_json::to_string_pretty(&info)?).await?;
    Ok(())
}

async fn create_session(
    id: &str,
    goal: &str,
    workdir: PathBuf,
    context_tokens: usize,
    compaction_threshold: usize,
) -> Result<Session> {
    validate_session_id(id)?;
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

    update_session_info(id, Some(goal), None, Some(&workdir)).await?;

    Ok(Session {
        id: id.to_string(),
        goal: goal.trim().to_string(),
        scratchpad: String::new(),
        messages: Vec::new(),
        transcript_path,
        todo_path,
        workdir,
        context_tokens,
        compaction_threshold,
        todo_cache: String::new(),
    })
}

/// Loads a session. Malformed transcript lines are skipped with a warning
/// rather than destroying the whole session; real IO errors propagate.
async fn load_session(
    id: &str,
    fallback_workdir: PathBuf,
    context_tokens: usize,
    compaction_threshold: usize,
    ui: Option<&UiLogger>,
) -> Result<Session> {
    validate_session_id(id)?;
    let dir = session_dir(id);
    if !dir.exists() {
        return Err(anyhow!("session {id} does not exist"));
    }

    let transcript_path = dir.join("transcript.jsonl");
    let todo_path = dir.join("todo.md");

    let goal = tokio::fs::read_to_string(dir.join("goal.txt"))
        .await
        .with_context(|| format!("failed to read goal for session {id}"))?
        .trim()
        .to_string();

    let mut scratchpad = String::new();
    let mut messages: Vec<Message> = Vec::new();
    let mut skipped = 0usize;
    let mut legacy_compactions = 0usize;

    let data = match tokio::fs::read_to_string(&transcript_path).await {
        Ok(d) => d,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).context("failed to read transcript"),
    };

    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Event>(line) {
            Ok(Event::Message(message)) => messages.push(message),
            Ok(Event::Compaction { summary, retained }) => {
                scratchpad = summary;
                match retained {
                    // Replaying compaction properly: history becomes exactly
                    // what was retained at that point in time.
                    Some(retained) => messages = retained,
                    None => legacy_compactions += 1,
                }
            }
            Err(_) => skipped += 1,
        }
    }

    if let Some(ui) = ui {
        if skipped > 0 {
            ui.log(format!(
                "Warning: skipped {skipped} malformed transcript line(s) in session {id}."
            ));
        }
        if legacy_compactions > 0 {
            ui.log(format!(
                "Warning: {legacy_compactions} legacy compaction event(s) without retained history; \
                 context may be larger than expected."
            ));
        }
    }

    let stored = read_session_info(id).await;
    let workdir = stored
        .as_ref()
        .and_then(|i| i.workdir.as_ref())
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .unwrap_or(fallback_workdir);

    update_session_info(id, Some(&goal), None, Some(&workdir)).await?;

    let mut session = Session {
        id: id.to_string(),
        goal,
        scratchpad,
        messages,
        transcript_path,
        todo_path,
        workdir,
        context_tokens,
        compaction_threshold,
        todo_cache: String::new(),
    };
    session.refresh_todo().await;
    Ok(session)
}

async fn get_session_list() -> Result<Vec<SessionInfo>> {
    let root = sessions_root();
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut entries = tokio::fs::read_dir(&root).await?;
    let mut sessions = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        if !entry.file_type().await?.is_dir() {
            continue;
        }
        let path = entry.path();
        let id = entry.file_name().to_string_lossy().to_string();

        let mut info = match tokio::fs::read_to_string(path.join("session.json")).await {
            Ok(text) => serde_json::from_str::<SessionInfo>(&text).ok(),
            Err(_) => None,
        };

        // Fallback for sessions created before session.json existed.
        if info.is_none() {
            let goal = tokio::fs::read_to_string(path.join("goal.txt"))
                .await
                .map(|g| g.trim().to_string())
                .unwrap_or_else(|_| "Unknown goal".to_string());
            let mut modified = 0u64;
            for file_name in ["transcript.jsonl", "goal.txt", "todo.md"] {
                if let Ok(metadata) = tokio::fs::metadata(path.join(file_name)).await {
                    if let Ok(mtime) = metadata.modified() {
                        if let Ok(duration) = mtime.duration_since(UNIX_EPOCH) {
                            modified = modified.max(duration.as_secs());
                        }
                    }
                }
            }
            if modified == 0 {
                if let Ok(metadata) = tokio::fs::metadata(&path).await {
                    if let Ok(mtime) = metadata.modified() {
                        if let Ok(duration) = mtime.duration_since(UNIX_EPOCH) {
                            modified = duration.as_secs();
                        }
                    }
                }
            }
            info = Some(SessionInfo {
                id: id.clone(),
                goal,
                summary: String::new(),
                created: modified,
                last_modified: modified,
                workdir: None,
            });
        }

        if let Some(mut info) = info {
            if info.summary.is_empty() {
                info.summary = truncate_display(&info.goal, 120);
            }
            sessions.push(info);
        }
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

async fn list_sessions() -> Result<()> {
    let sessions = get_session_list().await?;
    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }
    println!("Available sessions (most recent first):");
    for s in sessions {
        println!(
            "  {}  [{}]  {}\n         {}",
            s.id,
            format_age(s.last_modified),
            truncate_display(&s.goal, 100),
            if s.summary.is_empty() {
                "(no summary)".to_string()
            } else {
                truncate_display(&s.summary, 120)
            }
        );
    }
    Ok(())
}

fn format_age(timestamp: u64) -> String {
    if timestamp == 0 {
        return "unknown".to_string();
    }
    let secs = now_secs().saturating_sub(timestamp);
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;
    if days > 0 {
        format!("{days}d ago")
    } else if hours > 0 {
        format!("{hours}h ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        "just now".to_string()
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

// ============================================================
// Context Management
// ============================================================

fn system_prompt(goal: &str, scratchpad: &str, todo: &str, mode: ToolMode) -> String {
    // In native mode the tool list and call syntax come from the `tools`
    // schema. Repeating them here makes the model emit both shapes at once.
    let protocol = match mode {
        ToolMode::Native => String::new(),
        ToolMode::Text => "\nAvailable tools:\n\
- read_file(path)\n\
- write_file(path, content)\n\
- list_dir(path)\n\
- run_command(command)\n\
- update_todo(content)\n\
- get_todo()\n\
- search_web(query)\n\
- finish(reason)\n\
\nAlways output tool calls exactly as:\n\
<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{...}}\n</tool_call>\n"
            .to_string(),
    };
    format!(
        r#"You are an autonomous coding agent.

Goal:
{goal}

Current scratchpad:
{scratchpad}

Current todo list:
{todo}

{protocol}
Rules:
- Never ask for permission.
- Do not stop until the goal is verified, then call finish exactly once.
- Never describe a tool call in prose; issue the call itself.
- If a tool fails, read the error and try another approach.
- You are operating unsupervised.
- Maintain the todo list.
- Update the todo list whenever you start or finish meaningful work.
- When a task is completed, mark it done by changing "- [ ]" to "- [x]" for that item.
- Prefer actually testing changes instead of assuming they work.
- Long-running commands are killed after a timeout; do not start servers in the foreground.
"#
    )
}

const COMPACTION_KEEP: usize = 12;

/// Guards against a model that narrates intent ("I'll finish now") instead of
/// emitting a call, which would otherwise be nudged forever.
const MAX_NO_ACTION_TURNS: usize = 100;

/// Long runs drift away from the todo list, leaving finished work unchecked and
/// new work unrecorded. Re-anchor periodically.
const TODO_REMINDER_EVERY: usize = 20;

const TODO_REMINDER: &str =
    "Reminder: reconcile the todo list with reality now, using update_todo. \
Check off every item you have actually completed, delete items that are stale or no longer \
relevant, and add any new work you have discovered since. Keep exactly one item marked as in \
progress. Then carry on with the goal.";

/// Summarizes and drops old history. Honors `--compaction-threshold`, keeps
/// the previous scratchpad, only drains messages once summarization has
/// succeeded, and never aborts the agent on failure.
async fn maybe_compact(session: &mut Session, model: &Model, ui: &UiLogger, force: bool) {
    let pct = session.compaction_threshold.clamp(10, 95);
    let threshold = session.context_tokens.saturating_mul(pct) / 100;
    let used = session.estimate_tokens();

    if !force && used < threshold {
        return;
    }

    let keep = COMPACTION_KEEP.min(session.messages.len());
    let split = safe_cut(&session.messages, session.messages.len() - keep);
    if split == 0 {
        if force {
            ui.log("Nothing to compact yet (history is shorter than the retention window).");
        }
        return;
    }

    ui.log(format!(
        "Compacting: {used} est. tokens (threshold {threshold}), summarizing {split} message(s)."
    ));

    let old_text = session.messages[..split]
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let combined = if session.scratchpad.trim().is_empty() {
        format!("Goal: {}\n\nConversation:\n{}", session.goal, old_text)
    } else {
        format!(
            "Goal: {}\n\nSummary of everything before this point:\n{}\n\nNew conversation since \
             that summary:\n{}",
            session.goal, session.scratchpad, old_text
        )
    };

    let summary_messages = vec![
        Message {
            role: "system".into(),
            content:
                "You are a summarizer for a long-running autonomous agent. Merge the previous \
                      summary with the new conversation into a single self-contained summary. \
                      Preserve the goal, completed work, important findings, file paths, commands \
                      that worked or failed, the current plan, and next actions. Never drop \
                      information from the previous summary. Be concise."
                    .into(),
            ..Default::default()
        },
        Message {
            role: "user".into(),
            content: format!("Summarize:\n{}", truncate(&combined, 60000)),
            ..Default::default()
        },
    ];

    // Notify UI that model request starts.
    ui.send(UiEvent::ModelRequestStart);
    let result = model
        .chat_with_retry(&summary_messages, ui, 3, RequestKind::PlainText)
        .await;
    ui.send(UiEvent::ModelRequestEnd);

    match result {
        Ok(response) if !response.text.trim().is_empty() => {
            let summary = response.text;
            session.messages.drain(..split);
            session.scratchpad = summary.clone();
            if let Err(e) = session.append_compaction(&summary, &session.messages).await {
                ui.log(format!("Failed to persist compaction event: {e}"));
            }
            if let Err(e) =
                update_session_info(&session.id, Some(&session.goal), Some(&summary), None).await
            {
                ui.log(format!("Failed to update session metadata: {e}"));
            }
            ui.log(format!(
                "Compaction complete: {} est. tokens remaining.",
                session.estimate_tokens()
            ));
        }
        Ok(_) => ui.log("Compaction skipped: summarizer returned an empty summary."),
        Err(e) => ui.log(format!("Compaction failed, keeping full history: {e}")),
    }
}

/// Hard backstop below compaction: drops the oldest messages, reserving
/// headroom for the model's own output.
/// Advances a history cut point past any leading `tool` results so the
/// retained history never begins with a result whose assistant call was
/// dropped. Gateways reject such an orphan with a 400.
fn safe_cut(messages: &[Message], mut split: usize) -> usize {
    while split < messages.len() && messages[split].role == "tool" {
        split += 1;
    }
    split
}

fn trim_history_to_fit(session: &mut Session, ui: &UiLogger) {
    let reserve = (session.context_tokens / 4).clamp(512, 4096);
    let limit = session.context_tokens.saturating_sub(reserve);
    let mut removed = 0usize;
    while session.estimate_tokens() > limit && session.messages.len() > 2 {
        let cut = safe_cut(&session.messages, 1).min(session.messages.len());
        session.messages.drain(..cut);
        removed += cut;
    }
    if removed > 0 {
        ui.log(format!(
            "Trimmed {removed} oldest message(s) to fit the context window \
             (limit {limit} tokens, reserve {reserve})."
        ));
    }
}

// ============================================================
// Auto Context Detection
// ============================================================

async fn detect_context_size(base_url: &str, api_key: Option<&str>) -> Option<usize> {
    let base = base_url.trim_end_matches('/');
    let mut endpoints = vec![format!("{base}/props"), format!("{base}/models")];
    if !base.ends_with("/v1") {
        endpoints.push(format!("{base}/v1/models"));
    }

    for url in endpoints {
        let mut req = HTTP.get(&url);
        if let Some(key) = api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        let response = tokio::time::timeout(Duration::from_secs(3), req.send()).await;
        if let Ok(Ok(resp)) = response {
            if !resp.status().is_success() {
                continue;
            }
            if let Ok(Ok(value)) =
                tokio::time::timeout(Duration::from_secs(3), resp.json::<serde_json::Value>()).await
            {
                if let Some(ctx) = extract_context_from_json(&value) {
                    if ctx >= 1024 {
                        return Some(ctx);
                    }
                }
            }
        }
    }
    None
}

fn extract_context_from_json(value: &serde_json::Value) -> Option<usize> {
    match value {
        serde_json::Value::Object(map) => {
            for key in [
                "n_ctx",
                "context_length",
                "max_context_length",
                "max_seq_len",
                "max_position_embeddings",
                "n_positions",
            ] {
                if let Some(v) = map.get(key).and_then(|v| v.as_u64()) {
                    return Some(v as usize);
                }
            }
            if let Some(settings) = map.get("default_generation_settings") {
                if let Some(v) = extract_context_from_json(settings) {
                    return Some(v);
                }
            }
            if let Some(data) = map.get("data") {
                if let Some(arr) = data.as_array() {
                    for item in arr {
                        if let Some(v) = extract_context_from_json(item) {
                            return Some(v);
                        }
                    }
                } else if let Some(v) = extract_context_from_json(data) {
                    return Some(v);
                }
            }
            for (_k, v) in map.iter() {
                if let Some(v) = extract_context_from_json(v) {
                    return Some(v);
                }
            }
            None
        }
        _ => None,
    }
}

// ============================================================
// Auto-generate initial todo from goal
// ============================================================

async fn generate_initial_todo(model: &Model, goal: &str, ui: &UiLogger) -> Result<String> {
    let messages = vec![
        Message {
            role: "system".into(),
            content: "You are a planning assistant. Create a concise markdown todo list for the \
                      goal. Break it into small manageable tasks. Use '- [ ]' checkboxes. Do not \
                      include anything except the markdown list."
                .into(),
            ..Default::default()
        },
        Message {
            role: "user".into(),
            content: format!("Goal: {goal}"),
            ..Default::default()
        },
    ];
    ui.send(UiEvent::ModelRequestStart);
    let result = model
        .chat_with_retry(&messages, ui, 3, RequestKind::PlainText)
        .await;
    ui.send(UiEvent::ModelRequestEnd);
    result
        .map(|response| response.text)
        .map_err(|e| anyhow!("{e}"))
}

// ============================================================
// Agent Loop
// ============================================================

#[derive(Clone)]
struct RunConfig {
    max_iterations: usize,
    max_wall_secs: u64,
    command_timeout_secs: u64,
}

struct AgentState {
    iterations: usize,
    no_action_streak: usize,
    paused: bool,
    finished: bool,
    limit_reached: bool,
    start: Instant,
    deadline: Instant,
}

enum CmdOutcome {
    Continue,
    Quit,
}

fn send_status(ui: &UiLogger, session: &Session, st: &AgentState) {
    ui.send(UiEvent::Status {
        iteration: st.iterations,
        tokens: session.estimate_tokens(),
        context_tokens: session.context_tokens,
        goal: session.goal.clone(),
        todo: session.todo_cache.clone(),
        elapsed: st.start.elapsed(),
    });
}

async fn handle_command(
    command: AgentCommand,
    session: &mut Session,
    model: &Model,
    ui: &UiLogger,
    config: &RunConfig,
    st: &mut AgentState,
) -> Result<CmdOutcome> {
    match command {
        AgentCommand::Pause => {
            st.paused = true;
            ui.log("Paused by user.");
            ui.send(UiEvent::Running(false));
        }
        AgentCommand::Resume => {
            st.paused = false;
            if st.limit_reached {
                st.limit_reached = false;
                st.iterations = 0;
                st.start = Instant::now();
                st.deadline = st.start + Duration::from_secs(config.max_wall_secs);
                ui.log("Iteration and wall-clock limits reset; resuming.");
            } else {
                ui.log("Resumed by user.");
            }
            st.finished = false;
            ui.send(UiEvent::Running(true));
        }
        AgentCommand::UpdateGoal(new_goal) => {
            let new_goal = new_goal.trim().to_string();
            if new_goal.is_empty() {
                ui.log("Ignored empty goal.");
                return Ok(CmdOutcome::Continue);
            }
            session.goal = new_goal.clone();
            let dir = session_dir(&session.id);
            if let Err(e) = tokio::fs::write(dir.join("goal.txt"), &new_goal).await {
                ui.log(format!("Failed to save new goal: {e}"));
            }
            if let Err(e) = update_session_info(&session.id, Some(&new_goal), None, None).await {
                ui.log(format!("Failed to update session metadata: {e}"));
            }
            session
                .push_message(Message {
                    role: "user".into(),
                    content: format!("The user changed the goal to: {new_goal}"),
                    ..Default::default()
                })
                .await?;
            ui.log(format!("Goal updated to: {new_goal}"));
            st.finished = false;
            ui.send(UiEvent::Running(!st.paused));
        }
        AgentCommand::AddInstruction(instruction) => {
            let instruction = instruction.trim().to_string();
            if instruction.is_empty() {
                ui.log("Ignored empty instruction.");
                return Ok(CmdOutcome::Continue);
            }
            session
                .push_message(Message::new(
                    "user",
                    format!(
                        "New instruction from the user: {instruction}\n\n\
                         If this asks for work that is not already tracked, call update_todo \
                         first and add it as a new item (for example \"add feature A\" becomes \
                         \"- [ ] Add feature A\"), then start on it. If it only changes how \
                         existing work should be done, revise the affected item instead."
                    ),
                ))
                .await?;
            ui.log(format!("Instruction added: {instruction}"));
            st.finished = false;
            ui.send(UiEvent::Running(!st.paused));
        }
        AgentCommand::UpdateTodo(content) => {
            if let Err(e) = tokio::fs::write(&session.todo_path, &content).await {
                ui.log(format!("Failed to save todo: {e}"));
            } else {
                ui.log("Todo updated by user.");
            }
            session.refresh_todo().await;
            // Keep this short: the full todo is already in the system prompt.
            session
                .push_message(Message {
                    role: "user".into(),
                    content: "The user edited the todo list. The updated list is in your system \
                              prompt; call get_todo() if you need it again."
                        .into(),
                    ..Default::default()
                })
                .await?;
            st.finished = false;
            ui.send(UiEvent::Running(!st.paused));
            // Immediately notify the UI about the new todo content.
            send_status(ui, session, st);
        }
        AgentCommand::CompactNow => {
            ui.log("Manual compaction requested.");
            maybe_compact(session, model, ui, true).await;
            trim_history_to_fit(session, ui);
            send_status(ui, session, st);
        }
        AgentCommand::SwitchSession(new_id) => {
            if let Err(e) = validate_session_id(&new_id) {
                ui.log(format!("Refusing to switch: {e}"));
                return Ok(CmdOutcome::Continue);
            }
            ui.log(format!("Switching to session: {new_id}"));
            let _ = update_session_info(
                &session.id,
                Some(&session.goal),
                None,
                Some(&session.workdir),
            )
            .await;

            match load_session(
                &new_id,
                session.workdir.clone(),
                session.context_tokens,
                session.compaction_threshold,
                Some(ui),
            )
            .await
            {
                Ok(mut new_session) => {
                    new_session.refresh_todo().await;
                    trim_history_to_fit(&mut new_session, ui);
                    *session = new_session;
                    st.iterations = 0;
                    st.no_action_streak = 0;
                    st.paused = false;
                    st.finished = false;
                    st.limit_reached = false;
                    st.start = Instant::now();
                    st.deadline = st.start + Duration::from_secs(config.max_wall_secs);
                    ui.log(format!(
                        "Session {} loaded (workdir {}).",
                        session.id,
                        session.workdir.display()
                    ));
                    send_status(ui, session, st);
                    ui.send(UiEvent::Running(true));
                }
                Err(e) => ui.log(format!("Failed to load session {new_id}: {e}")),
            }
        }
        AgentCommand::Quit => {
            let _ = update_session_info(
                &session.id,
                Some(&session.goal),
                None,
                Some(&session.workdir),
            )
            .await;
            ui.log("Quit command received, stopping agent.");
            return Ok(CmdOutcome::Quit);
        }
    }
    Ok(CmdOutcome::Continue)
}

async fn run_agent(
    config: &RunConfig,
    model: &Model,
    session: &mut Session,
    ui: UiLogger,
    mut rx_cmd: mpsc::UnboundedReceiver<AgentCommand>,
    interactive: bool,
) -> Result<()> {
    let start = Instant::now();
    let mut st = AgentState {
        iterations: 0,
        no_action_streak: 0,
        paused: false,
        finished: false,
        limit_reached: false,
        start,
        deadline: start + Duration::from_secs(config.max_wall_secs),
    };
    let mut pending: VecDeque<AgentCommand> = VecDeque::new();

    ui.log(format!("Starting agent with goal: {}", session.goal));
    ui.log(format!("Session: {}", session.id));
    ui.log(format!("Workdir: {}", session.workdir.display()));
    ui.log(format!(
        "Context limit: {} tokens (compaction at {}%)",
        session.context_tokens,
        session.compaction_threshold.clamp(10, 95)
    ));
    ui.send(UiEvent::Running(true));

    session.refresh_todo().await;
    // Skip generating initial todo if the goal is empty.
    if session.todo_cache.trim().is_empty() && !session.goal.trim().is_empty() {
        ui.log("Generating initial todo list from goal...");
        match generate_initial_todo(model, &session.goal, &ui).await {
            Ok(todo) => {
                if let Err(e) = tokio::fs::write(&session.todo_path, &todo).await {
                    ui.log(format!("Failed to write initial todo: {e}"));
                } else {
                    session.refresh_todo().await;
                    ui.log(format!("Initial todo:\n{todo}"));
                }
            }
            Err(e) => ui.log(format!("Failed to generate initial todo: {e}")),
        }
    }
    send_status(&ui, session, &st);

    loop {
        // ---- Drain pending + queued commands ----
        loop {
            let next = match pending.pop_front() {
                Some(c) => Some(c),
                None => rx_cmd.try_recv().ok(),
            };
            let Some(command) = next else { break };
            match handle_command(command, session, model, &ui, config, &mut st).await? {
                CmdOutcome::Quit => {
                    ui.send(UiEvent::Quit);
                    return Ok(());
                }
                CmdOutcome::Continue => {}
            }
        }

        // ---- Non-interactive runs terminate as soon as work is done ----
        if st.finished && !interactive {
            return Ok(());
        }

        // ---- Idle: block on the command channel instead of busy-waiting ----
        if st.paused || st.finished {
            tokio::select! {
                command = rx_cmd.recv() => match command {
                    Some(c) => pending.push_back(c),
                    // UI is gone; nothing can ever un-pause us.
                    None => return Ok(()),
                },
                _ = tokio::time::sleep(Duration::from_millis(250)) => {}
            }
            continue;
        }

        // ---- Limits ----
        if st.iterations >= config.max_iterations || Instant::now() >= st.deadline {
            if st.iterations >= config.max_iterations {
                ui.log("Hit max iterations.");
            } else {
                ui.log("Hit wall-clock limit.");
            }
            st.limit_reached = true;
            st.finished = true;
            ui.send(UiEvent::AgentFinished {
                reason: "limits reached (press r to resume with fresh limits)".into(),
            });
            ui.send(UiEvent::Running(false));
            continue;
        }

        st.iterations += 1;

        // Safe to append here: every tool result from the previous turn has
        // already been pushed, so this cannot split a call from its result.
        if st.iterations % TODO_REMINDER_EVERY == 0 {
            ui.log("Reminding the model to reconcile the todo list.");
            session
                .push_message(Message::new("user", TODO_REMINDER))
                .await?;
        }

        maybe_compact(session, model, &ui, false).await;
        trim_history_to_fit(session, &ui);
        session.refresh_todo().await;

        let mut messages = vec![Message {
            role: "system".into(),
            content: system_prompt(
                &session.goal,
                &session.scratchpad,
                &session.todo_cache,
                model.tool_mode,
            ),
            ..Default::default()
        }];
        let mut history = session.messages.clone();
        if history
            .last()
            .map(|m| m.role == "assistant")
            .unwrap_or(false)
        {
            history.push(Message {
                role: "user".into(),
                content: "Continue working on the goal.".into(),
                ..Default::default()
            });
        }
        messages.extend(history);

        // ---- Model call, interruptible by user commands ----
        let call = model.chat_with_retry(&messages, &ui, 6, RequestKind::Agent);
        tokio::pin!(call);
        ui.send(UiEvent::ModelRequestStart);
        let response = tokio::select! {
            result = &mut call => {
                // Model call completed (or failed).
                ui.send(UiEvent::ModelRequestEnd);
                result
            }
            command = rx_cmd.recv() => {
                // Interrupted by user command.
                ui.send(UiEvent::ModelRequestEnd);
                match command {
                    Some(c) => {
                        ui.log("Interrupting in-flight model call to handle a user command.");
                        pending.push_back(c);
                        continue;
                    }
                    None => return Ok(()),
                }
            }
        };

        let response = match response {
            Ok(response) => response,
            Err(ChatError::Fatal(error)) => {
                ui.log(format!("Fatal model error: {error}"));
                ui.send(UiEvent::AgentError { error });
                st.finished = true;
                ui.send(UiEvent::Running(false));
                continue;
            }
            Err(ChatError::Transient(error)) => {
                ui.log(format!("Model unavailable: {error}. Backing off 15s."));
                tokio::time::sleep(Duration::from_secs(15)).await;
                continue;
            }
        };

        ui.send(UiEvent::Reasoning {
            content: response.text.clone(),
        });

        if matches!(response.finish_reason.as_deref(), Some("length")) {
            ui.log("Model output was cut off by the token limit; asking it to be brief.");
        }

        let native = !response.tool_calls.is_empty();

        session
            .push_message(Message {
                role: "assistant".into(),
                content: response.text.clone(),
                tool_calls: response.tool_calls.clone(),
                tool_call_id: None,
            })
            .await?;

        // Native calls arrive already delimited, so only text mode needs the
        // regex — which is what used to pick up drafts out of reasoning text.
        let (calls, errors) = if native {
            split_native_calls(&response.tool_calls)
        } else {
            let ExtractedCalls { calls, errors } = extract_tool_calls(&response.text);
            (
                calls.into_iter().map(|call| (None, call)).collect(),
                errors.into_iter().map(|error| (None, error)).collect(),
            )
        };

        // Report malformed calls, but still run the valid ones. A native call
        // must still receive a matching tool result or the next request is
        // invalid, so the error is delivered as that result.
        for (id, error) in &errors {
            ui.log(format!("Malformed tool call: {error}"));
            let message = match id {
                Some(id) => Message::tool_result(id, format!("ERROR: {error}")),
                None => Message::new(
                    "user",
                    format!(
                        "One of your tool calls was malformed and was ignored: {error}\n\
                         Please output valid JSON inside <tool_call> tags."
                    ),
                ),
            };
            session.push_message(message).await?;
        }

        if calls.is_empty() {
            st.no_action_streak += 1;
            if st.no_action_streak >= MAX_NO_ACTION_TURNS {
                let error = format!(
                    "Stopped after {} consecutive turns without a tool call. The model \
                     described what it would do instead of calling a tool. If this model \
                     uses structured function calling, run with --tool-mode native.",
                    st.no_action_streak
                );
                ui.log(error.clone());
                ui.send(UiEvent::AgentError { error });
                ui.send(UiEvent::Running(false));
                st.finished = true;
            } else if errors.is_empty() {
                let nudge = "Continue working autonomously. Output a tool call next.";
                ui.log(format!("No tool call, nudging: {nudge}"));
                session
                    .push_message(Message {
                        role: "user".into(),
                        content: nudge.into(),
                        ..Default::default()
                    })
                    .await?;
            }
        } else {
            st.no_action_streak = 0;
            for (id, call) in calls {
                if call.name == "finish" {
                    let reason = call.arguments["reason"]
                        .as_str()
                        .unwrap_or("done")
                        .to_string();
                    // Close out the call before stopping; a resumed session
                    // would otherwise replay an unanswered tool call.
                    if let Some(id) = &id {
                        session
                            .push_message(Message::tool_result(id, "finished"))
                            .await?;
                    }
                    ui.log(format!("Agent finished: {reason}"));
                    ui.log(format!("Iterations: {}", st.iterations));
                    ui.send(UiEvent::AgentFinished {
                        reason: reason.clone(),
                    });
                    ui.send(UiEvent::Running(false));
                    st.finished = true;
                    break;
                }

                ui.log(format!("Executing tool: {}", call.name));
                let result = execute_tool(
                    &call,
                    &session.workdir,
                    &session.todo_path,
                    config.command_timeout_secs,
                )
                .await
                .unwrap_or_else(|e| format!("ERROR: {e}"));
                ui.log(format!("Result: {}", truncate_display(&result, 200)));

                if matches!(call.name.as_str(), "update_todo") {
                    session.refresh_todo().await;
                }

                let trimmed = truncate(&result, 4000);
                let message = match &id {
                    Some(id) => Message::tool_result(id, trimmed),
                    None => Message::new(
                        "user",
                        format!("Tool result for {}: {}", call.name, trimmed),
                    ),
                };
                session.push_message(message).await?;
            }
        }

        session.refresh_todo().await;
        send_status(&ui, session, &st);
    }
}

// ============================================================
// Text measurement / wrapping (unicode-width aware)
// ============================================================

fn str_width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

fn char_width(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

fn hard_split(word: &str, width: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut w = 0usize;
    for ch in word.chars() {
        let cw = char_width(ch).max(1);
        if w + cw > width && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
            w = 0;
        }
        current.push(ch);
        w += cw;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![line.to_string()];
    }
    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w = 0usize;

    for word in line.split(' ') {
        let ww = str_width(word);
        if ww > width {
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
            }
            let mut parts = hard_split(word, width);
            let last = parts.pop().unwrap_or_default();
            rows.extend(parts);
            current_w = str_width(&last);
            current = last;
            continue;
        }
        if current.is_empty() {
            current.push_str(word);
            current_w = ww;
        } else if current_w + 1 + ww <= width {
            current.push(' ');
            current.push_str(word);
            current_w += 1 + ww;
        } else {
            rows.push(std::mem::take(&mut current));
            current.push_str(word);
            current_w = ww;
        }
    }
    if !current.is_empty() || rows.is_empty() {
        rows.push(current);
    }
    rows
}

fn count_wrapped_lines(text: &str, width: usize) -> usize {
    text.split('\n').map(|l| wrap_line(l, width).len()).sum()
}

fn byte_at_width(s: &str, target_col: usize) -> usize {
    let mut w = 0usize;
    for (i, ch) in s.char_indices() {
        if w >= target_col {
            return i;
        }
        w += char_width(ch);
    }
    s.len()
}

fn slice_by_width(s: &str, start_col: usize, width: usize) -> String {
    let start = byte_at_width(s, start_col);
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s[start..].chars() {
        let cw = char_width(ch);
        if w + cw > width {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out
}

/// A visual row of the multi-line editor, as byte offsets into the buffer.
#[derive(Clone, Copy)]
struct EditorRow {
    start: usize,
    end: usize,
}

/// Character-wraps the buffer ourselves so cursor math and rendering agree
/// exactly (we render these rows verbatim, without Paragraph's own wrapping).
fn editor_rows(buffer: &str, width: usize) -> Vec<EditorRow> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut line_start = 0usize;

    loop {
        let rest = &buffer[line_start..];
        let line_len = rest.find('\n').unwrap_or(rest.len());
        let line = &rest[..line_len];

        let mut seg_start = 0usize;
        let mut w = 0usize;
        for (i, ch) in line.char_indices() {
            let cw = char_width(ch).max(1);
            if w + cw > width && i > seg_start {
                rows.push(EditorRow {
                    start: line_start + seg_start,
                    end: line_start + i,
                });
                seg_start = i;
                w = 0;
            }
            w += cw;
        }
        rows.push(EditorRow {
            start: line_start + seg_start,
            end: line_start + line_len,
        });

        if line_start + line_len >= buffer.len() {
            break;
        }
        line_start += line_len + 1;
    }
    rows
}

fn cursor_row_col(buffer: &str, rows: &[EditorRow], cursor: usize) -> (usize, usize) {
    let mut fallback = (0usize, 0usize);
    for (idx, row) in rows.iter().enumerate() {
        if cursor < row.start {
            break;
        }
        if cursor <= row.end {
            // On a soft wrap boundary prefer the start of the next row.
            if cursor == row.end {
                if let Some(next) = rows.get(idx + 1) {
                    if next.start == row.end {
                        continue;
                    }
                }
            }
            return (idx, str_width(&buffer[row.start..cursor]));
        }
        fallback = (idx, str_width(&buffer[row.start..row.end]));
    }
    fallback
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum InputMode {
    EditingGoal,
    AddingInstruction,
    EditingTodo,
    SelectingSession,
}

const MIN_PANE_HEIGHT: u16 = 3;

#[derive(Debug, Clone, Copy)]
struct PaneAreas {
    transcript: Rect,
    model: Rect,
    todo: Rect,
}

#[derive(Debug, Clone, Copy)]
enum ResizeBoundary {
    TranscriptModel,
    ModelTodo,
}

#[derive(Debug, Clone, Copy)]
struct ResizeDrag {
    boundary: ResizeBoundary,
    start_row: u16,
    initial_heights: [u16; 3],
}

struct TuiState {
    status: Option<StatusInfo>,
    transcript: Vec<TranscriptEntry>,
    scroll_offset: usize,
    todo_text: String,
    goal_text: String,
    todo_scroll_offset: usize,
    input_mode: Option<InputMode>,
    input_buffer: String,
    cursor_position: usize,
    running: bool,
    agent_finished: bool,
    /// Indicates whether the agent is currently waiting for a model response.
    model_waiting: bool,
    /// Latest model output snippet for the mini model-output panel.
    model_snippet: String,
    /// Accumulates streamed tokens during a single model call.
    streaming_buffer: String,
    session_list: Vec<SessionInfo>,
    session_selection: usize,
    mouse_selecting: bool,
    selection_start: Option<(u16, u16)>,
    selection_end: Option<(u16, u16)>,
    capture_screen: bool,
    screen_text: Vec<Vec<char>>,
    /// Outer heights of transcript, model-output, and todo panes.
    pane_heights: Option<[u16; 3]>,
    pane_areas: Option<PaneAreas>,
    resize_drag: Option<ResizeDrag>,
    quit: bool,
    started: Instant,
}

impl TuiState {
    fn new() -> Self {
        Self {
            status: None,
            transcript: Vec::new(),
            scroll_offset: 0,
            todo_text: String::new(),
            goal_text: String::new(),
            todo_scroll_offset: 0,
            input_mode: None,
            input_buffer: String::new(),
            cursor_position: 0,
            running: false,
            agent_finished: false,
            model_waiting: false,
            model_snippet: String::new(),
            streaming_buffer: String::new(),
            session_list: Vec::new(),
            session_selection: 0,
            mouse_selecting: false,
            selection_start: None,
            selection_end: None,
            capture_screen: false,
            screen_text: Vec::new(),
            pane_heights: None,
            pane_areas: None,
            resize_drag: None,
            quit: false,
            started: Instant::now(),
        }
    }

    fn input_active(&self) -> bool {
        self.input_mode.is_some()
    }

    fn multiline(&self) -> bool {
        matches!(self.input_mode, Some(InputMode::EditingTodo))
    }

    fn input_label(&self) -> &'static str {
        match self.input_mode {
            Some(InputMode::EditingGoal) => "Goal (Enter: save, Esc: cancel)",
            Some(InputMode::AddingInstruction) => "Instruction (Enter: send, Esc: cancel)",
            Some(InputMode::EditingTodo) => "Todo (Enter: newline, F2: save, Esc: cancel)",
            Some(InputMode::SelectingSession) => "Select Session",
            None => "",
        }
    }

    fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.cursor_position = 0;
        self.todo_scroll_offset = 0;
    }

    fn insert_text(&mut self, text: &str) {
        // Pasting multi-line text into a single-line field must not smuggle
        // newlines into a goal/instruction.
        let text = if self.multiline() {
            text.replace("\r\n", "\n").replace('\r', "\n")
        } else {
            text.replace(['\n', '\r'], " ")
        };
        self.input_buffer.insert_str(self.cursor_position, &text);
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
        self.cursor_position = self.input_buffer[self.cursor_position..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.cursor_position + index)
            .unwrap_or(self.input_buffer.len());
    }

    fn move_home(&mut self) {
        self.cursor_position = 0;
    }

    fn move_end(&mut self) {
        self.cursor_position = self.input_buffer.len();
    }

    fn move_to_line_start(&mut self) {
        self.cursor_position = self.input_buffer[..self.cursor_position]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
    }

    fn move_to_line_end(&mut self) {
        self.cursor_position = self.input_buffer[self.cursor_position..]
            .find('\n')
            .map(|i| self.cursor_position + i)
            .unwrap_or(self.input_buffer.len());
    }

    /// Vertical movement follows *visual* rows, matching what is rendered.
    fn move_cursor_vertical(&mut self, width: usize, delta: isize) {
        let rows = editor_rows(&self.input_buffer, width);
        let (row, col) = cursor_row_col(&self.input_buffer, &rows, self.cursor_position);
        let target = row as isize + delta;
        if target < 0 || target as usize >= rows.len() {
            return;
        }
        let target_row = rows[target as usize];
        let text = &self.input_buffer[target_row.start..target_row.end];
        self.cursor_position = target_row.start + byte_at_width(text, col);
    }

    fn clear_selection(&mut self) {
        self.selection_start = None;
        self.selection_end = None;
        self.mouse_selecting = false;
        self.capture_screen = false;
    }
}

fn push_entry(state: &mut TuiState, kind: EntryKind, text: &str) {
    let normalized = text.replace("\r\n", "\n").replace('\r', "");
    for raw_line in normalized.split('\n') {
        state.transcript.push(TranscriptEntry {
            kind: kind.clone(),
            text: raw_line.replace('\t', "    "),
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
    tokio::task::block_in_place(|| {
        let mut clipboard =
            Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
        clipboard
            .get_text()
            .map_err(|e| anyhow!("failed to read clipboard: {e}"))
    })
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    tokio::task::block_in_place(|| {
        let mut clipboard =
            Clipboard::new().map_err(|e| anyhow!("failed to open clipboard: {e}"))?;
        clipboard
            .set_text(text.to_string())
            .map_err(|e| anyhow!("failed to write clipboard: {e}"))
    })
}

// ============================================================
// Mouse Drag-Selection
// ============================================================

fn normalize_selection(a: (u16, u16), b: (u16, u16)) -> (u16, u16, u16, u16) {
    if (a.1, a.0) <= (b.1, b.0) {
        (a.0, a.1, b.0, b.1)
    } else {
        (b.0, b.1, a.0, a.1)
    }
}

fn is_border_char(c: char) -> bool {
    matches!(
        c,
        '│' | '─'
            | '┌'
            | '┐'
            | '└'
            | '┘'
            | '├'
            | '┤'
            | '┬'
            | '┴'
            | '┼'
            | '╭'
            | '╮'
            | '╰'
            | '╯'
            | '═'
            | '║'
            | '╔'
            | '╗'
            | '╚'
            | '╝'
            | '╠'
            | '╣'
            | '╦'
            | '╩'
            | '╬'
            | '┃'
            | '━'
            | '┏'
            | '┓'
            | '┗'
            | '┛'
            | '┣'
            | '┫'
            | '┳'
            | '┻'
            | '╋'
    )
}

/// Preserves indentation and blank lines; only border glyphs are dropped.
fn extract_selected_text(screen: &[Vec<char>], start: (u16, u16), end: (u16, u16)) -> String {
    let (sx, sy, ex, ey) = normalize_selection(start, end);
    let (sx, sy, ex) = (sx as usize, sy as usize, ex as usize);
    if sy >= screen.len() {
        return String::new();
    }
    let ey = (ey as usize).min(screen.len().saturating_sub(1));

    let mut lines = Vec::new();
    for y in sy..=ey {
        let row = &screen[y];
        if row.is_empty() {
            lines.push(String::new());
            continue;
        }
        let row_max = row.len() - 1;
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
        let line: String = row[from..=to]
            .iter()
            .map(|&c| if is_border_char(c) { ' ' } else { c })
            .collect();
        lines.push(line.trim_end().to_string());
    }

    while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    lines.join("\n")
}

fn fit_pane_heights(mut heights: [u16; 3], total: u16) -> [u16; 3] {
    debug_assert!(total >= MIN_PANE_HEIGHT * 3);
    for height in &mut heights {
        *height = (*height).max(MIN_PANE_HEIGHT);
    }

    let current: u16 = heights.iter().copied().sum();
    if current < total {
        // Terminal growth benefits the transcript, which is normally the pane
        // where additional space is most useful.
        heights[0] = heights[0].saturating_add(total - current);
    } else if current > total {
        let mut excess = current - total;
        // Shrink the transcript first, then the todo and model panes, without
        // allowing any pane to become too small to have a bordered interior.
        for index in [0usize, 2, 1] {
            let available = heights[index].saturating_sub(MIN_PANE_HEIGHT);
            let reduction = available.min(excess);
            heights[index] -= reduction;
            excess -= reduction;
            if excess == 0 {
                break;
            }
        }
    }
    heights
}

fn resize_boundary_at(areas: PaneAreas, column: u16, row: u16) -> Option<ResizeBoundary> {
    let left = areas.transcript.x;
    let right = areas
        .transcript
        .x
        .saturating_add(areas.transcript.width.saturating_sub(1));
    if column < left || column > right {
        return None;
    }

    let touches = |upper: Rect, lower: Rect| {
        row == upper.y.saturating_add(upper.height.saturating_sub(1)) || row == lower.y
    };
    if touches(areas.transcript, areas.model) {
        Some(ResizeBoundary::TranscriptModel)
    } else if touches(areas.model, areas.todo) {
        Some(ResizeBoundary::ModelTodo)
    } else {
        None
    }
}

fn resize_panes(state: &mut TuiState, row: u16) {
    let Some(drag) = state.resize_drag else {
        return;
    };
    let delta = row as i32 - drag.start_row as i32;
    let mut heights = drag.initial_heights;
    let (upper, lower) = match drag.boundary {
        ResizeBoundary::TranscriptModel => (0usize, 1usize),
        ResizeBoundary::ModelTodo => (1usize, 2usize),
    };
    let pair_total = heights[upper] as i32 + heights[lower] as i32;
    let upper_height = (heights[upper] as i32 + delta)
        .clamp(MIN_PANE_HEIGHT as i32, pair_total - MIN_PANE_HEIGHT as i32);
    heights[upper] = upper_height as u16;
    heights[lower] = (pair_total - upper_height) as u16;
    state.pane_heights = Some(heights);
}

fn handle_mouse_event(mouse: MouseEvent, state: &mut TuiState) {
    // Pane resizing takes precedence over scrolling, editing, and selection.
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if let (Some(areas), Some(heights)) = (state.pane_areas, state.pane_heights) {
                if let Some(boundary) = resize_boundary_at(areas, mouse.column, mouse.row) {
                    state.clear_selection();
                    state.resize_drag = Some(ResizeDrag {
                        boundary,
                        start_row: mouse.row,
                        initial_heights: heights,
                    });
                    return;
                }
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if state.resize_drag.is_some() => {
            resize_panes(state, mouse.row);
            return;
        }
        MouseEventKind::Up(MouseButton::Left) if state.resize_drag.is_some() => {
            resize_panes(state, mouse.row);
            state.resize_drag = None;
            return;
        }
        _ => {}
    }

    // Wheel scrolling always works, even while editing.
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            state.scroll_offset = state.scroll_offset.saturating_add(3);
            return;
        }
        MouseEventKind::ScrollDown => {
            state.scroll_offset = state.scroll_offset.saturating_sub(3);
            return;
        }
        _ => {}
    }

    if state.input_active() {
        return;
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            state.mouse_selecting = true;
            state.capture_screen = true;
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
                                &format!("Copied {} chars to clipboard.", text.chars().count()),
                            ),
                            Err(e) => {
                                push_entry(state, EntryKind::Error, &format!("Copy failed: {e}"))
                            }
                        }
                    }
                }
                state.clear_selection();
            }
        }
        _ => {}
    }
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
        let editing_todo = matches!(state.input_mode, Some(InputMode::EditingTodo));
        let selecting_session = matches!(state.input_mode, Some(InputMode::SelectingSession));
        let bottom_height: u16 = if selecting_session { 12 } else { 3 };
        let fixed_height = 1u16.saturating_add(bottom_height);
        let minimum_height = fixed_height.saturating_add(MIN_PANE_HEIGHT * 3);
        if area.width < 10 || area.height < minimum_height {
            state.pane_areas = None;
            frame.render_widget(
                Paragraph::new(format!(
                    "Terminal too small (need at least 10x{minimum_height})"
                )),
                area,
            );
            return;
        }

        let todo_content = if editing_todo {
            state.input_buffer.clone()
        } else if state.todo_text.trim().is_empty() {
            "No todo yet.".to_string()
        } else {
            state.todo_text.clone()
        };

        let todo_width = area.width.saturating_sub(2).max(1) as usize;
        let wrapped_lines = count_wrapped_lines(&todo_content, todo_width);

        let pane_total = area.height - fixed_height;
        let preferred_model_height = 6u16
            .min(pane_total.saturating_sub(MIN_PANE_HEIGHT * 2))
            .max(MIN_PANE_HEIGHT);
        let max_todo_height = pane_total
            .saturating_sub(preferred_model_height)
            .saturating_sub(MIN_PANE_HEIGHT);
        let preferred_todo_height =
            (wrapped_lines.saturating_add(2) as u16).clamp(MIN_PANE_HEIGHT, max_todo_height);
        let default_heights = [
            pane_total - preferred_model_height - preferred_todo_height,
            preferred_model_height,
            preferred_todo_height,
        ];
        let pane_heights =
            fit_pane_heights(state.pane_heights.unwrap_or(default_heights), pane_total);
        state.pane_heights = Some(pane_heights);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(pane_heights[0]),
                Constraint::Length(pane_heights[1]),
                Constraint::Length(pane_heights[2]),
                Constraint::Length(1),
                Constraint::Length(bottom_height),
            ])
            .split(area);
        state.pane_areas = Some(PaneAreas {
            transcript: chunks[0],
            model: chunks[1],
            todo: chunks[2],
        });

        // ---- Transcript ----
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
                "Transcript (scrolled {} lines back — End to jump to latest)",
                state.scroll_offset
            )
        } else {
            "Transcript (drag bottom border to resize)".to_string()
        };
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(transcript_title),
            ),
            transcript_area,
        );

        // ---- Current model output snippet ----
        let model_snippet_area = chunks[1];
        let snippet_width = model_snippet_area.width.saturating_sub(2).max(1) as usize;
        let snippet_height = model_snippet_area.height.saturating_sub(2).max(1) as usize;

        let snippet_full_text = if !state.model_snippet.is_empty() {
            state.model_snippet.clone()
        } else if state.model_waiting {
            "⏳ waiting for model response…".to_string()
        } else {
            "No model output yet.".to_string()
        };

        let mut snippet_lines: Vec<Line> = Vec::new();
        for logical_line in snippet_full_text.split('\n') {
            for wrapped_line in wrap_line(logical_line, snippet_width) {
                snippet_lines.push(Line::from(wrapped_line));
            }
        }
        let total_snippet_lines = snippet_lines.len();
        let snippet_scroll = total_snippet_lines.saturating_sub(snippet_height);

        let snippet_paragraph = Paragraph::new(snippet_lines)
            .style(Style::default().fg(Color::Yellow))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Current model output"),
            )
            .scroll((snippet_scroll as u16, 0));

        frame.render_widget(snippet_paragraph, model_snippet_area);

        // ---- Todo panel ----
        let todo_area = chunks[2];
        let inner_w = todo_area.width.saturating_sub(2).max(1) as usize;
        let inner_h = todo_area.height.saturating_sub(2).max(1) as usize;

        if editing_todo {
            let title = "Todo — editing (Enter: newline, F2: save, Esc: cancel)";
            let rows = editor_rows(&state.input_buffer, inner_w);
            let (cursor_row, cursor_col) =
                cursor_row_col(&state.input_buffer, &rows, state.cursor_position);

            if state.todo_scroll_offset > cursor_row {
                state.todo_scroll_offset = cursor_row;
            }
            if cursor_row >= state.todo_scroll_offset + inner_h {
                state.todo_scroll_offset = cursor_row - inner_h + 1;
            }
            let max_scroll = rows.len().saturating_sub(inner_h);
            state.todo_scroll_offset = state.todo_scroll_offset.min(max_scroll);

            let visible_end = (state.todo_scroll_offset + inner_h).min(rows.len());
            let visible: Vec<Line> = rows[state.todo_scroll_offset..visible_end]
                .iter()
                .map(|r| Line::from(state.input_buffer[r.start..r.end].to_string()))
                .collect();

            frame.render_widget(
                Paragraph::new(visible).block(Block::default().borders(Borders::ALL).title(title)),
                todo_area,
            );

            let cursor_x = todo_area.x + 1 + cursor_col.min(inner_w.saturating_sub(1)) as u16;
            let cursor_y =
                todo_area.y + 1 + (cursor_row - state.todo_scroll_offset).min(inner_h - 1) as u16;
            frame.set_cursor(cursor_x, cursor_y);
        } else {
            let mut visible: Vec<Line> = Vec::new();
            for logical in todo_content.split('\n') {
                for row in wrap_line(logical, inner_w) {
                    visible.push(Line::from(row));
                }
            }
            visible.truncate(inner_h);
            frame.render_widget(
                Paragraph::new(visible).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Todo (t to edit)"),
                ),
                todo_area,
            );
            state.todo_scroll_offset = 0;
        }

        // ---- Status line ----
        let agent_state = if state.agent_finished {
            "finished"
        } else if state.running {
            "running"
        } else {
            "paused"
        };

        let spinner: char = if state.model_waiting {
            SPINNER[(state.started.elapsed().as_millis() / 100) as usize % SPINNER.len()]
        } else {
            '●'
        };

        let status_text = match &state.status {
            Some(s) => {
                let pct = if s.context_tokens > 0 {
                    s.tokens * 100 / s.context_tokens
                } else {
                    0
                };
                format!(
                    "{} | Goal: {} | iter {} | ctx {}/{} ({}%) | elapsed {} | agent: {}",
                    spinner,
                    truncate_display(&s.goal, 40),
                    s.iteration,
                    s.tokens,
                    s.context_tokens,
                    pct,
                    format_duration(s.elapsed),
                    agent_state
                )
            }
            None => format!("{} | waiting for status... | agent: {agent_state}", spinner),
        };
        frame.render_widget(
            Paragraph::new(status_text).style(Style::default().fg(Color::DarkGray)),
            chunks[3],
        );

        // ---- Bottom area ----
        let bottom = chunks[4];
        if selecting_session {
            let title = "Select Session (↑↓: navigate, Enter: select, Esc: cancel)";
            let mut list_lines: Vec<Line> = Vec::new();

            if state.session_list.is_empty() {
                list_lines.push(Line::from("No sessions found."));
            } else {
                let rows_available = bottom.height.saturating_sub(2).max(1) as usize;
                let per_item = 2usize;
                let visible_items = (rows_available / per_item).max(1);
                let total = state.session_list.len();
                let max_offset = total.saturating_sub(visible_items);
                let mut start_idx = state.session_selection.saturating_sub(visible_items / 2);
                if start_idx > max_offset {
                    start_idx = max_offset;
                }
                let end_idx = (start_idx + visible_items).min(total);

                for idx in start_idx..end_idx {
                    let s = &state.session_list[idx];
                    let selected = idx == state.session_selection;
                    let style = if selected {
                        Style::default()
                            .bg(Color::White)
                            .fg(Color::Black)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    list_lines.push(Line::from(Span::styled(
                        format!(
                            "{}{}  [{}]  {}",
                            if selected { "> " } else { "  " },
                            s.id,
                            format_age(s.last_modified),
                            truncate_display(&s.goal, 40)
                        ),
                        style,
                    )));
                    list_lines.push(Line::from(Span::styled(
                        format!(
                            "      {}",
                            if s.summary.is_empty() {
                                "(no summary)".to_string()
                            } else {
                                truncate_display(&s.summary, 80)
                            }
                        ),
                        style,
                    )));
                }
            }

            frame.render_widget(
                Paragraph::new(list_lines)
                    .block(Block::default().borders(Borders::ALL).title(title)),
                bottom,
            );
        } else if state.input_active() {
            let inner_w = bottom.width.saturating_sub(2).max(1) as usize;

            // For the multiline todo editor, the small input box below the
            // todo panel is only a single-line view of the *current* line.
            // Previously it rendered from the beginning of input_buffer, so
            // after pressing Enter it continued showing the first/old todo
            // line even though the cursor had moved to the new line.
            let (visible, cursor_col) = if state.multiline() {
                let line_start = state.input_buffer[..state.cursor_position]
                    .rfind('\n')
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let line_end = state.input_buffer[state.cursor_position..]
                    .find('\n')
                    .map(|i| state.cursor_position + i)
                    .unwrap_or(state.input_buffer.len());
                let current_line = &state.input_buffer[line_start..line_end];
                let cursor_col = str_width(&state.input_buffer[line_start..state.cursor_position]);
                let scroll = cursor_col.saturating_sub(inner_w.saturating_sub(1));
                (
                    slice_by_width(current_line, scroll, inner_w),
                    cursor_col - scroll,
                )
            } else {
                let cursor_col = str_width(&state.input_buffer[..state.cursor_position]);
                let scroll = cursor_col.saturating_sub(inner_w.saturating_sub(1));
                (
                    slice_by_width(&state.input_buffer, scroll, inner_w),
                    cursor_col - scroll,
                )
            };

            frame.render_widget(
                Paragraph::new(visible)
                    .style(Style::default().add_modifier(Modifier::BOLD))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(state.input_label()),
                    ),
                bottom,
            );
            let cursor_x = bottom.x + 1 + cursor_col.min(inner_w.saturating_sub(1)) as u16;
            frame.set_cursor(cursor_x, bottom.y + 1);
        } else {
            let controls = "p:pause r:resume i:instruction g:goal t:todo m:compact s:sessions \
                            q:quit  ↑↓/PgUp/PgDn/wheel: scroll  drag: select+copy";
            frame.render_widget(
                Paragraph::new(controls)
                    .block(Block::default().borders(Borders::ALL).title("Controls")),
                bottom,
            );
        }

        // ---- Selection highlight ----
        if let (Some(start), Some(end)) = (state.selection_start, state.selection_end) {
            let (sx, sy, ex, ey) = normalize_selection(start, end);
            let row_max = area.width.saturating_sub(1);
            let sy = sy.min(area.height.saturating_sub(1));
            let ey = ey.min(area.height.saturating_sub(1));
            let highlight = Style::default()
                .bg(Color::White)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD);

            if sy <= ey {
                for y in sy..=ey {
                    let (from, to) = if sy == ey {
                        (sx.min(row_max), ex.min(row_max))
                    } else if y == sy {
                        (sx.min(row_max), row_max)
                    } else if y == ey {
                        (0, ex.min(row_max))
                    } else {
                        (0, row_max)
                    };
                    if from <= to {
                        for x in from..=to {
                            frame.buffer_mut().get_mut(x, y).set_style(highlight);
                        }
                    }
                }
            }
        }

        // ---- Capture the screen only while a drag is in progress ----
        if state.capture_screen {
            let mut screen = vec![vec![' '; area.width as usize]; area.height as usize];
            for y in 0..area.height {
                for x in 0..area.width {
                    let symbol = frame.buffer_mut().get(x, y).symbol();
                    screen[y as usize][x as usize] = symbol.chars().next().unwrap_or(' ');
                }
            }
            state.screen_text = screen;
        }
    })?;
    Ok(())
}

// ============================================================
// Input Handling
// ============================================================

fn start_input(state: &mut TuiState, mode: InputMode, prefill: &str) {
    state.input_mode = Some(mode);
    // If starting a new todo and the current todo is empty, provide a template.
    if mode == InputMode::EditingTodo && prefill.trim().is_empty() {
        state.input_buffer = "- [ ] ".to_string();
        state.cursor_position = state.input_buffer.len();
    } else {
        state.input_buffer = prefill.to_string();
        state.cursor_position = state.input_buffer.len();
    }
    state.todo_scroll_offset = 0;
    state.clear_selection();
}

fn finish_input(state: &mut TuiState, tx_cmd: &mpsc::UnboundedSender<AgentCommand>) {
    let input = state.input_buffer.clone();
    match state.input_mode {
        Some(InputMode::EditingGoal) => {
            if !input.trim().is_empty() {
                let _ = tx_cmd.send(AgentCommand::UpdateGoal(input));
            }
        }
        Some(InputMode::AddingInstruction) => {
            if !input.trim().is_empty() {
                let _ = tx_cmd.send(AgentCommand::AddInstruction(input));
            }
        }
        Some(InputMode::EditingTodo) => {
            state.todo_text = input.clone();
            let _ = tx_cmd.send(AgentCommand::UpdateTodo(input));
        }
        Some(InputMode::SelectingSession) | None => {}
    }
    state.input_mode = None;
    state.clear_input();
}

fn cancel_input(state: &mut TuiState) {
    state.input_mode = None;
    state.clear_input();
    push_entry(state, EntryKind::Log, "Input cancelled.");
}

/// Handles one terminal event. Quit requests set `state.quit` rather than
/// returning, so the caller's single quit check covers every path.
async fn handle_terminal_input(
    input: CEvent,
    state: &mut TuiState,
    tx_cmd: &mpsc::UnboundedSender<AgentCommand>,
    editor_width: usize,
    dirty: &mut bool,
) -> Result<()> {
    match input {
        CEvent::Paste(text) => {
            if state.input_active()
                && !matches!(state.input_mode, Some(InputMode::SelectingSession))
            {
                state.insert_text(&text);
            }
        }
        CEvent::Key(key) => {
            if state.input_active() {
                handle_input_key(key, state, tx_cmd, editor_width);
            } else {
                if key.kind != KeyEventKind::Press {
                    return Ok(());
                }
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                match key.code {
                    KeyCode::Char('c') if ctrl => {
                        let _ = tx_cmd.send(AgentCommand::Quit);
                        state.quit = true;
                    }
                    KeyCode::Char('p') if !ctrl && !shift => {
                        let _ = tx_cmd.send(AgentCommand::Pause);
                    }
                    KeyCode::Char('r') if !ctrl && !shift => {
                        let _ = tx_cmd.send(AgentCommand::Resume);
                    }
                    KeyCode::Char('i') if !ctrl && !shift => {
                        start_input(state, InputMode::AddingInstruction, "");
                    }
                    KeyCode::Char('g') if !ctrl && !shift => {
                        let prefill = state.goal_text.clone();
                        start_input(state, InputMode::EditingGoal, &prefill);
                    }
                    KeyCode::Char('t') if !ctrl && !shift => {
                        let prefill = state.todo_text.clone();
                        start_input(state, InputMode::EditingTodo, &prefill);
                    }
                    KeyCode::Char('m') if !ctrl && !shift => {
                        let _ = tx_cmd.send(AgentCommand::CompactNow);
                    }
                    KeyCode::Char('s') if !ctrl && !shift => {
                        state.session_list = get_session_list().await.unwrap_or_default();
                        state.session_selection = 0;
                        state.input_mode = Some(InputMode::SelectingSession);
                        state.clear_selection();
                    }
                    KeyCode::Char('q') if !ctrl && !shift => {
                        let _ = tx_cmd.send(AgentCommand::Quit);
                        state.quit = true;
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
                    KeyCode::Home => state.scroll_offset = usize::MAX,
                    KeyCode::End => state.scroll_offset = 0,
                    _ => {}
                }
            }
        }
        CEvent::Mouse(mouse_event) => handle_mouse_event(mouse_event, state),
        CEvent::Resize(_, _) => {}
        CEvent::FocusGained | CEvent::FocusLost => *dirty = false,
    }
    Ok(())
}

fn handle_input_key(
    key: KeyEvent,
    state: &mut TuiState,
    tx_cmd: &mpsc::UnboundedSender<AgentCommand>,
    editor_width: usize,
) {
    if key.kind != KeyEventKind::Press {
        return;
    }

    if state.input_mode == Some(InputMode::SelectingSession) {
        match key.code {
            KeyCode::Up => {
                state.session_selection = state.session_selection.saturating_sub(1);
            }
            KeyCode::Down => {
                if !state.session_list.is_empty()
                    && state.session_selection + 1 < state.session_list.len()
                {
                    state.session_selection += 1;
                }
            }
            KeyCode::Enter => {
                if !state.session_list.is_empty() {
                    let selected = state.session_list[state.session_selection].id.clone();
                    let _ = tx_cmd.send(AgentCommand::SwitchSession(selected));
                }
                state.input_mode = None;
                state.clear_input();
            }
            KeyCode::Esc => {
                state.input_mode = None;
                state.clear_input();
            }
            _ => {}
        }
        return;
    }

    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let multiline = state.multiline();

    match key.code {
        KeyCode::Enter if multiline => {
            // Handle empty buffer: insert template directly.
            if state.input_buffer.is_empty() {
                state.insert_text("- [ ] ");
                return;
            }
            let line_start = state.input_buffer[..state.cursor_position]
                .rfind('\n')
                .map(|i| i + 1)
                .unwrap_or(0);
            let line_end = state.input_buffer[state.cursor_position..]
                .find('\n')
                .map(|i| state.cursor_position + i)
                .unwrap_or(state.input_buffer.len());
            let line_text = &state.input_buffer[line_start..line_end];
            let trimmed = line_text.trim_start();
            let is_list_item = trimmed.starts_with("- [ ]")
                || trimmed.starts_with("- [x]")
                || trimmed.starts_with("- [X]");
            let cursor_at_end = state.cursor_position == line_end
                || state.input_buffer[state.cursor_position..line_end]
                    .trim()
                    .is_empty();

            if (is_list_item && cursor_at_end) || (trimmed.is_empty() && cursor_at_end) {
                // If the current line is empty and we are at the end of the buffer,
                // insert "- [ ] " without a leading newline.
                if trimmed.is_empty() && state.cursor_position == state.input_buffer.len() {
                    state.insert_text("- [ ] ");
                } else {
                    state.insert_text("\n- [ ] ");
                }
            } else {
                state.insert_char('\n');
            }
        }
        KeyCode::Enter => finish_input(state, tx_cmd),
        KeyCode::Char('s') if ctrl => finish_input(state, tx_cmd),
        KeyCode::F(2) if multiline => finish_input(state, tx_cmd),
        KeyCode::Esc => cancel_input(state),
        KeyCode::Char('c') if ctrl => cancel_input(state),
        KeyCode::Char('v') | KeyCode::Char('V') if ctrl => match read_clipboard() {
            Ok(text) => state.insert_text(&text),
            Err(error) => push_entry(state, EntryKind::Error, &format!("Paste failed: {error}")),
        },
        KeyCode::Backspace => state.backspace(),
        KeyCode::Delete => state.delete(),
        KeyCode::Left => state.move_left(),
        KeyCode::Right => state.move_right(),
        KeyCode::Up if multiline => state.move_cursor_vertical(editor_width, -1),
        KeyCode::Down if multiline => state.move_cursor_vertical(editor_width, 1),
        KeyCode::Home if multiline => state.move_to_line_start(),
        KeyCode::End if multiline => state.move_to_line_end(),
        KeyCode::Home => state.move_home(),
        KeyCode::End => state.move_end(),
        KeyCode::Tab if multiline => state.insert_text("  "),
        KeyCode::Char(c) if !ctrl => state.insert_char(c),
        _ => {}
    }
}

// ============================================================
// Terminal lifecycle (RAII + panic hook)
// ============================================================

fn restore_terminal() -> Result<()> {
    let mut out = io::stdout();
    let _ = execute!(
        out,
        Show,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
    let _ = out.flush();
    Ok(())
}

struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut out = io::stdout();
        execute!(
            out,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture,
            Hide
        )?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = restore_terminal();
    }
}

fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        default_hook(info);
    }));
}

// ============================================================
// TUI
// ============================================================

async fn run_tui(
    mut rx_ui: mpsc::UnboundedReceiver<UiEvent>,
    tx_cmd: mpsc::UnboundedSender<AgentCommand>,
) -> Result<()> {
    let _guard = TerminalGuard::new()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let (tx_input, mut rx_input) = mpsc::unbounded_channel::<CEvent>();
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = stop.clone();
    let input_thread = std::thread::spawn(move || {
        while !thread_stop.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if tx_input.send(ev).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });

    let mut state = TuiState::new();
    let mut agent_alive = true;
    let mut dirty = true;

    // A streaming model emits one UI event per token. Drawing on every event
    // means one full-screen write per token, and once the terminal stops
    // draining them the write blocks, which stalls this loop and leaves
    // keypresses unread. Coalesce instead and redraw at a fixed ceiling.
    const FRAME: Duration = Duration::from_millis(33);
    const IDLE_REFRESH: Duration = Duration::from_millis(100);
    let mut last_draw = Instant::now()
        .checked_sub(FRAME)
        .unwrap_or_else(Instant::now);

    let result: Result<()> = loop {
        if dirty && last_draw.elapsed() >= FRAME {
            last_draw = Instant::now();
            if let Err(e) = draw_ui(&mut terminal, &mut state) {
                break Err(e);
            }
            if state.input_active()
                && !matches!(state.input_mode, Some(InputMode::SelectingSession))
            {
                let _ = terminal.show_cursor();
            } else {
                let _ = terminal.hide_cursor();
            }
            dirty = false;
        }

        let editor_width = terminal
            .size()
            .map(|r| r.width.saturating_sub(2).max(1) as usize)
            .unwrap_or(80);

        // Wake up exactly when the next frame is allowed, so a pending redraw
        // is never delayed longer than the frame budget.
        let wait = if dirty {
            FRAME.saturating_sub(last_draw.elapsed())
        } else {
            IDLE_REFRESH
        };

        tokio::select! {
            // Input is polled first so a keypress is never starved by the
            // permanently-ready stream of model events.
            biased;

            maybe_input = rx_input.recv() => {
                let Some(input) = maybe_input else { break Ok(()) };
                dirty = true;
                handle_terminal_input(input, &mut state, &tx_cmd, editor_width, &mut dirty)
                    .await?;
                if state.quit_requested() {
                    break Ok(());
                }
            }
            maybe_event = rx_ui.recv(), if agent_alive => {
                match maybe_event {
                    Some(event) => {
                        apply_ui_event(&mut state, event, &mut dirty);
                        while let Ok(event) = rx_ui.try_recv() {
                            apply_ui_event(&mut state, event, &mut dirty);
                        }
                        if state.quit_requested() {
                            break Ok(());
                        }
                    }
                    None => {
                        agent_alive = false;
                        push_entry(&mut state, EntryKind::Error, "Agent task ended unexpectedly.");
                        state.running = false;
                        dirty = true;
                    }
                }
            }
            _ = tokio::time::sleep(wait) => {
                dirty = true;
            }
        }
    };

    stop.store(true, Ordering::Relaxed);
    drop(rx_input);
    let _ = input_thread.join();
    result
}

impl TuiState {
    fn quit_requested(&self) -> bool {
        self.quit
    }
}

fn apply_ui_event(state: &mut TuiState, event: UiEvent, dirty: &mut bool) {
    *dirty = true;
    match event {
        UiEvent::Log(line) => push_entry(state, EntryKind::Log, &line),
        UiEvent::Reasoning { content } => {
            state.model_snippet = content.clone();
            state.streaming_buffer = content.clone();
            push_entry(
                state,
                EntryKind::Reasoning,
                &format!("── model output ──\n{content}"),
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
            state.goal_text = goal.clone();
            state.status = Some(StatusInfo {
                iteration,
                tokens,
                context_tokens,
                goal,
                elapsed,
            });
            if !matches!(state.input_mode, Some(InputMode::EditingTodo)) {
                state.todo_text = todo;
            }
        }
        UiEvent::Running(running) => {
            state.running = running;
            if running {
                state.agent_finished = false;
            }
        }
        UiEvent::AgentFinished { reason } => {
            push_entry(state, EntryKind::Log, &format!("Agent finished: {reason}"));
            state.agent_finished = true;
            state.running = false;
        }
        UiEvent::AgentError { error } => {
            push_entry(state, EntryKind::Error, &format!("Agent error: {error}"));
            state.agent_finished = true;
            state.running = false;
        }
        UiEvent::ModelRequestStart => {
            state.model_waiting = true;
        }
        UiEvent::ModelRequestEnd => {
            state.model_waiting = false;
        }
        UiEvent::ReasoningReset => {
            state.streaming_buffer.clear();
            state.model_snippet.clear();
        }
        UiEvent::ReasoningChunk { delta } => {
            state.streaming_buffer.push_str(&delta);
            const MAX_SNIPPET: usize = 8000;
            if state.streaming_buffer.len() > MAX_SNIPPET {
                // Use floor_char_boundary to avoid slicing in the middle of a multi-byte char.
                let target = state.streaming_buffer.len() - MAX_SNIPPET;
                let start = state.streaming_buffer.floor_char_boundary(target);
                state.model_snippet = state.streaming_buffer[start..].to_string();
            } else {
                state.model_snippet = state.streaming_buffer.clone();
            }
        }
        UiEvent::Quit => state.quit = true,
    }
}

// ============================================================
// CLI
// ============================================================

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// Base URL of an OpenAI-compatible endpoint.
    #[clap(long)]
    base_url: Option<String>,

    /// Model name.
    #[clap(long)]
    model: Option<String>,

    /// Agent goal (optional; can also be set later in TUI with `g`).
    #[clap(long, num_args = 0..=1, default_missing_value = "")]
    goal: Option<String>,

    /// Working directory. NOT a security boundary — see the module docs.
    #[clap(long)]
    workdir: Option<PathBuf>,

    /// Existing session to resume.
    #[clap(long)]
    session: Option<String>,

    /// Context token budget. 0 = auto-detect from server.
    #[clap(long, default_value_t = 0)]
    context_tokens: usize,

    /// Reasoning effort for thinking models, e.g. none|low|medium|high|xhigh.
    #[clap(long)]
    reasoning_effort: Option<String>,

    /// How the model is asked to call tools. `text` suits Qwen-style models;
    /// `native` uses OpenAI structured function calling (GPT-class).
    #[clap(long, value_enum, default_value_t = ToolMode::Text)]
    tool_mode: ToolMode,

    /// Maximum iterations.
    #[clap(long, default_value_t = 10000)]
    max_iterations: usize,

    /// Maximum wall-clock runtime in seconds.
    #[clap(long, default_value_t = 360000)]
    max_wall_secs: u64,

    /// Timeout for a single model request, in seconds.
    #[clap(long, default_value_t = 36000)]
    model_timeout_secs: u64,

    /// Timeout for a single run_command invocation, in seconds.
    #[clap(long, default_value_t = 600)]
    command_timeout_secs: u64,

    /// Disable the TUI and use plain logging.
    #[clap(long)]
    no_tui: bool,

    /// List available sessions and exit.
    #[clap(long)]
    list_sessions: bool,

    /// Resume the most recently modified session.
    #[clap(long)]
    resume_latest: bool,

    /// Compaction threshold as a percentage of the context budget (10-95).
    #[clap(long, default_value_t = 85)]
    compaction_threshold: usize,
}

// ============================================================
// Main
// ============================================================

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();

    if config.list_sessions {
        return list_sessions().await;
    }

    let base_url = config
        .base_url
        .clone()
        .ok_or_else(|| anyhow!("--base-url is required"))?;
    let model_name = config
        .model
        .clone()
        .ok_or_else(|| anyhow!("--model is required"))?;

    let api_key = std::env::var("LLM_API_KEY")
        .ok()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let context_tokens = if config.context_tokens > 0 {
        config.context_tokens
    } else {
        match detect_context_size(&base_url, api_key.as_deref()).await {
            Some(ctx) => ctx,
            None => {
                eprintln!(
                    "Could not auto-detect context size. Defaulting to 8192. \
                     Use --context-tokens to override."
                );
                8192
            }
        }
    };
    println!("Using context size: {context_tokens}");

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.model_timeout_secs + 30))
        .connect_timeout(Duration::from_secs(15))
        .build()?;

    let model = Model {
        client,
        base_url,
        model: model_name,
        temperature: 0.7,
        api_key,
        request_timeout_secs: config.model_timeout_secs,
        reasoning_effort: config.reasoning_effort.clone(),
        tool_mode: config.tool_mode,
    };

    let session_id = if config.resume_latest {
        get_session_list()
            .await?
            .into_iter()
            .next()
            .map(|s| s.id)
            .ok_or_else(|| anyhow!("No previous sessions found to resume."))?
    } else if let Some(id) = config.session.clone() {
        id
    } else {
        Uuid::new_v4().to_string()
    };
    validate_session_id(&session_id)
        .with_context(|| format!("refusing to use session id {session_id:?}"))?;

    let session_exists = session_dir(&session_id).exists();

    let stored_workdir = read_session_info(&session_id)
        .await
        .and_then(|i| i.workdir)
        .map(PathBuf::from);
    let requested_workdir = config
        .workdir
        .clone()
        .or(stored_workdir)
        .unwrap_or(std::env::current_dir()?);

    let workdir = match std::fs::canonicalize(&requested_workdir) {
        Ok(canon) => canon,
        Err(_) => {
            std::fs::create_dir_all(&requested_workdir)?;
            std::fs::canonicalize(&requested_workdir)?
        }
    };

    let mut session = if session_exists {
        println!("Resuming session {session_id}");
        load_session(
            &session_id,
            workdir.clone(),
            context_tokens,
            config.compaction_threshold,
            None,
        )
        .await
        .with_context(|| {
            format!("failed to load existing session {session_id}; refusing to overwrite it")
        })?
    } else {
        let goal = config.goal.clone().unwrap_or_default();
        println!("Creating new session {session_id}");
        create_session(
            &session_id,
            &goal,
            workdir.clone(),
            context_tokens,
            config.compaction_threshold,
        )
        .await?
    };

    if session_exists {
        if let Some(goal) = config.goal.clone() {
            let goal = goal.trim().to_string();
            if !goal.is_empty() && goal != session.goal {
                session.goal = goal.clone();
                tokio::fs::write(session_dir(&session_id).join("goal.txt"), &goal).await?;
                update_session_info(&session_id, Some(&goal), None, Some(&workdir)).await?;
            }
        }
    }

    tokio::fs::create_dir_all(&session.workdir).await?;

    let run_config = RunConfig {
        max_iterations: config.max_iterations,
        max_wall_secs: config.max_wall_secs,
        command_timeout_secs: config.command_timeout_secs,
    };

    if config.no_tui {
        let (tx_ui, mut rx_ui) = mpsc::unbounded_channel();
        let (_tx_cmd, rx_cmd) = mpsc::unbounded_channel();

        let printer = tokio::spawn(async move {
            while let Some(event) = rx_ui.recv().await {
                match event {
                    UiEvent::Log(line) => println!("[log] {line}"),
                    UiEvent::Reasoning { content } => println!("[model]\n{content}"),
                    UiEvent::Status {
                        iteration,
                        tokens,
                        context_tokens,
                        elapsed,
                        ..
                    } => println!(
                        "[status] iter {iteration} ctx {tokens}/{context_tokens} elapsed {}",
                        format_duration(elapsed)
                    ),
                    UiEvent::Running(running) => println!("[state] running={running}"),
                    UiEvent::AgentFinished { reason } => println!("[done] {reason}"),
                    UiEvent::AgentError { error } => eprintln!("[error] {error}"),
                    UiEvent::ModelRequestStart => println!("[model] request started"),
                    UiEvent::ModelRequestEnd => println!("[model] request ended"),
                    UiEvent::ReasoningReset => {}
                    UiEvent::ReasoningChunk { delta } => {
                        print!("{delta}");
                    }
                    UiEvent::Quit => break,
                }
                let _ = io::stdout().flush();
            }
        });

        let result = run_agent(
            &run_config,
            &model,
            &mut session,
            UiLogger::new(tx_ui),
            rx_cmd,
            false,
        )
        .await;

        let _ = tokio::time::timeout(Duration::from_secs(2), printer).await;
        result?;
    } else {
        install_panic_hook();

        let (tx_ui, rx_ui) = mpsc::unbounded_channel();
        let (tx_cmd, rx_cmd) = mpsc::unbounded_channel();
        let agent_config = run_config.clone();
        let agent_model = model.clone();
        let mut agent_session = session;
        let ui = UiLogger::new(tx_ui);
        let error_ui = ui.clone();

        let mut agent_handle = tokio::spawn(async move {
            if let Err(error) = run_agent(
                &agent_config,
                &agent_model,
                &mut agent_session,
                ui,
                rx_cmd,
                true,
            )
            .await
            {
                error_ui.send(UiEvent::AgentError {
                    error: error.to_string(),
                });
            }
        });

        let tui_result = run_tui(rx_ui, tx_cmd.clone()).await;

        let _ = tx_cmd.send(AgentCommand::Quit);
        drop(tx_cmd);
        if tokio::time::timeout(Duration::from_secs(5), &mut agent_handle)
            .await
            .is_err()
        {
            eprintln!("Agent did not stop within 5s; aborting it.");
            agent_handle.abort();
            let _ = agent_handle.await;
        }

        tui_result?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content_of(line: &[u8]) -> Option<String> {
        match parse_stream_line(line).unwrap() {
            StreamLine::Record(record) => record.content,
            _ => None,
        }
    }

    #[test]
    fn multibyte_char_split_across_chunks_is_not_corrupted() {
        let payload =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"caf\xc3\xa9 \xd0\xb4\xd0\xb0\"}}]}\n";
        // Split inside the two-byte 'é' and again inside 'д'.
        for cut in [payload.len() - 12, payload.len() - 7, payload.len() - 6] {
            let mut buf: Vec<u8> = Vec::new();
            let mut seen = String::new();
            for chunk in [&payload[..cut], &payload[cut..]] {
                buf.extend_from_slice(chunk);
                while let Some(line) = take_sse_line(&mut buf) {
                    seen.push_str(&content_of(&line).unwrap_or_default());
                }
            }
            assert_eq!(seen, "café да", "corrupted when split at {cut}");
            assert!(!seen.contains('\u{FFFD}'));
        }
    }

    #[test]
    fn native_tool_call_is_assembled_from_fragmented_deltas() {
        let ui = UiLogger::disabled();
        let mut acc = StreamAccumulator::default();
        // id/name arrive once, arguments dribble in across chunks.
        for (id, name, args) in [
            (Some("call_1"), Some("run_command"), Some("{\"comm")),
            (None, None, Some("and\":\"ls")),
            (None, None, Some(" -la\"}")),
        ] {
            acc.push(
                StreamRecord {
                    tool_calls: vec![ToolCallDelta {
                        index: 0,
                        id: id.map(str::to_string),
                        name: name.map(str::to_string),
                        arguments: args.map(str::to_string),
                    }],
                    ..Default::default()
                },
                &ui,
            );
        }
        let response = acc.finish().unwrap();
        let (calls, errors) = split_native_calls(&response.tool_calls);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(calls.len(), 1);
        let (id, call) = &calls[0];
        assert_eq!(id.as_deref(), Some("call_1"));
        assert_eq!(call.name, "run_command");
        assert_eq!(call.arguments["command"], "ls -la");
    }

    #[test]
    fn reasoning_is_never_scanned_for_tool_calls() {
        let ui = UiLogger::disabled();
        let mut acc = StreamAccumulator::default();
        // A draft call inside reasoning must not reach `text`, which is what
        // the text-mode extractor scans.
        acc.push(
            StreamRecord {
                reasoning: Some(
                    "<tool_call>\n{\"name\":\"run_command\",\"arguments\":{\"command\":\"ls\"}\n</tool_call>"
                        .into(),
                ),
                ..Default::default()
            },
            &ui,
        );
        acc.push(
            StreamRecord {
                content: Some("all done".into()),
                ..Default::default()
            },
            &ui,
        );
        let response = acc.finish().unwrap();
        assert_eq!(response.text, "all done");
        assert!(extract_tool_calls(&response.text).errors.is_empty());
    }

    #[test]
    fn history_cut_never_orphans_a_tool_result() {
        let messages = vec![
            Message::new("user", "go"),
            Message {
                role: "assistant".into(),
                tool_calls: vec![NativeToolCall::default()],
                ..Default::default()
            },
            Message::tool_result("call_1", "ok"),
            Message::tool_result("call_2", "ok"),
            Message::new("assistant", "done"),
        ];
        // Cutting at 2 would leave the history starting on a tool result.
        assert_eq!(safe_cut(&messages, 2), 4);
        assert_eq!(messages[safe_cut(&messages, 2)].role, "assistant");
        assert_eq!(safe_cut(&messages, 1), 1);
    }

    #[test]
    fn native_history_is_flattened_for_text_mode_models() {
        let messages = vec![
            Message {
                role: "assistant".into(),
                tool_calls: vec![NativeToolCall {
                    id: "call_1".into(),
                    kind: "function".into(),
                    function: NativeFunctionCall {
                        name: "run_command".into(),
                        arguments: "{\"command\":\"ls\"}".into(),
                    },
                }],
                ..Default::default()
            },
            Message::tool_result("call_1", "a.txt"),
        ];
        let flat = flatten_for_text_mode(&messages);
        assert!(flat.iter().all(|m| m.tool_calls.is_empty()));
        assert!(flat.iter().all(|m| m.tool_call_id.is_none()));
        assert!(flat[0].content.contains("<tool_call>"));
        assert_eq!(extract_tool_calls(&flat[0].content).calls.len(), 1);
        assert_eq!(flat[1].role, "user");
        assert!(flat[1].content.contains("Tool result for run_command"));
    }

    #[test]
    fn tools_are_sent_only_for_native_agent_turns() {
        let model = |mode| Model {
            client: reqwest::Client::new(),
            base_url: "http://localhost/v1".into(),
            model: "m".into(),
            temperature: 0.7,
            api_key: None,
            request_timeout_secs: 30,
            reasoning_effort: None,
            tool_mode: mode,
        };
        let messages = [Message::new("user", "hi")];

        let native_agent =
            model(ToolMode::Native).request_body(&messages, true, RequestKind::Agent);
        assert!(native_agent["tools"].is_array());
        assert_eq!(native_agent["tool_choice"], "required");

        // Summarization must stay tool-free or it gets answered with a call.
        let native_plain =
            model(ToolMode::Native).request_body(&messages, true, RequestKind::PlainText);
        assert!(native_plain["tools"].is_null());

        // Text mode must be byte-identical to the pre-change request shape.
        let text_agent = model(ToolMode::Text).request_body(&messages, true, RequestKind::Agent);
        assert!(text_agent["tools"].is_null());
        assert!(text_agent["tool_choice"].is_null());
    }

    #[test]
    fn pane_heights_grow_and_shrink_without_collapsing() {
        assert_eq!(fit_pane_heights([10, 6, 5], 25), [14, 6, 5]);
        assert_eq!(fit_pane_heights([10, 6, 5], 15), [4, 6, 5]);
        assert_eq!(fit_pane_heights([4, 8, 7], 9), [3, 3, 3]);
    }

    #[test]
    fn pane_boundaries_are_hit_tested_only_within_the_panes() {
        let areas = PaneAreas {
            transcript: Rect::new(2, 1, 20, 10),
            model: Rect::new(2, 11, 20, 6),
            todo: Rect::new(2, 17, 20, 5),
        };
        assert!(matches!(
            resize_boundary_at(areas, 10, 10),
            Some(ResizeBoundary::TranscriptModel)
        ));
        assert!(matches!(
            resize_boundary_at(areas, 10, 16),
            Some(ResizeBoundary::ModelTodo)
        ));
        assert!(resize_boundary_at(areas, 1, 10).is_none());
        assert!(resize_boundary_at(areas, 10, 9).is_none());
    }

    #[test]
    fn pane_resize_preserves_pair_total_and_minimums() {
        let mut state = TuiState::new();
        state.resize_drag = Some(ResizeDrag {
            boundary: ResizeBoundary::TranscriptModel,
            start_row: 10,
            initial_heights: [10, 6, 5],
        });
        resize_panes(&mut state, 14);
        assert_eq!(state.pane_heights, Some([13, 3, 5]));

        state.resize_drag = Some(ResizeDrag {
            boundary: ResizeBoundary::ModelTodo,
            start_row: 20,
            initial_heights: [13, 3, 5],
        });
        resize_panes(&mut state, 0);
        assert_eq!(state.pane_heights, Some([13, 3, 5]));
    }
}
