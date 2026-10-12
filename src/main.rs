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
use clap::Parser;
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
use futures_util::StreamExt;
use once_cell::sync::Lazy;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use uuid::Uuid;

mod media;
mod search;
mod tools;

use tools::ToolOutcome;

const MIN_MODEL_TIMEOUT_SECS: u64 = 15 * 60;
const MODEL_RETRY_ATTEMPTS: usize = 3;

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
    /// Base64 image attached to this message for the live request only.
    #[serde(default, skip)]
    image: Option<String>,
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
            image: None,
        }
    }
    fn with_image(
        role: &str,
        content: impl Into<String>,
        image: String,
        tool_call_id: Option<String>,
    ) -> Self {
        Message {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id,
            image: Some(image),
        }
    }
}

fn message_wire_content(message: &Message) -> serde_json::Value {
    match &message.image {
        Some(image) => {
            let url = format!("data:image/png;base64,{}", image);
            if message.content.is_empty() {
                serde_json::json!([
                    { "type": "image_url", "image_url": { "url": url } }
                ])
            } else {
                serde_json::json!([
                    { "type": "text", "text": message.content },
                    { "type": "image_url", "image_url": { "url": url } }
                ])
            }
        }
        None => serde_json::Value::String(message.content.clone()),
    }
}

fn to_native_wire_message(message: &Message) -> serde_json::Value {
    let mut value = serde_json::to_value(message).expect("Message should serialize");
    value["content"] = message_wire_content(message);
    let _ = value.as_object_mut().map(|obj| obj.remove("image"));
    value
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
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) arguments: serde_json::Value,
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
    /// Bad credentials. No amount of retrying or rewriting helps.
    Auth(String),
    /// The server rejected the request itself. Retrying the *same* bytes is
    /// pointless; the request has to change first.
    BadRequest(String),
    /// Network blip, rate limit, timeout, or server fault. Retry as-is with
    /// bounded backoff so temporary outages do not stop long runs.
    Transient(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Auth(m) | ChatError::BadRequest(m) | ChatError::Transient(m) => {
                write!(f, "{m}")
            }
        }
    }
}

impl std::error::Error for ChatError {}

/// A 5xx is treated as transient; the agent retries with capped backoff until
/// the service recovers or the user stops the run.
fn classify_status(status: reqwest::StatusCode, body: &str) -> ChatError {
    let message = format!("HTTP {status}: {}", truncate(body, 400));
    match status.as_u16() {
        401 | 403 => ChatError::Auth(message),
        408 | 425 | 429 => ChatError::Transient(message),
        _ if status.is_server_error() => ChatError::Transient(message),
        _ => ChatError::BadRequest(message),
    }
}

/// What a server reports when it rejects an oversized request, if it says.
#[derive(Default)]
struct ContextOverflow {
    /// Tokens the server counted in the rejected request.
    prompt_tokens: Option<usize>,
    /// The server's total context window (prompt + completion).
    n_ctx: Option<usize>,
}

impl ContextOverflow {
    /// True only when the body says the prompt is at/over the window, which is
    /// the one 400 we can actually fix by shrinking history.
    fn is_oversized(&self) -> bool {
        self.prompt_tokens
            .zip(self.n_ctx)
            .is_some_and(|(p, c)| p >= c)
    }
}

/// Pulls the token counts out of an error body so recovery can act on the
/// server's ground truth instead of the local char-based estimate. Understands
/// llama.cpp's `exceed_context_size_error` shape (`n_prompt_tokens` / `n_ctx`)
/// and the OpenAI-style `prompt_tokens` / `context_length` fields.
fn extract_context_overflow(body: &str) -> ContextOverflow {
    let number = |keys: &[&str]| -> Option<usize> {
        keys.iter().find_map(|key| {
            let start = body.find(key)?;
            let rest = &body[start + key.len()..];
            let rest = rest.find(':').map(|i| &rest[i + 1..])?;
            let digits: String = rest
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().ok().filter(|v: &usize| *v > 0)
        })
    };
    let prompt_tokens = number(&["n_prompt_tokens", "prompt_tokens", "input_tokens"]);
    let n_ctx = number(&["n_ctx", "context_length", "max_context_tokens"]);
    ContextOverflow {
        prompt_tokens,
        n_ctx,
    }
}

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
    tool_choice: String,
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
    /// Text shown to the user.
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
    response_bytes: usize,
}

impl StreamAccumulator {
    fn push(&mut self, record: StreamRecord, ui: &UiLogger) -> Result<(), ChatError> {
        const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
        let incoming_bytes = record.content.as_ref().map_or(0, String::len)
            + record.reasoning.as_ref().map_or(0, String::len)
            + record
                .tool_calls
                .iter()
                .map(|call| {
                    call.id.as_ref().map_or(0, String::len)
                        + call.name.as_ref().map_or(0, String::len)
                        + call.arguments.as_ref().map_or(0, String::len)
                })
                .sum::<usize>();
        if self.response_bytes.saturating_add(incoming_bytes) > MAX_RESPONSE_BYTES {
            return Err(ChatError::BadRequest(format!(
                "model response exceeded the {MAX_RESPONSE_BYTES} byte safety limit"
            )));
        }
        self.response_bytes += incoming_bytes;

        if let Some(reason) = record.finish_reason {
            self.finish_reason = Some(reason);
        }

        // Reasoning is surfaced live but deliberately kept out of `text` and
        // out of tool-call parsing.
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
        Ok(())
    }

    fn finish(self) -> Result<ChatResponse, ChatError> {
        let tool_calls: Vec<NativeToolCall> = self.calls.into_values().collect();
        if self.text.is_empty() && tool_calls.is_empty() && !self.saw_reasoning {
            return Err(ChatError::Transient("empty streamed response".into()));
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
            return Err(ChatError::BadRequest(format!("api error: {err}")));
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
        for (ordinal, item) in items.iter().enumerate() {
            tool_calls.push(ToolCallDelta {
                index: item
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(ordinal as u64) as usize,
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
    fn retry_budget(&self, max_attempts: usize) -> Duration {
        let attempts = max_attempts.max(1);
        let attempts_u64 = u64::try_from(attempts).unwrap_or(u64::MAX);
        let mut backoff = 1u64;
        let mut backoff_total = 0u64;
        for _ in 1..attempts {
            backoff_total = backoff_total.saturating_add(backoff);
            backoff = backoff.saturating_mul(2).min(60);
        }
        Duration::from_secs(
            self.request_timeout_secs
                .saturating_mul(attempts_u64)
                .saturating_add(backoff_total)
                .saturating_add(5),
        )
    }

    fn request_body(
        &self,
        messages: &[Message],
        stream: bool,
        kind: RequestKind,
    ) -> serde_json::Value {
        let wire: Vec<serde_json::Value> = messages.iter().map(to_native_wire_message).collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": wire,
            "temperature": self.temperature,
            "stream": stream
        });
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning_effort"] = serde_json::Value::String(effort.clone());
        }
        if kind == RequestKind::Agent {
            body["tools"] = tools::tool_definitions();
            body["tool_choice"] = serde_json::Value::String(self.tool_choice.clone());
        }
        body
    }

    /// SSE streaming variant. Sends `ReasoningChunk` events as tokens arrive.
    async fn chat_stream(
        &self,
        messages: &[Message],
        ui: &UiLogger,
        kind: RequestKind,
    ) -> Result<ChatResponse, ChatError> {
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
            let body = read_response_body_limited(resp, 64 * 1024).await;
            return Err(classify_status(status, &body));
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
                    StreamLine::Record(record) => acc.push(record, ui)?,
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
                StreamLine::Record(record) => acc.push(record, ui)?,
            }
        }

        acc.finish()
    }

    /// Retries transient failures with exponential backoff; returns fatal
    /// authentication or request errors to the agent recovery path.
    async fn chat_with_retry(
        &self,
        messages: &[Message],
        ui: &UiLogger,
        max_attempts: usize,
        kind: RequestKind,
    ) -> Result<ChatResponse, ChatError> {
        let max_attempts = max_attempts.max(1);
        let mut backoff = 1u64;
        let mut last: ChatError = ChatError::Transient("no attempts made".into());
        for attempt in 1..=max_attempts {
            ui.send(UiEvent::ReasoningReset);
            let result = tokio::time::timeout(
                Duration::from_secs(self.request_timeout_secs),
                self.chat_stream(messages, ui, kind),
            )
            .await;
            match result {
                Ok(Ok(content)) => return Ok(content),
                Ok(Err(fatal @ (ChatError::Auth(_) | ChatError::BadRequest(_)))) => {
                    return Err(fatal)
                }
                Ok(Err(ChatError::Transient(m))) => {
                    let retry = if attempt < max_attempts {
                        format!("; retrying in {backoff}s")
                    } else {
                        "; returning to agent recovery".to_string()
                    };
                    ui.log(format!(
                        "model error (attempt {attempt}/{max_attempts}): {m}{retry}"
                    ));
                    last = ChatError::Transient(m);
                }
                Err(_) => {
                    let m = format!("model call timed out after {}s", self.request_timeout_secs);
                    let retry = if attempt < max_attempts {
                        format!("; retrying in {backoff}s")
                    } else {
                        "; returning to agent recovery".to_string()
                    };
                    ui.log(format!(
                        "model error (attempt {attempt}/{max_attempts}): {m}{retry}"
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

// ============================================================
// Tool Call Extraction
// ============================================================

type IdentifiedCall = (Option<String>, ToolCall);
type IdentifiedError = (Option<String>, String);

/// Removes native calls that cannot be answered in the API protocol. Empty or
/// duplicate IDs otherwise leave an invalid assistant tool-call record in the
/// session and make every later request fail.
fn sanitize_native_tool_calls(
    calls: &[NativeToolCall],
) -> (Vec<NativeToolCall>, Vec<IdentifiedError>) {
    let mut accepted = Vec::new();
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();
    for call in calls {
        if call.id.trim().is_empty() {
            errors.push((
                None,
                format!("tool call {} did not include an ID", call.function.name),
            ));
        } else if !ids.insert(call.id.clone()) {
            errors.push((
                None,
                format!("tool call {} repeated ID {}", call.function.name, call.id),
            ));
        } else {
            accepted.push(call.clone());
        }
    }
    (accepted, errors)
}

/// Parses assembled native calls, pairing each with its `tool_call_id` so a
/// result (or an error) can be attributed back to it.
fn split_native_calls(calls: &[NativeToolCall]) -> (Vec<IdentifiedCall>, Vec<IdentifiedError>) {
    let mut parsed = Vec::new();
    let (calls, mut errors) = sanitize_native_tool_calls(calls);

    for call in &calls {
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

/// Resolves `p` relative to `workdir`, rejecting absolute paths, escaping
/// `..`, and — crucially — symlinked ancestors for paths that do not yet
/// exist (the classic write-through-a-symlink escape).
pub(crate) fn safe_path(workdir: &Path, p: &str) -> Result<PathBuf> {
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
    // `Path::exists` follows symlinks and reports false for a dangling link.
    // Walking past one would let a later write follow that link outside the
    // workdir. `symlink_metadata` sees the link itself, so canonicalization
    // below rejects both live and dangling links that cannot be resolved.
    while matches!(
        std::fs::symlink_metadata(&existing),
        Err(ref error) if error.kind() == io::ErrorKind::NotFound
    ) {
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

pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() > max_chars {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push_str("\n...[truncated]");
        out
    } else {
        s.to_string()
    }
}

fn truncate_tail(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let start = s
        .char_indices()
        .nth(count.saturating_sub(max_chars))
        .map(|(index, _)| index)
        .unwrap_or(s.len());
    format!("[earlier conversation omitted]\n{}", &s[start..])
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
// Shared HTTP client
// ============================================================

pub(crate) static HTTP: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(10))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .build()
        .expect("failed to build HTTP client")
});

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
        // Once history is compacted, everything before this snapshot is
        // redundant. Replacing the transcript keeps overnight sessions from
        // growing without bound while preserving the exact resumable state.
        atomic_write(
            &self.transcript_path,
            &format!(
                "{}\n",
                serde_json::to_string(&Event::Compaction {
                    summary: summary.to_string(),
                    retained: Some(retained.to_vec()),
                })?
            ),
        )
        .await
    }

    async fn refresh_todo(&mut self) -> String {
        match tokio::fs::read_to_string(&self.todo_path).await {
            Ok(todo) => self.todo_cache = todo,
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.todo_cache.clear(),
            // Retain the last known plan during transient permission or disk
            // errors instead of silently telling the model that all work is
            // gone.
            Err(_) => {}
        }
        self.todo_cache.clone()
    }

    /// Estimates the size of the *entire* request we will send, including the
    /// system prompt, goal, todo list, roles, tool definitions/arguments, and
    /// protocol overhead.
    fn estimate_tokens(&self) -> usize {
        estimate_tokens_for(
            &self.messages,
            &self.scratchpad,
            &self.goal,
            &self.todo_cache,
        )
    }
}

/// Atomically replaces a small session file. The temporary file is created in
/// the same directory so the final rename stays on one filesystem.
pub(crate) async fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?;
    tokio::fs::create_dir_all(parent).await?;
    if path.file_name().is_none() {
        return Err(anyhow!("{} has no file name", path.display()));
    }
    let temporary = parent.join(format!(".tmp-{}.tmp", Uuid::new_v4()));
    let _temporary_guard = TemporaryFileGuard(temporary.clone());
    let existing_permissions = match tokio::fs::metadata(path).await {
        Ok(metadata) => {
            let permissions = metadata.permissions();
            if permissions.readonly() {
                return Err(anyhow!("{} is read-only", path.display()));
            }
            Some(permissions)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };

    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await?;
        file.write_all(contents.as_bytes()).await?;
        file.flush().await?;
        if let Some(permissions) = existing_permissions {
            tokio::fs::set_permissions(&temporary, permissions).await?;
        }
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temporary, path).await?;
        Ok::<_, anyhow::Error>(())
    }
    .await;

    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

struct TemporaryFileGuard(PathBuf);

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Reads only a bounded prefix of an HTTP error response. Broken endpoints can
/// return arbitrarily large bodies; diagnostics need enough context to act,
/// not enough data to exhaust the overnight process.
async fn read_response_body_limited(resp: reqwest::Response, limit: usize) -> String {
    let mut body = Vec::with_capacity(limit.min(8192));
    let mut truncated = false;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        let remaining = limit.saturating_sub(body.len());
        let keep = chunk.len().min(remaining);
        body.extend_from_slice(&chunk[..keep]);
        if keep < chunk.len() {
            truncated = true;
            break;
        }
    }
    let mut text = String::from_utf8_lossy(&body).into_owned();
    if truncated {
        text.push_str("\n...[error response truncated]");
    }
    text
}

/// Static text in `static_system_prompt` is roughly this many characters.
const SYSTEM_PROMPT_OVERHEAD_CHARS: usize = 600;
/// Rough per-message wire overhead (role, JSON punctuation, chat template).
const PER_MESSAGE_OVERHEAD_CHARS: usize = 16;
/// Every agent turn sends the full `tools` schema array plus `tool_choice` as
/// top-level fields outside of `messages`. Those bytes are on the wire on
/// every request but previously were invisible to the estimator entirely,
/// which was the single biggest source of undercounting (a full tool
/// definitions block is easily tens of thousands of characters). This is a
/// rough fixed estimate of that block's size; it is deliberately generous
/// since undercounting here silently disables proactive compaction.
const TOOL_DEFINITIONS_OVERHEAD_CHARS: usize = 6000;

/// Improved token estimator with a 20% safety margin.
fn estimate_tokens_for(messages: &[Message], scratchpad: &str, goal: &str, todo: &str) -> usize {
    let mut chars = SYSTEM_PROMPT_OVERHEAD_CHARS
        + TOOL_DEFINITIONS_OVERHEAD_CHARS
        + scratchpad.chars().count()
        + goal.chars().count()
        + todo.chars().count();
    for message in messages {
        // `tool_calls` (function name + JSON arguments string) are part of
        // the wire payload (see `to_native_wire_message`) but were previously
        // omitted from the estimate entirely. For a turn where the model
        // calls a tool with a large payload (e.g. a full file body passed to
        // an edit tool, or a multi-line shell script), this was the dominant
        // source of the size, so it must be counted here.
        let tool_call_chars: usize = message
            .tool_calls
            .iter()
            .map(|c| c.function.name.chars().count() + c.function.arguments.chars().count() + 24)
            .sum();
        let image_chars = message
            .image
            .as_ref()
            .map(|image| (image.len() / 1024).clamp(1500, 6000) * 3)
            .unwrap_or(0);
        chars += message.content.chars().count()
            + message.role.chars().count()
            + tool_call_chars
            + image_chars
            + PER_MESSAGE_OVERHEAD_CHARS;
    }
    // Apply a 20% safety margin to avoid underestimating.
    ((chars / 3) as f64 * 1.2) as usize
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

    atomic_write(&path, &serde_json::to_string_pretty(&info)?).await?;
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
    atomic_write(&dir.join("goal.txt"), goal).await?;

    let transcript_path = dir.join("transcript.jsonl");
    if !transcript_path.exists() {
        atomic_write(&transcript_path, "").await?;
    }
    let todo_path = dir.join("todo.md");
    if !todo_path.exists() {
        atomic_write(&todo_path, "").await?;
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
                    // Old snapshots omitted their retained tail. The old
                    // compactor had already dropped that prefix in memory, so
                    // replay must also clear the pre-compaction messages.
                    None => {
                        messages.clear();
                        legacy_compactions += 1;
                    }
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
    session.recover_interrupted_tool_calls(ui).await?;
    session.refresh_todo().await;
    Ok(session)
}

impl Session {
    /// A process can stop after saving an assistant tool call but before it
    /// saves the result. Close any such calls before resuming so the next API
    /// request is a valid tool conversation instead of a permanently rejected
    /// transcript.
    async fn recover_interrupted_tool_calls(&mut self, ui: Option<&UiLogger>) -> Result<()> {
        let mut unresolved = BTreeSet::new();
        let mut repaired_messages = Vec::with_capacity(self.messages.len());
        let mut deferred_images = Vec::new();
        let mut invalid_calls = 0usize;
        let mut orphan_results = 0usize;
        for mut message in std::mem::take(&mut self.messages) {
            if message.role == "user" && message.content.starts_with("Image returned by ") {
                deferred_images.push(message);
                continue;
            }
            if message.role != "tool" && !deferred_images.is_empty() {
                repaired_messages.append(&mut deferred_images);
            }

            if message.role == "assistant" {
                let (valid_calls, errors) = sanitize_native_tool_calls(&message.tool_calls);
                invalid_calls += errors.len();
                message.tool_calls = valid_calls;
                for call in &message.tool_calls {
                    if !call.id.trim().is_empty() {
                        unresolved.insert(call.id.clone());
                    }
                }
                repaired_messages.push(message);
            } else if message.role == "tool" {
                if message
                    .tool_call_id
                    .as_ref()
                    .is_some_and(|id| unresolved.remove(id))
                {
                    repaired_messages.push(message);
                } else {
                    orphan_results += 1;
                }
            } else {
                repaired_messages.push(message);
            }
        }
        repaired_messages.append(&mut deferred_images);
        self.messages = repaired_messages;

        if invalid_calls > 0 || orphan_results > 0 {
            let summary = self.scratchpad.clone();
            self.append_compaction(&summary, &self.messages).await?;
            if let Some(ui) = ui {
                ui.log(format!(
                    "Repaired the saved conversation: removed {invalid_calls} invalid tool call(s) and {orphan_results} orphaned tool result(s)."
                ));
            }
        }

        let count = unresolved.len();
        for id in unresolved {
            self.push_message(Message::tool_result(
                id,
                "The previous process stopped before recording this tool result. Re-evaluate the task and continue.",
            ))
            .await?;
        }
        if invalid_calls > 0 {
            self.push_message(Message::new(
                "user",
                format!(
                    "A previous model response included {invalid_calls} tool call(s) with invalid or duplicate IDs, so they were removed. Continue with valid structured tool calls."
                ),
            ))
            .await?;
        }
        if count > 0 {
            if let Some(ui) = ui {
                ui.log(format!(
                    "Recovered {count} interrupted tool call(s) from the saved session."
                ));
            }
        }
        Ok(())
    }
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
            // session.json used to be refreshed only at startup and goal
            // changes, so an active session looked stale to --resume-latest.
            // Transcript and todo mtimes reflect actual work even if the
            // metadata write was interrupted.
            for file_name in ["transcript.jsonl", "goal.txt", "todo.md"] {
                if let Ok(metadata) = tokio::fs::metadata(path.join(file_name)).await {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(age) = modified.duration_since(UNIX_EPOCH) {
                            info.last_modified = info.last_modified.max(age.as_secs());
                        }
                    }
                }
            }
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

/// Static system prompt — no dynamic content. Goal, scratchpad, and todo are
/// placed in a separate user message before the conversation history.
fn static_system_prompt() -> String {
    r#"You are an autonomous coding agent.

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
- You can call screenshot() to see the user's monitor.
- You can call view_image(path) to see an image file from just its path.
- You can call render_page(target) to see a web page you are building, rendered offscreen. Prefer this over opening a browser, which would interrupt the user.
"#
    .to_string()
}

const COMPACTION_KEEP: usize = 8; // reduced from 12 to keep tighter history

/// Guards against a model that narrates intent ("I'll finish now") instead of
/// emitting a call, which would otherwise be nudged forever.
const MAX_NO_ACTION_TURNS: usize = 100;

/// Builds the history entries for one tool result.
///
/// An image always goes on its own user turn rather than on the tool result:
/// chat templates only reliably render media markers for user messages, and a
/// marker the template drops makes the server reject the entire request at
/// tokenize time.
fn tool_result_messages(id: Option<&str>, name: &str, outcome: ToolOutcome) -> Vec<Message> {
    let text = truncate(&outcome.text, 4000);
    let result = match id {
        Some(id) => Message::tool_result(id, text),
        None => Message::new("user", format!("Tool result for {name}: {text}")),
    };
    match outcome.image {
        None => vec![result],
        Some(image) => vec![
            result,
            Message::with_image("user", format!("Image returned by {name}."), image, None),
        ],
    }
}

/// Drops every attached image from history and reports how many went.
///
/// Images are the largest and least portable thing the harness puts on the
/// wire, which makes them the first suspect when a request the server used to
/// accept starts failing.
fn strip_images(messages: &mut [Message]) -> usize {
    let mut dropped = 0;
    for message in messages.iter_mut() {
        if message.image.take().is_some() {
            dropped += 1;
        }
    }
    dropped
}

/// Discards the most recent user turn and everything after it.
///
/// This is the escape hatch for a turn the server will never accept: without
/// it the loop rebuilds the identical request forever. It always shortens the
/// history, so repeated calls are guaranteed to make progress.
fn rollback_last_turn(messages: &mut Vec<Message>) -> bool {
    match messages.iter().rposition(|m| m.role == "user") {
        Some(index) if index > 0 => {
            messages.truncate(index);
            true
        }
        _ => false,
    }
}

/// Retries transient failures indefinitely with capped backoff. A rejected
/// request changes the history or stops with an actionable error.
async fn recover_from_model_error(
    error: ChatError,
    session: &mut Session,
    ui: &UiLogger,
    st: &mut AgentState,
) -> Option<Duration> {
    st.failed_turns = st.failed_turns.saturating_add(1);

    match error {
        ChatError::Auth(message) => {
            let error = format!("Authentication failed: {message}");
            ui.log(error.clone());
            ui.send(UiEvent::AgentError { error });
            st.finished = true;
            ui.send(UiEvent::Running(false));
            None
        }
        ChatError::Transient(message) => {
            let exponent = st.failed_turns.saturating_sub(1).min(5) as u32;
            let seconds = (15u64.saturating_mul(2u64.pow(exponent))).min(300);
            let delay = Duration::from_secs(seconds);
            ui.log(format!(
                "Transient model failure ({} consecutive): {message}; retrying in {delay:?}.",
                st.failed_turns
            ));
            Some(delay)
        }
        ChatError::BadRequest(message) => {
            ui.log(format!("Model rejected the request: {message}"));

            let dropped = strip_images(&mut session.messages);
            if dropped > 0 {
                ui.log(format!(
                    "Dropped {dropped} attached image(s) from context and retrying."
                ));
                return None;
            }

            let overflow = extract_context_overflow(&message);
            if overflow.is_oversized() {
                if fit_to_context(session, ui, &overflow).await {
                    ui.log("History now fits the context window; retrying.");
                    return None;
                }
                let window = overflow.n_ctx.unwrap_or(session.context_tokens);
                let error = format!(
                    "Context overflow remains after trimming history. The goal, scratchpad, todo, or a single message is too large for the {window} token window. Shorten it or increase the model context size."
                );
                ui.log(error.clone());
                ui.send(UiEvent::AgentError { error });
                st.finished = true;
                ui.send(UiEvent::Running(false));
                return None;
            }

            if rollback_last_turn(&mut session.messages) {
                ui.log("Discarded the last conversation turn rejected by the server; retrying.");
                let summary = session.scratchpad.clone();
                if let Err(e) = session.append_compaction(&summary, &session.messages).await {
                    let error = format!("Could not persist request recovery: {e}");
                    ui.log(error.clone());
                    ui.send(UiEvent::AgentError { error });
                    st.finished = true;
                    ui.send(UiEvent::Running(false));
                    return None;
                }
                let note = format!(
                    "The previous turn was rejected by the model ({message}). Take a different, smaller step."
                );
                if let Err(e) = session.push_message(Message::new("user", note)).await {
                    let error = format!("Could not persist the request recovery note: {e}");
                    ui.log(error.clone());
                    ui.send(UiEvent::AgentError { error });
                    st.finished = true;
                    ui.send(UiEvent::Running(false));
                }
                return None;
            }

            let error = format!("Request is still being rejected after recovery: {message}");
            ui.log(error.clone());
            ui.send(UiEvent::AgentError { error });
            st.finished = true;
            ui.send(UiEvent::Running(false));
            None
        }
    }
}

/// Long runs drift away from the todo list, leaving finished work unchecked and
/// new work unrecorded. Re-anchor periodically.
const TODO_REMINDER_EVERY: usize = 20;

const TODO_REMINDER: &str =
    "Reminder: reconcile the todo list with reality now, using update_todo. \
Check off every item you have actually completed, delete items that are stale or no longer \
relevant, and add any new work you have discovered since. Keep exactly one item marked as in \
progress. Then carry on with the goal.";

/// Hard ceiling on the size of a compacted scratchpad summary.
///
/// `maybe_compact` merges the *previous* scratchpad into the *new* one on
/// every call, so a summarizer that is merely "concise" rather than bounded
/// can grow monotonically across repeated compactions over a long run,
/// eroding the very budget compaction exists to protect. This truncation is
/// the hard backstop; the prompt's "be concise" instruction is only a
/// preference the model may not honor.
const MAX_SCRATCHPAD_CHARS: usize = 12000;
/// Renders one history message for the summarizer, including any tool call
/// the assistant made.
///
/// Previously this used only `m.content`, which is empty or near-empty for a
/// typical assistant turn under `tool_choice = "required"` — the actual
/// action (which command was run, which file was edited, with what
/// arguments) lives in `m.tool_calls` instead. Omitting it meant the rolling
/// summary only ever recorded *what came back* from a tool, never *what was
/// asked of it*, which is precisely the information ("commands that worked or
/// failed, file paths") the summarizer prompt asks it to preserve.
fn render_message_for_summary(m: &Message) -> String {
    let image_note = if m.image.is_some() {
        " [image attached]"
    } else {
        ""
    };
    if m.tool_calls.is_empty() {
        format!("{}: {}{}", m.role, m.content, image_note)
    } else {
        let calls: Vec<String> = m
            .tool_calls
            .iter()
            .map(|c| format!("{}({})", c.function.name, c.function.arguments.trim()))
            .collect();
        format!(
            "{}: {}{} [tool calls: {}]",
            m.role,
            m.content,
            image_note,
            calls.join(", ")
        )
    }
}

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
        .map(render_message_for_summary)
        .collect::<Vec<_>>()
        .join("\n");

    let full_combined = if session.scratchpad.trim().is_empty() {
        format!("Goal: {}\n\nConversation:\n{}", session.goal, old_text)
    } else {
        format!(
            "Goal: {}\n\nSummary of everything before this point:\n{}\n\nNew conversation since \
             that summary:\n{}",
            session.goal, session.scratchpad, old_text
        )
    };
    let combined = if full_combined.chars().count() <= 60000 {
        full_combined
    } else {
        let goal = truncate(&session.goal, 4000);
        let previous = truncate(&session.scratchpad, MAX_SCRATCHPAD_CHARS);
        let prefix = format!(
            "Goal: {goal}\n\nSummary of everything before this point:\n{previous}\n\n\
             Recent conversation (earlier portion omitted):\n"
        );
        let omission_marker = "[earlier conversation omitted]\n";
        let budget = 60000usize
            .saturating_sub(prefix.chars().count())
            .saturating_sub(omission_marker.chars().count());
        format!("{prefix}{}", truncate_tail(&old_text, budget))
    };

    let summary_messages = vec![
        Message {
            role: "system".into(),
            content: format!(
                "You are a summarizer for a long-running autonomous agent. Merge the previous \
                      summary with the new conversation into a single self-contained summary. \
                      Preserve the goal, completed work, important findings, file paths, commands \
                      that worked or failed (including the exact command or tool arguments used), \
                      the current plan, and next actions. Never drop information from the previous \
                      summary. Be concise: keep the summary under {} characters.",
                MAX_SCRATCHPAD_CHARS
            ),
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
    let retry_budget = model.retry_budget(MODEL_RETRY_ATTEMPTS);
    let result = tokio::time::timeout(
        retry_budget,
        model.chat_with_retry(
            &summary_messages,
            ui,
            MODEL_RETRY_ATTEMPTS,
            RequestKind::PlainText,
        ),
    )
    .await
    .unwrap_or_else(|_| {
        Err(ChatError::Transient(format!(
            "compaction timed out after {retry_budget:?}"
        )))
    });
    ui.send(UiEvent::ModelRequestEnd);

    match result {
        Ok(response) if !response.text.trim().is_empty() => {
            // Hard backstop: the model may ignore the "be concise" prompt
            // instruction, and this call merges the old scratchpad into the
            // new one every time, so an unbounded summary would grow across
            // repeated compactions instead of shrinking the context.
            let summary = truncate(response.text.trim(), MAX_SCRATCHPAD_CHARS);
            let retained = session.messages[split..].to_vec();
            if let Err(e) = session.append_compaction(&summary, &retained).await {
                ui.log(format!("Failed to persist compaction event: {e}"));
                return;
            }
            session.messages = retained;
            session.scratchpad = summary.clone();
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

async fn trim_history_to_fit(session: &mut Session, ui: &UiLogger) {
    let reserve = (session.context_tokens / 4).clamp(512, 4096);
    let limit = session.context_tokens.saturating_sub(reserve);
    let original = session.messages.clone();
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
        let summary = session.scratchpad.clone();
        if let Err(e) = session.append_compaction(&summary, &session.messages).await {
            ui.log(format!("Could not persist the trimmed history: {e}"));
            session.messages = original;
        }
    }
}

/// Hard guarantee that the next request fits the context window, driven by the
/// token counts the server reported in its rejection.
///
/// `trim_history_to_fit` runs every turn but trusts the local char-based
/// estimate, which can sit well below the server's real count (the system
/// prompt and chat-template overhead are only approximated) and so miss the
/// overflow. This is the emergency brake that runs *after* the server has
/// actually rejected the request: it converts the server's numbers into a local
/// trim target and drops the oldest whole turns until the request is small
/// enough, leaving headroom for the model's own output. Returns `true` when
/// history shrank so the caller can tell a real fix from a request that is
/// already at its floor.
async fn fit_to_context(session: &mut Session, ui: &UiLogger, overflow: &ContextOverflow) -> bool {
    let before = session.messages.len();
    let original = session.messages.clone();
    let window = overflow.n_ctx.unwrap_or(session.context_tokens);
    if window == 0 {
        return false;
    }
    let server = overflow.prompt_tokens.unwrap_or(session.estimate_tokens());
    // Leave ~5% of the window for the completion; the prompt alone must fit
    // inside the rest or the server rejects it before generating anything.
    let target_server = (window as f64) * 0.95;
    let local = session.estimate_tokens().max(1);
    // Server tokens per local token: how much the real count inflates the
    // estimate. If the server saw more than we think, we must trim more.
    let cost = (server as f64) / (local as f64);
    let mut to_drop_local = ((server as f64 - target_server).max(0.0)) / cost;

    while to_drop_local > 0.0 && session.messages.len() > 2 {
        // Drop the oldest whole turn; `safe_cut` skips any leading `tool`
        // results so we never orphan a result from its assistant call, and
        // the `.min(len - 1)` keeps at least the newest message so the loop
        // always makes progress without emptying the history.
        let cut = safe_cut(&session.messages, 1)
            .min(session.messages.len() - 1)
            .max(1);
        // The drop's cost is the exact delta of the request estimate, which
        // (unlike re-estimating the dropped slice, which would re-add the
        // fixed system/scratchpad/goal overhead) accounts for exactly what
        // left the request.
        let before_est = session.estimate_tokens();
        session.messages.drain(..cut);
        let dropped_local = before_est.saturating_sub(session.estimate_tokens());
        to_drop_local -= dropped_local as f64;
        ui.log(format!(
            "Context overflow: dropped {cut} oldest message(s) to fit the window \
             (server saw {server} of {window} tokens)."
        ));
    }

    if session.messages.len() != before {
        // Persist so a reload replays the shrunken history, not the oversized
        // request that got rejected. Crucially, this must NOT overwrite
        // `session.scratchpad` with a throwaway status line: `load_session`
        // treats every `Event::Compaction.summary` as the scratchpad
        // unconditionally, so a prior call's real distilled summary (built up
        // over possibly many proper `maybe_compact` runs) would otherwise be
        // silently replaced by "History trimmed..." on the next reload, and
        // in-memory vs. reloaded state would diverge. We keep persisting the
        // *existing* scratchpad text unchanged and only log the trim event.
        let summary = session.scratchpad.clone();
        if let Err(e) = session.append_compaction(&summary, &session.messages).await {
            ui.log(format!("Could not record the context trim: {e}"));
            session.messages = original;
            return false;
        }
        return true;
    }
    false
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

/// Sends one throwaway tools request so a misconfigured server is caught at
/// startup, not mid-run. llama-server rejects `tools` with a 400 unless it was
/// launched with `--jinja`, which otherwise looks like the model simply never
/// calling anything.
async fn warn_if_tools_unsupported(model: &Model) {
    let probe = [Message::new("user", "ping")];
    let mut body = model.request_body(&probe, false, RequestKind::Agent);
    body["max_tokens"] = serde_json::Value::from(1);

    let url = join_url(&model.base_url, "chat/completions");
    let mut req = model.client.post(&url).json(&body);
    if let Some(key) = &model.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }

    let Ok(Ok(resp)) = tokio::time::timeout(Duration::from_secs(10), req.send()).await else {
        return;
    };
    if resp.status() != reqwest::StatusCode::BAD_REQUEST {
        return;
    }
    let body = tokio::time::timeout(
        Duration::from_secs(5),
        read_response_body_limited(resp, 64 * 1024),
    )
    .await
    .unwrap_or_else(|_| "error response body timed out".to_string());
    if body.contains("jinja") || body.contains("tool") {
        eprintln!(
            "WARNING: the server rejected a tool-calling request:\n  {}\n\
             Start llama-server with --jinja (and a tool-aware chat template) \
             or the agent cannot call tools.",
            truncate(&body, 300)
        );
    }
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
    let retry_budget = model.retry_budget(MODEL_RETRY_ATTEMPTS);
    // Planning is optional, but local models may need the full request timeout
    // for each retry before we fall back to a minimal todo list.
    let result = match tokio::time::timeout(
        retry_budget,
        model.chat_with_retry(&messages, ui, MODEL_RETRY_ATTEMPTS, RequestKind::PlainText),
    )
    .await
    {
        Ok(result) => result
            .map(|response| response.text)
            .map_err(|e| anyhow!("{e}")),
        Err(_) => Err(anyhow!(
            "initial todo generation timed out after {retry_budget:?}"
        )),
    };
    ui.send(UiEvent::ModelRequestEnd);
    result
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
    failed_turns: usize,
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
            st.no_action_streak = 0;
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
            let dir = session_dir(&session.id);
            if let Err(e) = atomic_write(&dir.join("goal.txt"), &new_goal).await {
                ui.log(format!("Failed to save new goal: {e}"));
                return Ok(CmdOutcome::Continue);
            }
            session.goal = new_goal.clone();
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
            st.no_action_streak = 0;
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
            st.no_action_streak = 0;
            st.finished = false;
            ui.send(UiEvent::Running(!st.paused));
        }
        AgentCommand::UpdateTodo(content) => {
            if let Err(e) = atomic_write(&session.todo_path, &content).await {
                ui.log(format!("Failed to save todo: {e}"));
                return Ok(CmdOutcome::Continue);
            }
            ui.log("Todo updated by user.");
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
            st.no_action_streak = 0;
            st.finished = false;
            ui.send(UiEvent::Running(!st.paused));
            // Immediately notify the UI about the new todo content.
            send_status(ui, session, st);
        }
        AgentCommand::CompactNow => {
            ui.log("Manual compaction requested.");
            maybe_compact(session, model, ui, true).await;
            trim_history_to_fit(session, ui).await;
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
                    trim_history_to_fit(&mut new_session, ui).await;
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

/// Lets the agent consume pause, goal, todo, and quit commands while a tool is
/// running. Quit drops the tool future; shell children are configured to die
/// when that future is dropped.
async fn execute_tool_interruptibly(
    call: &ToolCall,
    workdir: &Path,
    todo_path: &Path,
    command_timeout_secs: u64,
    rx_cmd: &mut mpsc::UnboundedReceiver<AgentCommand>,
    pending: &mut VecDeque<AgentCommand>,
) -> Option<Result<ToolOutcome>> {
    let execution = tools::execute_tool(call, workdir, todo_path, command_timeout_secs);
    tokio::pin!(execution);
    loop {
        tokio::select! {
            result = &mut execution => return Some(result),
            command = rx_cmd.recv() => match command {
                Some(AgentCommand::Quit) | None => return None,
                Some(command) => pending.push_back(command),
            }
        }
    }
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
        failed_turns: 0,
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
                let todo = if todo.trim().is_empty() {
                    "- [ ] Make progress toward the goal\n- [ ] Verify the goal is complete"
                        .to_string()
                } else {
                    todo
                };
                if let Err(e) = atomic_write(&session.todo_path, &todo).await {
                    ui.log(format!("Failed to write initial todo: {e}"));
                } else {
                    session.refresh_todo().await;
                    ui.log(format!("Initial todo:\n{todo}"));
                }
            }
            Err(e) => {
                ui.log(format!(
                    "Failed to generate initial todo: {e}; using a minimal plan."
                ));
                let fallback =
                    "- [ ] Make progress toward the goal\n- [ ] Verify the goal is complete";
                if let Err(e) = atomic_write(&session.todo_path, fallback).await {
                    ui.log(format!("Failed to write fallback todo: {e}"));
                } else {
                    session.refresh_todo().await;
                }
            }
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
        trim_history_to_fit(session, &ui).await;
        session.refresh_todo().await;

        // ----- Build the request: static system prompt + goal/scratchpad/todo + history -----
        let mut messages = vec![
            Message {
                role: "system".into(),
                content: static_system_prompt(),
                ..Default::default()
            },
            Message {
                role: "user".into(),
                content: format!(
                    "Goal:\n{}\n\nScratchpad:\n{}\n\nTodo:\n{}",
                    session.goal, session.scratchpad, session.todo_cache
                ),
                ..Default::default()
            },
        ];
        // Append the conversation history (which may end with assistant or tool)
        messages.extend(session.messages.clone());

        // ---- Model call, interruptible by user commands ----
        send_status(&ui, session, &st);
        let call = model.chat_with_retry(&messages, &ui, MODEL_RETRY_ATTEMPTS, RequestKind::Agent);
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
            Ok(response) => {
                st.failed_turns = 0;
                response
            }
            Err(error) => {
                if let Some(delay) = recover_from_model_error(error, session, &ui, &mut st).await {
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        command = rx_cmd.recv() => match command {
                            Some(command) => pending.push_back(command),
                            None => return Ok(()),
                        }
                    }
                }
                continue;
            }
        };

        ui.send(UiEvent::Reasoning {
            content: response.text.clone(),
        });

        if matches!(response.finish_reason.as_deref(), Some("length")) {
            ui.log("Model output was cut off by the token limit; asking it to be brief.");
        }

        let (persisted_tool_calls, invalid_id_errors) =
            sanitize_native_tool_calls(&response.tool_calls);
        let (calls, mut errors) = split_native_calls(&persisted_tool_calls);
        errors.extend(invalid_id_errors);

        session
            .push_message(Message {
                role: "assistant".into(),
                content: response.text.clone(),
                tool_calls: persisted_tool_calls,
                tool_call_id: None,
                image: None,
            })
            .await?;

        // Report malformed calls, but still run the valid ones. A native call
        // must still receive a matching tool result or the next request is
        // invalid, so the error is delivered as that result.
        let mut deferred_user_errors = Vec::new();
        for (id, error) in &errors {
            ui.log(format!("Malformed tool call: {error}"));
            let message = match id {
                Some(id) => Message::tool_result(id, format!("ERROR: {error}")),
                None => Message::new(
                    "user",
                    format!(
                        "One of your tool calls was malformed and was ignored: {error}\n\
                         Please issue a valid structured tool call with a non-empty ID and valid JSON arguments."
                    ),
                ),
            };
            if message.role == "user" {
                deferred_user_errors.push(message);
            } else {
                session.push_message(message).await?;
            }
        }

        let mut quit_after_tool = false;
        if calls.is_empty() {
            st.no_action_streak += 1;
            if st.no_action_streak >= MAX_NO_ACTION_TURNS {
                let error = format!(
                    "Stopped after {} consecutive turns without a tool call. The server may \
                     not be returning structured tool_calls; start llama-server with --jinja.",
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
            // A `finish` call may not be the last entry in a multi-call
            // batch. Previously, hitting `finish` mid-batch `break`s
            // immediately, leaving every call *after* it in this response
            // without a matching tool result. Since the assistant message
            // carrying all of those `tool_calls` is already persisted, a
            // resumed session would replay it with an unanswered
            // `tool_call_id`, which most OpenAI-compatible servers reject
            // outright — burning part of the model-failure budget on a
            // harness bug rather than a real transient error. Every call in
            // the batch must get a result; calls after `finish` are answered
            // with a "skipped" result instead of being executed.
            let mut finish_reason: Option<String> = None;
            let mut deferred_images = Vec::new();
            let mut calls = calls.into_iter();
            while let Some((id, call)) = calls.next() {
                if finish_reason.is_some() {
                    if let Some(id) = &id {
                        session
                            .push_message(Message::tool_result(
                                id,
                                "skipped: agent already called finish() earlier in this turn",
                            ))
                            .await?;
                    }
                    continue;
                }

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
                    finish_reason = Some(reason);
                    continue;
                }

                ui.log(format!("Executing tool: {}", call.name));
                let execution = execute_tool_interruptibly(
                    &call,
                    &session.workdir,
                    &session.todo_path,
                    config.command_timeout_secs,
                    &mut rx_cmd,
                    &mut pending,
                )
                .await;
                let outcome = match execution {
                    Some(Ok(outcome)) => outcome,
                    Some(Err(e)) => ToolOutcome::plain(format!("ERROR: {e}")),
                    None => {
                        ui.log(format!("Cancelled tool {} while quitting.", call.name));
                        if let Some(id) = &id {
                            session
                                .push_message(Message::tool_result(
                                    id,
                                    "cancelled because the agent is quitting",
                                ))
                                .await?;
                        }
                        for (skipped_id, _) in calls {
                            if let Some(skipped_id) = skipped_id {
                                session
                                    .push_message(Message::tool_result(
                                        skipped_id,
                                        "skipped because the agent is quitting",
                                    ))
                                    .await?;
                            }
                        }
                        quit_after_tool = true;
                        break;
                    }
                };
                ui.log(format!("Result: {}", truncate_display(&outcome.text, 200)));

                if matches!(call.name.as_str(), "update_todo") {
                    session.refresh_todo().await;
                }

                for message in tool_result_messages(id.as_deref(), &call.name, outcome) {
                    if message.image.is_some() {
                        deferred_images.push(message);
                    } else {
                        session.push_message(message).await?;
                    }
                }
            }

            // Keep every result adjacent to the assistant's tool-call batch.
            // Some endpoints reject a user image message inserted between
            // tool results, so attach captured images only after all results
            // have been recorded.
            for image_message in deferred_images {
                session.push_message(image_message).await?;
            }

            if let Some(reason) = finish_reason {
                ui.send(UiEvent::AgentFinished {
                    reason: reason.clone(),
                });
                ui.send(UiEvent::Running(false));
                st.finished = true;
            }
        }

        for error_message in deferred_user_errors {
            session.push_message(error_message).await?;
        }

        if quit_after_tool {
            let _ = update_session_info(
                &session.id,
                Some(&session.goal),
                None,
                Some(&session.workdir),
            )
            .await;
            ui.log("Quit command received, stopping agent.");
            ui.send(UiEvent::Quit);
            return Ok(());
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

/// Builds only the transcript rows visible at the current scroll position.
/// Rewrapping and copying the entire 5,000-entry history every frame made the
/// TUI progressively slower during long model streams.
fn visible_transcript_lines(
    state: &mut TuiState,
    width: usize,
    height: usize,
) -> Vec<(EntryKind, String)> {
    loop {
        let requested_offset = state.scroll_offset;
        let mut skip = requested_offset;
        let mut scanned = 0usize;
        let mut visible_reversed = Vec::with_capacity(height);
        let mut exhausted = true;

        'entries: for entry in state.transcript.iter().rev() {
            for logical_line in entry.text.split('\n').rev() {
                let wrapped = wrap_line(logical_line, width);
                for line in wrapped.iter().rev() {
                    scanned += 1;
                    if skip > 0 {
                        skip -= 1;
                        continue;
                    }
                    visible_reversed.push((entry.kind.clone(), line.clone()));
                    if visible_reversed.len() >= height {
                        exhausted = false;
                        break 'entries;
                    }
                }
            }
        }

        if exhausted {
            let max_offset = scanned.saturating_sub(height);
            if requested_offset > max_offset {
                state.scroll_offset = max_offset;
                continue;
            }
        }

        visible_reversed.reverse();
        return visible_reversed;
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
    updated_at: Instant,
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
    /// Bounded preview of the current model call. The full streamed output is
    /// appended to the scrollable transcript as it arrives.
    streaming_buffer: String,
    reasoning_entry_active: bool,
    reasoning_entry_chars: usize,
    reasoning_stream_started: bool,
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
            reasoning_entry_active: false,
            reasoning_entry_chars: 0,
            reasoning_stream_started: false,
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

/// Appends streamed reasoning/output to bounded transcript entries. This
/// preserves scrollback across tool commands and splits very long generations
/// into manageable pieces instead of retaining one unbounded string.
fn append_reasoning_log(state: &mut TuiState, mut remaining: &str) {
    const MAX_ENTRY_CHARS: usize = 4096;

    while !remaining.is_empty() {
        let last_is_reasoning = state
            .transcript
            .last()
            .is_some_and(|entry| entry.kind == EntryKind::Reasoning);
        if !state.reasoning_entry_active || !last_is_reasoning {
            let title = if state.reasoning_stream_started {
                "── model output (continued) ──"
            } else {
                "── model output ──"
            };
            push_entry(state, EntryKind::Reasoning, title);
            state.reasoning_entry_active = true;
            state.reasoning_entry_chars = 0;
        }

        let available = MAX_ENTRY_CHARS.saturating_sub(state.reasoning_entry_chars);
        if available == 0 {
            state.reasoning_entry_active = false;
            continue;
        }
        let end = remaining
            .char_indices()
            .nth(available)
            .map(|(index, _)| index)
            .unwrap_or(remaining.len());
        let (piece, rest) = remaining.split_at(end);
        if let Some(entry) = state.transcript.last_mut() {
            entry.text.push_str(piece);
        }
        let appended = piece.chars().count();
        state.reasoning_entry_chars += appended;
        state.reasoning_stream_started = true;
        remaining = rest;

        if state.reasoning_entry_chars >= MAX_ENTRY_CHARS {
            state.reasoning_entry_active = false;
        }
    }
}

fn truncate_stream_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let start = text.floor_char_boundary(text.len() - max_bytes);
    text[start..].to_string()
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

        let visible = visible_transcript_lines(state, inner_width, visible_height);
        let lines: Vec<Line> = visible
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
                    .title("Current model output (full stream in transcript)"),
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
                let pct = s
                    .tokens
                    .saturating_mul(100)
                    .checked_div(s.context_tokens)
                    .unwrap_or(0);
                format!(
                    "{} | Goal: {} | iter {} | ctx {}/{} ({}%) | elapsed {} | agent: {}",
                    spinner,
                    truncate_display(&s.goal, 40),
                    s.iteration,
                    s.tokens,
                    s.context_tokens,
                    pct,
                    format_duration(s.elapsed + s.updated_at.elapsed()),
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

/// On the legacy Windows console, bracketed paste is unavailable, so a paste
/// arrives as a rapid burst of individual key events (an `Enter` for every
/// newline). If a printable key is immediately followed by more queued events,
/// treat the whole burst as one paste: rebuild the text and return it as
/// `CEvent::Paste`. That way embedded newlines are inserted as text instead of
/// submitting the entry (and then leaking the remaining characters into the
/// global key handler, where a `t` opens the todo editor and an `Enter`
/// auto-adds `- [ ] ` items).
/// True for the events that make up pasted text: a plain (no Control)
/// printable key press, an Enter (a pasted newline), or a real bracketed
/// paste. Everything else in a burst (mouse, resize, focus, control keys) is
/// not paste text and must be handled normally rather than swallowed.
fn is_paste_text(ev: &CEvent) -> bool {
    match ev {
        CEvent::Key(k) => {
            k.kind == KeyEventKind::Press
                && !k.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(k.code, KeyCode::Char(_) | KeyCode::Enter)
        }
        CEvent::Paste(_) => true,
        _ => false,
    }
}

/// Splits a captured burst into the pasted text and the events that are not
/// paste text. The text is rebuilt from every paste-text event in order (so a
/// multi-line paste keeps its newlines and its tail); the non-text events are
/// returned so the caller can re-inject them instead of dropping them.
fn split_burst(events: &[CEvent]) -> (String, Vec<CEvent>) {
    let mut text = String::new();
    let mut extra = Vec::new();
    for ev in events {
        if is_paste_text(ev) {
            match ev {
                CEvent::Key(k) => match k.code {
                    KeyCode::Char(c) => text.push(c),
                    KeyCode::Enter => text.push('\n'),
                    _ => extra.push(ev.clone()),
                },
                CEvent::Paste(p) => text.push_str(p),
                _ => {}
            }
        } else {
            extra.push(ev.clone());
        }
    }
    (text, extra)
}

async fn coalesce_paste_burst(
    rx: &mut mpsc::UnboundedReceiver<CEvent>,
    first: CEvent,
) -> (CEvent, Vec<CEvent>) {
    // Only a printable key press or an Enter can start a paste burst.
    if !is_paste_text(&first) {
        return (first, Vec::new());
    }
    // If nothing else arrives within a short window, this is a lone
    // keystroke; keep it as-is. A paste has more events queued, so wait a
    // little rather than using a pure try_recv (which misses a burst whose
    // tail is a few milliseconds behind).
    let mut events = vec![first.clone()];
    match tokio::time::timeout(Duration::from_millis(15), rx.recv()).await {
        Ok(Some(ev)) => events.push(ev),
        Ok(None) | Err(_) => return (first, Vec::new()),
    }
    // A burst is underway. Keep collecting while events keep arriving within a
    // short gap, so a large paste is captured as one contiguous block.
    let gap = Duration::from_millis(30);
    loop {
        match tokio::time::timeout(gap, rx.recv()).await {
            Ok(Some(ev)) => events.push(ev),
            Ok(None) | Err(_) => break,
        }
    }
    // The event that broke the loop is the paste's tail: if it is still queued
    // (it arrived exactly as the gap expired), salvage it so the paste is not
    // truncated and its final character is not lost.
    if let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    let (text, extra) = split_burst(&events);
    if text.is_empty() {
        // No paste text at all: hand the first event back and re-inject the
        // rest so nothing is lost.
        let rest = events.into_iter().skip(1).collect();
        (first, rest)
    } else {
        // One atomic paste plus the non-text events the burst carried.
        (CEvent::Paste(text), extra)
    }
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
        if let Err(error) = execute!(
            out,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture,
            Hide
        ) {
            let _ = disable_raw_mode();
            return Err(error.into());
        }
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
        'thread: while !thread_stop.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => {
                    // Drain every event already in the OS buffer so a paste
                    // (hundreds of keys) is pushed to the channel as one tight
                    // burst. Reading one event per poll would space the burst
                    // out by the poll interval, wider than the coalescer's
                    // gap, and leak the paste into the global key handler.
                    loop {
                        if !event::poll(Duration::from_millis(1)).unwrap_or(false) {
                            break;
                        }
                        match event::read() {
                            Ok(ev) => {
                                if tx_input.send(ev).is_err() {
                                    break 'thread;
                                }
                            }
                            Err(_) => break 'thread,
                        }
                    }
                }
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
                let Some(first) = maybe_input else { break Ok(()) };
                dirty = true;
                // On the legacy Windows console a paste arrives as a rapid
                // burst of key events (bracketed paste is a no-op), so an
                // embedded newline is an `Enter` that would submit the entry
                // and leak the rest of the paste into the global key handler.
                // Coalesce the burst into a single paste instead.
                let (input, extra) = coalesce_paste_burst(&mut rx_input, first).await;
                // The paste is one atomic event, so nothing from it leaks into
                // the global key handler; the burst's non-text events (mouse,
                // resize, focus, ...) are handled right after so they are not
                // dropped.
                handle_terminal_input(input, &mut state, &tx_cmd, editor_width, &mut dirty)
                    .await?;
                for ev in extra {
                    handle_terminal_input(ev, &mut state, &tx_cmd, editor_width, &mut dirty)
                        .await?;
                }
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
                        state.agent_finished = true;
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
            if !state.reasoning_stream_started && !content.is_empty() {
                append_reasoning_log(state, &content);
            }
            state.streaming_buffer = truncate_stream_preview(&content, 8000);
            state.model_snippet = state.streaming_buffer.clone();
            state.reasoning_entry_active = false;
            state.reasoning_entry_chars = 0;
            state.reasoning_stream_started = false;
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
                updated_at: Instant::now(),
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
            // Keep the previous completed output visible in the preview. The
            // transcript holds the full scrollback across retries and tools.
            state.reasoning_entry_active = false;
            state.reasoning_entry_chars = 0;
            state.reasoning_stream_started = false;
        }
        UiEvent::ReasoningChunk { delta } => {
            state.streaming_buffer.push_str(&delta);
            if state.streaming_buffer.len() > 8000 {
                let target = state.streaming_buffer.len() - 8000;
                let start = state.streaming_buffer.floor_char_boundary(target);
                state.streaming_buffer.drain(..start);
            }
            state.model_snippet = state.streaming_buffer.clone();
            if !delta.is_empty() {
                append_reasoning_log(state, &delta);
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

    /// How the model must pick a tool each agent turn: `required` forces a
    /// call, `auto` lets it answer in prose. `required` avoids narration but
    /// misbehaves on some Qwen3 reasoning templates; switch to `auto` if the
    /// model stalls at the token limit without calling anything.
    #[clap(long, default_value = "required")]
    tool_choice: String,

    /// Maximum iterations before stopping (defaults high enough for overnight runs).
    #[clap(long, default_value_t = 1_000_000)]
    max_iterations: usize,

    /// Maximum wall-clock runtime in seconds.
    #[clap(long, default_value_t = 360000)]
    max_wall_secs: u64,

    /// Timeout for a single model request, in seconds (minimum and default: 15 minutes).
    #[clap(long, default_value_t = 900)]
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
    #[clap(long, default_value_t = 70)] // changed from 85 to 70
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

    let model_timeout_secs = config.model_timeout_secs.max(MIN_MODEL_TIMEOUT_SECS);
    if config.model_timeout_secs < MIN_MODEL_TIMEOUT_SECS {
        eprintln!(
            "Model timeout raised to the 15-minute minimum ({}s).",
            MIN_MODEL_TIMEOUT_SECS
        );
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(model_timeout_secs.saturating_add(30)))
        .connect_timeout(Duration::from_secs(15))
        .build()?;

    let model = Model {
        client,
        base_url,
        model: model_name,
        temperature: 0.7,
        api_key,
        request_timeout_secs: model_timeout_secs,
        reasoning_effort: config.reasoning_effort.clone(),
        tool_choice: config.tool_choice.clone(),
    };

    warn_if_tools_unsupported(&model).await;

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

    // `load_session` prefers the workdir saved in session.json so switching
    // sessions restores their own directory. An explicit CLI override should
    // take precedence over that saved value.
    if session_exists && config.workdir.is_some() {
        session.workdir = workdir.clone();
        update_session_info(&session_id, Some(&session.goal), None, Some(&workdir)).await?;
    }

    if session_exists {
        if let Some(goal) = config.goal.clone() {
            let goal = goal.trim().to_string();
            if !goal.is_empty() && goal != session.goal {
                atomic_write(&session_dir(&session_id).join("goal.txt"), &goal).await?;
                update_session_info(&session_id, Some(&goal), None, Some(&workdir)).await?;
                session.goal = goal.clone();
                session
                    .push_message(Message::new(
                        "user",
                        format!("The user changed the goal to: {goal}"),
                    ))
                    .await?;
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
            let mut last_flush = Instant::now();
            let mut wrote_model_output = false;
            while let Some(event) = rx_ui.recv().await {
                let flush_now = !matches!(&event, UiEvent::ReasoningChunk { .. });
                if wrote_model_output && !matches!(&event, UiEvent::ReasoningChunk { .. }) {
                    println!();
                    wrote_model_output = false;
                }
                match event {
                    UiEvent::Log(line) => println!("[log] {line}"),
                    // The response has already been printed incrementally by
                    // ReasoningChunk; printing the aggregate here duplicated
                    // every completed model response in no-TUI mode.
                    UiEvent::Reasoning { .. } => {}
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
                        wrote_model_output |= !delta.is_empty();
                    }
                    UiEvent::Quit => break,
                }
                if flush_now || last_flush.elapsed() >= Duration::from_millis(100) {
                    let _ = io::stdout().flush();
                    last_flush = Instant::now();
                }
            }
            let _ = io::stdout().flush();
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

        let _ = printer.await;
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
mod tests;
