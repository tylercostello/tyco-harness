//! Tool schemas and their execution.
//!
//! Schema and dispatch live in one file on purpose: a tool added to
//! [`tool_definitions`] without a matching arm in [`execute_tool`] (or the
//! reverse) is a bug that is easy to spot when the two sit together.

use anyhow::{anyhow, Context, Result};
use std::io;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use crate::{media, safe_path, search, truncate, ToolCall};

#[derive(Clone, Default)]
pub(crate) struct ToolOutcome {
    pub(crate) text: String,
    pub(crate) image: Option<String>,
}

impl ToolOutcome {
    pub(crate) fn plain(text: String) -> Self {
        Self { text, image: None }
    }
}

// ============================================================
// Tool Execution
// ============================================================

pub(crate) async fn execute_tool(
    call: &ToolCall,
    workdir: &Path,
    todo_path: &Path,
    command_timeout_secs: u64,
) -> Result<ToolOutcome> {
    match call.name.as_str() {
        "read_file" => {
            let p = call.arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow!("missing path"))?;
            let full = safe_path(workdir, p)?;
            let content = tokio::fs::read_to_string(&full).await?;
            Ok(ToolOutcome::plain(truncate(&content, 16000)))
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
            Ok(ToolOutcome::plain(format!(
                "wrote {} ({} bytes)",
                full.display(),
                content.len()
            )))
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
            Ok(ToolOutcome::plain(truncate(&out, 4000)))
        }
        "run_command" => {
            let cmd = call.arguments["command"]
                .as_str()
                .ok_or_else(|| anyhow!("missing command"))?;
            run_shell_command(cmd, workdir, command_timeout_secs)
                .await
                .map(ToolOutcome::plain)
        }
        "update_todo" => {
            let content = call.arguments["content"]
                .as_str()
                .ok_or_else(|| anyhow!("missing content"))?;
            tokio::fs::write(todo_path, content).await?;
            Ok(ToolOutcome::plain(format!("todo list updated:\n{content}")))
        }
        "get_todo" => match tokio::fs::read_to_string(todo_path).await {
            Ok(content) if !content.trim().is_empty() => Ok(ToolOutcome::plain(content)),
            _ => Ok(ToolOutcome::plain("No todo list yet.".to_string())),
        },
        "search_web" => {
            let query = call.arguments["query"]
                .as_str()
                .ok_or_else(|| anyhow!("missing query"))?;
            search::search_web(query).await.map(ToolOutcome::plain)
        }
        // Normally intercepted by the agent loop before dispatch.
        "finish" => {
            let reason = call.arguments["reason"].as_str().unwrap_or("done");
            Ok(ToolOutcome::plain(format!("finish: {reason}")))
        }
        "screenshot" => media::capture_screen().await,
        "view_image" => {
            let p = call.arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow!("missing path"))?;
            media::load_image_file(workdir, p).await
        }
        "render_page" => {
            let target = call.arguments["target"]
                .as_str()
                .ok_or_else(|| anyhow!("missing target"))?;
            let full_page = call.arguments["full_page"].as_bool().unwrap_or(true);
            media::render_page(workdir, target, full_page).await
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

pub(crate) fn tool_definitions() -> serde_json::Value {
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
        spec(
            "screenshot",
            "Capture the user's primary monitor and attach it as visual input.",
            serde_json::json!({}),
            &[]
        ),
        spec(
            "view_image",
            "Load an image file (png/jpeg/webp) from the working directory and attach it as visual input.",
            serde_json::json!({ "path": string_prop("Relative path to the image file.") }),
            &["path"]
        ),
        spec(
            "render_page",
            "Render a local HTML file or URL in an offscreen browser and attach the result as visual input. No window appears, so the user's desktop is undisturbed. Use this to look at web pages you are building.",
            serde_json::json!({
                "target": string_prop("Relative path to an .html file, or an http(s) URL."),
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the entire scrollable page rather than just the viewport. Defaults to true.",
                },
            }),
            &["target"]
        ),
    ])
}
