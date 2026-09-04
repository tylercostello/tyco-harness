//! Unit tests for the harness.
//!
//! Kept out of `main.rs` so the agent logic stays readable; this is a child
//! module of the crate root, so `use super::*` reaches the private internals
//! that these tests exercise.

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
    assert!(response.tool_calls.is_empty());
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
fn tools_are_sent_only_on_agent_turns() {
    let model = |tool_choice: &str| Model {
        client: reqwest::Client::new(),
        base_url: "http://localhost/v1".into(),
        model: "m".into(),
        temperature: 0.7,
        api_key: None,
        request_timeout_secs: 30,
        reasoning_effort: None,
        tool_choice: tool_choice.into(),
    };
    let messages = [Message::new("user", "hi")];

    let agent = model("required").request_body(&messages, true, RequestKind::Agent);
    assert!(agent["tools"].is_array());
    assert_eq!(agent["tool_choice"], "required");

    // The configured choice is passed through verbatim.
    let auto = model("auto").request_body(&messages, true, RequestKind::Agent);
    assert_eq!(auto["tool_choice"], "auto");

    // Summarization must stay tool-free or it gets answered with a call.
    let plain = model("required").request_body(&messages, true, RequestKind::PlainText);
    assert!(plain["tools"].is_null());
    assert!(plain["tool_choice"].is_null());
}
#[test]
fn vision_tools_are_registered() {
    let tools = super::tools::tool_definitions();
    let names: Vec<&str> = tools
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["function"]["name"].as_str())
        .collect();
    for expected in ["screenshot", "view_image", "render_page"] {
        assert!(names.contains(&expected), "{expected} is not registered");
    }
}

#[test]
fn native_wire_message_exposes_image_content() {
    let mut message = Message::new("user", "look");
    message.image = Some("aW1hZ2U=".into());
    let wire = to_native_wire_message(&message);
    assert!(wire.get("image").is_none());
    assert!(wire["content"].is_array());
    assert_eq!(
        wire["content"][1]["image_url"]["url"],
        "data:image/png;base64,aW1hZ2U="
    );
}

/// Returns `(workdir, image_path)`.
fn png_fixture(name: &str, width: u32, height: u32) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("tyco-test-{}-{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::RgbaImage::from_pixel(width, height, image::Rgba([10, 20, 30, 255]))
        .save(&path)
        .unwrap();
    (dir, path)
}

fn decode_attached(outcome: &super::tools::ToolOutcome) -> image::DynamicImage {
    use base64::Engine;
    let encoded = outcome.image.as_ref().expect("expected an attached image");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("attachment should be valid base64");
    image::load_from_memory(&bytes).expect("attachment should be a decodable image")
}

#[test]
fn view_image_round_trips_a_real_png() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (dir, _path) = png_fixture("roundtrip", 64, 48);

    let outcome = rt
        .block_on(media::load_image_file(&dir, "roundtrip.png"))
        .unwrap();

    let decoded = decode_attached(&outcome);
    assert_eq!((decoded.width(), decoded.height()), (64, 48));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn oversized_images_are_downscaled_before_they_reach_the_model() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (dir, _path) = png_fixture("oversized", 4000, 2000);

    let outcome = rt
        .block_on(media::load_image_file(&dir, "oversized.png"))
        .unwrap();

    let decoded = decode_attached(&outcome);
    assert!(
        decoded.width().max(decoded.height()) <= 1280,
        "image was not downscaled: {}x{}",
        decoded.width(),
        decoded.height()
    );
    assert!(
        outcome.text.contains("downscaled from 4000x2000"),
        "downscaling should be reported to the model, got: {}",
        outcome.text
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn undecodable_image_reports_an_error_instead_of_attaching_garbage() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = std::env::temp_dir().join(format!("tyco-test-garbage-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("bad.png"), b"\x89PNG\r\n\x1a\nfake").unwrap();

    let result = rt.block_on(media::load_image_file(&dir, "bad.png"));

    assert!(result.is_err(), "a corrupt image must not be attached");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn image_tools_never_escape_the_workdir() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (dir, _path) = png_fixture("escape", 8, 8);

    let result = rt.block_on(media::load_image_file(&dir, "../../../etc/hosts"));

    assert!(result.is_err(), "path traversal must be rejected");
    std::fs::remove_dir_all(&dir).ok();
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

#[test]
fn an_image_is_never_attached_to_a_tool_result() {
    let outcome = super::tools::ToolOutcome {
        text: "captured".into(),
        image: Some("aW1n".into()),
    };

    let messages = tool_result_messages(Some("call_1"), "screenshot", outcome);

    assert!(
        messages
            .iter()
            .all(|m| m.role != "tool" || m.image.is_none()),
        "an image on a tool result is rejected at tokenize time"
    );
    let carriers: Vec<&Message> = messages.iter().filter(|m| m.image.is_some()).collect();
    assert_eq!(carriers.len(), 1);
    assert_eq!(carriers[0].role, "user");
    assert_eq!(messages[0].role, "tool");
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn a_textual_tool_result_stays_a_single_message() {
    let outcome = super::tools::ToolOutcome::plain("exit code: 0".into());

    let messages = tool_result_messages(Some("call_1"), "run_command", outcome);

    assert_eq!(messages.len(), 1);
    assert!(messages[0].image.is_none());
}

#[test]
fn stripping_images_clears_every_attachment() {
    let mut messages = vec![
        Message::new("user", "hi"),
        Message::with_image("user", "a", "aW1n".into(), None),
        Message::new("assistant", "ok"),
        Message::with_image("user", "b", "aW1n".into(), None),
    ];

    let dropped = strip_images(&mut messages);

    assert_eq!(dropped, 2);
    assert!(messages.iter().all(|m| m.image.is_none()));
    assert_eq!(messages.len(), 4, "stripping must not remove messages");
}

#[test]
fn rolling_back_always_shortens_history_so_the_loop_terminates() {
    let mut messages = vec![
        Message::new("system", "s"),
        Message::new("user", "first"),
        Message::new("assistant", "reply"),
        Message::tool_result("call_1", "result"),
        Message::new("user", "second"),
    ];

    let mut previous = messages.len();
    while rollback_last_turn(&mut messages) {
        assert!(
            messages.len() < previous,
            "rollback must shrink history or the recovery ladder spins"
        );
        previous = messages.len();
    }
    assert!(!messages.is_empty(), "rollback must not empty the history");
}

#[test]
fn rollback_refuses_when_only_the_opening_turn_remains() {
    let mut messages = vec![Message::new("user", "only")];

    assert!(!rollback_last_turn(&mut messages));
    assert_eq!(messages.len(), 1);
}

#[test]
fn server_faults_are_retryable_but_rejected_requests_are_not() {
    use reqwest::StatusCode;

    assert!(matches!(
        classify_status(StatusCode::INTERNAL_SERVER_ERROR, "boom"),
        ChatError::Transient(_)
    ));
    assert!(matches!(
        classify_status(StatusCode::TOO_MANY_REQUESTS, "slow down"),
        ChatError::Transient(_)
    ));
    assert!(matches!(
        classify_status(StatusCode::BAD_REQUEST, "tools param requires --jinja flag"),
        ChatError::BadRequest(_)
    ));
    assert!(matches!(
        classify_status(StatusCode::UNAUTHORIZED, "nope"),
        ChatError::Auth(_)
    ));
}

#[test]
fn every_declared_tool_can_be_dispatched() {
    use super::{tools, ToolCall};

    let names: Vec<String> = tools::tool_definitions()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = std::env::temp_dir();
    for name in names {
        let call = ToolCall {
            name: name.clone(),
            arguments: serde_json::json!({}),
        };
        // A registered tool must not report itself unknown; missing arguments
        // may legitimately error, but "unknown tool" means schema and dispatch
        // have drifted apart.
        if let Err(e) = rt.block_on(tools::execute_tool(&call, &dir, &dir, 1)) {
            assert!(
                !e.to_string().contains("unknown tool"),
                "{name} is declared but has no dispatch arm"
            );
        }
    }
}

// ============================================================
// Context overflow recovery
// ============================================================

#[test]
fn extract_context_overflow_parses_llamacpp_shape() {
    let body = "HTTP 400 Bad Request: {\"error\":{\"code\":400,\"message\":\"request (110321 tokens) exceeds the available context size (110080 tokens), try increasing it\",\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":110321,\"n_ctx\":110080}}";
    let ov = extract_context_overflow(body);
    assert_eq!(ov.prompt_tokens, Some(110321));
    assert_eq!(ov.n_ctx, Some(110080));
    assert!(ov.is_oversized());
}

#[test]
fn extract_context_overflow_parses_openai_shape() {
    let body = r#"{"usage":{"prompt_tokens":150000},"context_length":128000}"#;
    let ov = extract_context_overflow(body);
    assert_eq!(ov.prompt_tokens, Some(150000));
    assert_eq!(ov.n_ctx, Some(128000));
    assert!(ov.is_oversized());
}

#[test]
fn extract_context_overflow_flags_only_real_overflows() {
    let small = extract_context_overflow(r#"{"n_prompt_tokens":1000,"n_ctx":110080}"#);
    assert_eq!(small.prompt_tokens, Some(1000));
    assert!(!small.is_oversized());
    let garbage = extract_context_overflow("HTTP 500: internal error");
    assert!(!garbage.is_oversized());
}

#[test]
fn fit_to_context_shrinks_history_to_fit_the_window() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = std::env::temp_dir().join(format!("tyco-fit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Twenty turns of a few thousand chars each: far over the window once the
    // server's (larger-than-local) token count is applied.
    let mut messages = Vec::new();
    for i in 0..20u32 {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        messages.push(Message::new(role, format!("m{i}").repeat(500)));
    }
    let mut session = Session {
        id: "fit-test".into(),
        goal: "g".into(),
        scratchpad: String::new(),
        messages,
        transcript_path: dir.join("transcript.jsonl"),
        todo_path: dir.join("todo.md"),
        workdir: dir.clone(),
        context_tokens: 110080,
        compaction_threshold: 90,
        todo_cache: String::new(),
    };
    let before = session.messages.len();
    let ui = UiLogger::disabled();
    let overflow = ContextOverflow { prompt_tokens: Some(200_000), n_ctx: Some(110_080) };
    let shrank = rt.block_on(fit_to_context(&mut session, &ui, &overflow));
    assert!(shrank, "fit_to_context must report a shrink");
    assert!(
        session.messages.len() >= 2 && session.messages.len() < before,
        "history must shrink but never empty: {} -> {}",
        before,
        session.messages.len()
    );
    // The oldest turns went; the newest one survived.
    assert!(!session.messages.first().unwrap().content.starts_with("m0"));
    assert!(session.messages.last().unwrap().content.starts_with("m19"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fit_to_context_is_a_noop_when_the_request_already_fits() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = std::env::temp_dir().join(format!("tyco-fit2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut session = Session {
        id: "fit-test".into(),
        goal: "g".into(),
        scratchpad: String::new(),
        messages: vec![Message::new("user", "hi"), Message::new("assistant", "hello")],
        transcript_path: dir.join("transcript.jsonl"),
        todo_path: dir.join("todo.md"),
        workdir: dir.clone(),
        context_tokens: 110080,
        compaction_threshold: 90,
        todo_cache: String::new(),
    };
    let before = session.messages.len();
    let ui = UiLogger::disabled();
    let overflow = ContextOverflow { prompt_tokens: Some(50_000), n_ctx: Some(110_080) };
    let shrank = rt.block_on(fit_to_context(&mut session, &ui, &overflow));
    assert!(!shrank);
    assert_eq!(session.messages.len(), before);
    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================
// Paste coalescing
// ============================================================

#[test]
fn split_burst_keeps_text_and_returns_non_text_events() {
    use crossterm::event::{KeyEvent, KeyModifiers, KeyCode, MouseButton, MouseEvent, MouseEventKind};
    let mouse = MouseEvent {
        kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
        column: 1,
        row: 2,
        modifiers: KeyModifiers::NONE,
    };
    let events = vec![
        crossterm::event::Event::Key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE)),
        crossterm::event::Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        crossterm::event::Event::Mouse(mouse.clone()),
        crossterm::event::Event::Key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE)),
        crossterm::event::Event::Key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE)),
    ];
    let (text, extra) = split_burst(&events);
    assert_eq!(text, "a\nbt");
    assert_eq!(extra.len(), 1);
    assert!(matches!(&extra[0], crossterm::event::Event::Mouse(_)));
}

#[test]
fn control_keys_are_not_paste_text() {
    use crossterm::event::{KeyEvent, KeyModifiers, KeyCode};
    let ctrl_c = crossterm::event::Event::Key(KeyEvent::new(
        KeyCode::Char('c'),
        KeyModifiers::CONTROL,
    ));
    assert!(!is_paste_text(&ctrl_c));
    let plain = crossterm::event::Event::Key(KeyEvent::new(
        KeyCode::Char('c'),
        KeyModifiers::NONE,
    ));
    assert!(is_paste_text(&plain));
}

#[test]
fn coalesce_paste_burst_folds_a_multi_line_paste_into_one_event() {
    use crossterm::event::{KeyEvent, KeyModifiers, KeyCode};
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<crossterm::event::Event>();
        let first = crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Char('h'),
            KeyModifiers::NONE,
        ));
        tx.send(crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Char('i'),
            KeyModifiers::NONE,
        )))
        .unwrap();
        tx.send(crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();
        tx.send(crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Char('!'),
            KeyModifiers::NONE,
        )))
        .unwrap();
        let (ev, extra) = coalesce_paste_burst(&mut rx, first).await;
        match ev {
            crossterm::event::Event::Paste(text) => assert_eq!(text, "hi\n!"),
            other => panic!("expected a paste, got {other:?}"),
        }
        assert!(extra.is_empty());
    });
}

#[test]
fn coalesce_paste_burst_returns_non_text_tail_events() {
    // The "extra event isn't more paste text" case: a non-text event queued
    // right after the first char must be returned for handling, not swallowed.
    use crossterm::event::{KeyEvent, KeyModifiers, KeyCode, MouseButton, MouseEvent, MouseEventKind};
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<crossterm::event::Event>();
        let first = crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Char('a'),
            KeyModifiers::NONE,
        ));
        let mouse = crossterm::event::Event::Mouse(MouseEvent {
            kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        tx.send(mouse).unwrap();
        let (ev, extra) = coalesce_paste_burst(&mut rx, first).await;
        match ev {
            crossterm::event::Event::Paste(text) => assert_eq!(text, "a"),
            other => panic!("expected a paste, got {other:?}"),
        }
        assert_eq!(extra.len(), 1);
        assert!(matches!(&extra[0], crossterm::event::Event::Mouse(_)));
    });
}

#[test]
fn coalesce_paste_burst_keeps_a_lone_keystroke() {
    use crossterm::event::{KeyEvent, KeyModifiers, KeyCode};
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (_tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<crossterm::event::Event>();
        let first = crossterm::event::Event::Key(KeyEvent::new(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
        ));
        let (ev, extra) = coalesce_paste_burst(&mut rx, first).await;
        match ev {
            crossterm::event::Event::Key(k) => assert_eq!(k.code, KeyCode::Char('q')),
            other => panic!("expected the keystroke back, got {other:?}"),
        }
        assert!(extra.is_empty());
    });
}
