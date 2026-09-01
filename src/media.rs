//! Image capture, rendering and encoding for the vision tools.
//!
//! Every path here funnels through [`encode_image`], which is the single place
//! that enforces the size budget. Nothing else in the harness is allowed to
//! build a base64 image, so an oversized payload cannot reach the model.

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::{Browser, BrowserConfig};
use futures::StreamExt;
use image::{DynamicImage, ImageFormat};
use std::path::Path;
use std::time::Duration;

use crate::safe_path;
use crate::tools::ToolOutcome;

/// Vision models bill images against a pixel budget, so resolution past this
/// costs context without adding detail the model can actually resolve.
const MAX_IMAGE_DIM: u32 = 1280;

/// Hard ceiling on the encoded payload. Unbounded images are what turn a slow
/// turn into a request the server rejects outright.
const MAX_IMAGE_B64_CHARS: usize = 6 * 1024 * 1024;

const BROWSER_TIMEOUT: Duration = Duration::from_secs(60);

fn encode_image(image: DynamicImage, label: &str) -> Result<ToolOutcome> {
    let (source_w, source_h) = (image.width(), image.height());
    let image = if source_w.max(source_h) > MAX_IMAGE_DIM {
        image.resize(
            MAX_IMAGE_DIM,
            MAX_IMAGE_DIM,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image
    };
    let (width, height) = (image.width(), image.height());

    let mut png = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut png, ImageFormat::Png)
        .with_context(|| format!("failed to encode {label} as PNG"))?;
    let png = png.into_inner();

    let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
    if encoded.len() > MAX_IMAGE_B64_CHARS {
        return Err(anyhow!(
            "{label} is {} base64 chars, over the {MAX_IMAGE_B64_CHARS} limit",
            encoded.len()
        ));
    }

    let kb = png.len() / 1024;
    let text = if (width, height) == (source_w, source_h) {
        format!("{label}: {width}x{height} PNG, {kb} KB")
    } else {
        format!("{label}: {width}x{height} PNG, {kb} KB (downscaled from {source_w}x{source_h})")
    };
    Ok(ToolOutcome {
        text,
        image: Some(encoded),
    })
}

pub async fn capture_screen() -> Result<ToolOutcome> {
    let frame = tokio::task::spawn_blocking(|| -> Result<image::RgbaImage> {
        let monitors =
            xcap::Monitor::all().map_err(|e| anyhow!("cannot enumerate monitors: {e}"))?;
        let monitor = monitors
            .iter()
            .find(|m| m.is_primary().unwrap_or(false))
            .or_else(|| monitors.first())
            .ok_or_else(|| anyhow!("no monitor available to capture"))?;
        monitor
            .capture_image()
            .map_err(|e| anyhow!("screen capture failed: {e}"))
    })
    .await
    .context("screen capture task panicked")??;

    encode_image(DynamicImage::ImageRgba8(frame), "screen")
}

pub async fn load_image_file(workdir: &Path, path: &str) -> Result<ToolOutcome> {
    let full = safe_path(workdir, path)?;
    let bytes = tokio::fs::read(&full)
        .await
        .with_context(|| format!("cannot read {}", full.display()))?;
    if bytes.is_empty() {
        return Err(anyhow!("image file is empty: {}", full.display()));
    }
    let image = image::load_from_memory(&bytes)
        .with_context(|| format!("{} is not a decodable image", full.display()))?;
    encode_image(image, &format!("image {}", full.display()))
}

/// Builds a URL the browser can open. Windows paths need backslashes swapped
/// and an extra leading slash, or `file://C:/x` is read as host `C:`.
fn page_url(workdir: &Path, target: &str) -> Result<String> {
    if target.starts_with("http://") || target.starts_with("https://") {
        return Ok(target.to_string());
    }
    let full = safe_path(workdir, target)?;
    if !full.exists() {
        return Err(anyhow!("no such file to render: {}", full.display()));
    }
    let path = full.to_string_lossy().replace('\\', "/");
    Ok(if path.starts_with('/') {
        format!("file://{path}")
    } else {
        format!("file:///{path}")
    })
}

/// Renders a page offscreen and returns it as an image.
///
/// Uses Chrome's *new* headless mode, which never creates a window, so the
/// user's desktop stays untouched while the agent inspects its own work.
pub async fn render_page(workdir: &Path, target: &str, full_page: bool) -> Result<ToolOutcome> {
    let url = page_url(workdir, target)?;
    let png = tokio::time::timeout(BROWSER_TIMEOUT, screenshot_url(&url, full_page))
        .await
        .map_err(|_| anyhow!("rendering {url} timed out after {BROWSER_TIMEOUT:?}"))??;

    let image = image::load_from_memory(&png).context("browser returned an undecodable image")?;
    encode_image(image, &format!("rendered {target}"))
}

async fn screenshot_url(url: &str, full_page: bool) -> Result<Vec<u8>> {
    let config = BrowserConfig::builder()
        .new_headless_mode()
        .window_size(1440, 900)
        .build()
        .map_err(|e| anyhow!("invalid browser config: {e}"))?;

    let (mut browser, mut handler) = Browser::launch(config)
        .await
        .context("could not launch a headless browser; install Google Chrome or Microsoft Edge")?;

    // The handler future drives the CDP connection; without it polling, every
    // page call below would hang rather than fail.
    let pump = tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if event.is_err() {
                break;
            }
        }
    });

    let result = async {
        let page = browser.new_page(url).await?;
        page.wait_for_navigation().await?;
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(full_page)
            .build();
        let png = page.screenshot(params).await?;
        Ok::<_, anyhow::Error>(png)
    }
    .await;

    let _ = browser.close().await;
    pump.abort();
    result
}
