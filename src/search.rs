//! DuckDuckGo HTML-endpoint web search.
//!
//! The HTML scrape is deliberate: it needs no API key. The scrape is pinned to
//! DuckDuckGo's `result__a` / `result__snippet` markup, so a redesign there is
//! what breaks this, not a bug here.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::{truncate, HTTP};

static RESULT_A_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<a([^>]*class="[^"]*result__a[^"]*"[^>]*)>(.*?)</a>"#).unwrap());
static SNIPPET_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>"#).unwrap()
});
static HREF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"href="([^"]*)""#).unwrap());
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<[^>]*>").unwrap());

pub async fn search_web(query: &str) -> Result<String> {
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
