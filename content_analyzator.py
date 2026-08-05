"""Analyze exported Telegram message JSON with local heuristics and an optional local LLM.

The script summarizes topics, finds high-traction posts, identifies active
members, detects repeated/reposted messages, and extracts non-Telegram links.
It intentionally avoids psychological or clinical profiling of individuals.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
import textwrap
import urllib.parse
from pathlib import Path
from typing import Any

import ollama


URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
TELEGRAM_HOST_RE = re.compile(r"(^|\.)t\.me$|(^|\.)telegram\.me$|(^|\.)telegram\.org$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
MENTION_RE = re.compile(r"@\w+")


@dataclasses.dataclass(slots=True)
class MessageRecord:
    id: Any
    source_name: str
    date: str
    sender_id: Any
    text: str
    reply_to_msg_id: Any
    views: int
    forwards: int
    reactions: int
    links: list[str]
    normalized_text: str
    fingerprint: str
    traction_score: float


def load_messages(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("messages", "items", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported JSON structure. Expected a list of message objects.")


def discover_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        files = [item for item in sorted(path.glob("*.json")) if item.is_file()]
        files = [item for item in files if not item.name.endswith(".llm_payload.json") and not item.name.endswith(".analysis.json")]
        return files

    raise FileNotFoundError(f"Input path not found: {path}")


def source_name_for_path(path: Path) -> str:
    name = path.stem
    if name.startswith("messages_"):
        return name[len("messages_"):]
    return name


def telegram_channel_url(source_name: str) -> str:
    normalized = str(source_name or "").strip().lstrip("@")
    if not normalized:
        return ""
    return f"https://t.me/{normalized}"


def telegram_post_url(source_name: str, post_id: Any) -> str:
    channel_url = telegram_channel_url(source_name)
    if not channel_url:
        return ""
    if post_id in (None, "", 0):
        return channel_url
    return f"{channel_url}/{post_id}"


def as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_text(message: dict[str, Any]) -> str:
    text = message.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: list[str] = []
        for item in text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return ""


def extract_links(text: str) -> list[str]:
    links: list[str] = []
    for match in URL_RE.findall(text or ""):
        cleaned = match.rstrip(".,;:!?)]}\"'")
        parsed = urllib.parse.urlparse(cleaned)
        host = parsed.netloc.lower()
        if TELEGRAM_HOST_RE.search(host):
            continue
        links.append(cleaned)
    return list(dict.fromkeys(links))


def normalize_text(text: str) -> str:
    cleaned = (text or "").lower()
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u200b", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def fingerprint_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def reaction_count(message: dict[str, Any]) -> int:
    reactions = message.get("reactions")
    if isinstance(reactions, dict):
        if "results" in reactions and isinstance(reactions["results"], list):
            return sum(as_int(result.get("count")) for result in reactions["results"] if isinstance(result, dict))
        if "count" in reactions:
            return as_int(reactions.get("count"))
    if isinstance(reactions, list):
        return sum(as_int(item.get("count")) for item in reactions if isinstance(item, dict))
    return 0


def traction_score(message: dict[str, Any]) -> float:
    views = as_int(message.get("views"))
    forwards = as_int(message.get("forwards"))
    replies = 1 if message.get("reply_to_msg_id") not in (None, 0, "") else 0
    reactions = reaction_count(message)
    return float(views + (3 * forwards) + (2 * replies) + reactions)


def build_records(messages: list[dict[str, Any]], source_name: str) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for message in messages:
        text = extract_text(message)
        records.append(
            MessageRecord(
                id=message.get("id"),
                source_name=source_name,
                date=str(message.get("date", "")),
                sender_id=message.get("sender_id"),
                text=text,
                reply_to_msg_id=message.get("reply_to_msg_id"),
                views=as_int(message.get("views")),
                forwards=as_int(message.get("forwards")),
                reactions=reaction_count(message),
                links=extract_links(text),
                normalized_text=normalize_text(text),
                fingerprint=fingerprint_text(text),
                traction_score=traction_score(message),
            )
        )
    return records


def summarize_source_summary(records: list[MessageRecord], source_name: str) -> dict[str, Any]:
    source_summary = summarize_records(records)
    return {
        "source_name": source_name,
        "message_count": source_summary["message_count"],
        "unique_senders": source_summary["unique_senders"],
        "top_posts": source_summary["top_posts"][:15],
        "repeated_messages": source_summary["repeated_messages"][:15],
        "non_telegram_links": source_summary["non_telegram_links"][:10],
    }


def summarize_records(records: list[MessageRecord]) -> dict[str, Any]:
    by_sender = collections.Counter()
    by_day = collections.Counter()
    by_source = collections.Counter()
    duplicate_groups: dict[str, list[MessageRecord]] = collections.defaultdict(list)
    all_links: list[str] = []

    for record in records:
        by_source[record.source_name] += 1
        if record.sender_id is not None:
            by_sender[str(record.sender_id)] += 1
        if record.date:
            by_day[record.date[:10]] += 1
        if record.fingerprint:
            duplicate_groups[record.fingerprint].append(record)
        all_links.extend(record.links)

    repeated_messages = []
    for group in duplicate_groups.values():
        if len(group) > 1:
            sample_record = next((item for item in group if item.text.strip()), group[0])
            repeated_messages.append(
                {
                    "count": len(group),
                    "message_ids": [item.id for item in group],
                    "sender_ids": [item.sender_id for item in group],
                    "sample_text": textwrap.shorten(sample_record.text or "[no text]", width=300, placeholder="..."),
                }
            )
    repeated_messages.sort(key=lambda item: item["count"], reverse=True)

    top_posts = sorted(records, key=lambda item: item.traction_score, reverse=True)[:20]
    top_members = by_sender.most_common(20)
    top_sources = by_source.most_common(20)

    unique_links = list(dict.fromkeys(all_links))
    non_telegram_links = unique_links

    cross_source_duplicates = []
    for group in duplicate_groups.values():
        if len(group) > 1:
            source_names = sorted({item.source_name for item in group})
            if len(source_names) > 1:
                sample_record = next((item for item in group if item.text.strip()), group[0])
                cross_source_duplicates.append(
                    {
                        "count": len(group),
                        "source_names": source_names,
                        "message_ids": [item.id for item in group],
                        "sample_text": textwrap.shorten(sample_record.text or "[no text]", width=300, placeholder="..."),
                    }
                )
    cross_source_duplicates.sort(key=lambda item: item["count"], reverse=True)

    link_sources: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        for link in record.links:
            link_sources[link].add(record.source_name)

    shared_links = [
        {
            "link": link,
            "source_names": sorted(source_names),
            "source_count": len(source_names),
        }
        for link, source_names in link_sources.items()
        if len(source_names) > 1
    ]
    shared_links.sort(key=lambda item: item["source_count"], reverse=True)

    return {
        "message_count": len(records),
        "unique_senders": len(by_sender),
        "sources": [
            {"source_name": source_name, "message_count": count}
            for source_name, count in top_sources
        ],
        "messages_by_day": dict(by_day.most_common(30)),
        "top_members": [
            {"sender_id": sender_id, "message_count": count}
            for sender_id, count in top_members
        ],
        "top_posts": [
            {
                "id": record.id,
                "source_name": record.source_name,
                "date": record.date,
                "sender_id": record.sender_id,
                "views": record.views,
                "forwards": record.forwards,
                "reactions": record.reactions,
                "traction_score": record.traction_score,
                "text": record.text[:1000],
                "links": record.links,
            }
            for record in top_posts
        ],
        "repeated_messages": repeated_messages[:20],
        "non_telegram_links": non_telegram_links,
        "cross_source_duplicates": cross_source_duplicates[:20],
        "shared_links": shared_links[:20],
    }


def build_llm_payload(records: list[MessageRecord], summary: dict[str, Any], sample_limit: int) -> dict[str, Any]:
    sample_messages = [
        {
            "id": record.id,
            "source_name": record.source_name,
            "date": record.date,
            "sender_id": record.sender_id,
            "views": record.views,
            "forwards": record.forwards,
            "reactions": record.reactions,
            "text": record.text[:1000],
            "links": record.links,
            "traction_score": record.traction_score,
        }
        for record in sorted(records, key=lambda item: item.traction_score, reverse=True)[:sample_limit]
    ]

    return {
        "summary": summary,
        "sample_messages": sample_messages,
        "source_summaries": summary.get("source_summaries", []),
        "instructions": [
            "Summarize the main topics across all Telegram exports in this folder.",
            "Highlight shared content patterns, recurring themes, and high-traction posts.",
            "Compare channels/groups and explain commonalities, overlaps, and differences.",
            "Identify repeated stories, shared links, or identical posts that appear across multiple sources.",
            "Describe which channels appear to amplify the same narratives, sources, or links.",
            "Point out any clusters of channels/groups that cover the same story from similar angles.",
            "Describe communication patterns for active members.",
            "Return only valid JSON with keys: topics, summary, traction_insights, active_member_patterns, repeated_content_observations, caveats.",
            "Disregard Good morning, Hello, Hi, and similar greetings in the analysis.",
            "Ignore greetings, sign-offs, and non-substantive messages when identifying repeated content.",
            "Be descriptive with the topics in deeper llm analysis.",
        ],
    }


def call_ollama_llm(model: str, payload: dict[str, Any]) -> str:
    try:
        response = ollama.chat(
            model=model,
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful analyst. Return only valid JSON, with no markdown, no code fences, and no extra commentary.",
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                }
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc

    message: Any = None
    if hasattr(response, "message"):
        message = response.message
    elif isinstance(response, dict):
        message = response.get("message")

    if message is not None:
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = ""

        if isinstance(content, str) and content.strip():
            return content.strip()

    raise RuntimeError("Ollama response did not contain any content.")


def parse_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    json_start = stripped.find("{")
    json_end = stripped.rfind("}")
    if 0 <= json_start < json_end:
        candidate = stripped[json_start : json_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {"raw": stripped}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def escape_cell(value: str) -> str:
        return value.replace("|", r"\|").replace("\n", " ")

    header_row = "| " + " | ".join(escape_cell(header) for header in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def format_llm_section(title: str, value: Any) -> str:
    if value is None:
        return f"### {title}\n\n_Nenalezeno._"

    if isinstance(value, list):
        if not value:
            return f"### {title}\n\n_Nenalezeno._"
        return "\n".join([f"### {title}"] + [f"- {item}" for item in value])

    if isinstance(value, dict):
        if not value:
            return f"### {title}\n\n_Nenalezeno._"
        lines = [f"### {title}"]
        for key, item in value.items():
            if isinstance(item, list):
                lines.append(f"- **{key}**")
                lines.extend(f"  - {subitem}" for subitem in item)
            else:
                lines.append(f"- **{key}**: {item}")
        return "\n".join(lines)

    return f"### {title}\n\n{value}"


def build_markdown_report(input_path: Path, summary: dict[str, Any], llm_result: Any, llm_provider: str, model_name: str) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Telegram cross-channel analysis")
    lines.append("")
    lines.append(f"- **Input**: {input_path}")
    lines.append(f"- **Generated**: {generated_at}")
    lines.append(f"- **LLM provider**: {llm_provider}")
    lines.append(f"- **LLM model**: {model_name}")
    lines.append(f"- **Sources analyzed**: {summary.get('source_count', 0)}")
    lines.append(f"- **Total messages**: {summary.get('message_count', 0)}")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        "This report combines every Telegram export in the selected folder and looks for shared narratives, repeated stories, shared links, and cross-channel amplification. "
        "The local analysis highlights the most active sources and most traction-heavy posts, while the Llama pass provides higher-level interpretation of how the channels relate to one another."
    )
    lines.append("")

    lines.append("## Sources analyzed")
    lines.append("")
    source_rows = []
    for item in summary.get("source_summaries", []):
        source_name = item.get("source_name", "")
        source_url = telegram_channel_url(source_name)
        display_name = f"[{source_name}]({source_url})" if source_url else source_name
        source_rows.append([display_name, str(item.get("message_count", "")), str(item.get("unique_senders", ""))])
    if source_rows:
        lines.append(markdown_table(["Source", "Messages", "Unique senders"], source_rows))
    else:
        lines.append("_No source breakdown available._")
    lines.append("")

    lines.append("## Main themes and relations")
    lines.append("")
    llm_topics = llm_result.get("topics") if isinstance(llm_result, dict) else None
    if llm_topics:
        if isinstance(llm_topics, list):
            lines.extend([f"- {topic}" for topic in llm_topics])
        else:
            lines.append(str(llm_topics))
    else:
        lines.append("_The LLM did not return a structured topic list._")
    lines.append("")

    llm_summary = llm_result.get("summary") if isinstance(llm_result, dict) else None
    if llm_summary:
        lines.append(str(llm_summary))
    elif isinstance(llm_result, dict) and llm_result.get("raw"):
        lines.append(str(llm_result["raw"]))
    elif isinstance(llm_result, dict) and llm_result.get("raw"):
        lines.append(str(llm_result["raw"]))
    else:
        lines.append("_No summary text returned from the LLM._")
    lines.append("")

    lines.append("## Cross-channel overlap")
    lines.append("")
    cross_source_duplicates = summary.get("cross_source_duplicates", [])
    if cross_source_duplicates:
        for item in cross_source_duplicates[:10]:
            lines.append(f"- **Repeated across**: {', '.join(item.get('source_names', []))}")
            lines.append(f"  - Count: {item.get('count', 0)}")
            lines.append(f"  - Sample: {item.get('sample_text', '')}")
    else:
        lines.append("- No identical posts were detected across multiple sources.")

    shared_links = summary.get("shared_links", [])
    if shared_links:
        lines.append("")
        lines.append("Shared links seen in more than one source:")
        for item in shared_links[:10]:
            lines.append(f"- {item.get('link', '')} ({', '.join(item.get('source_names', []))})")
    lines.append("")

    lines.append("## Most popular posts")
    lines.append("")
    top_posts = summary.get("top_posts", [])
    if top_posts:
        table_rows: list[list[str]] = []
        for post in top_posts[:10]:
            excerpt = textwrap.shorten(str(post.get("text") or "[no text]"), width=180, placeholder="...")
            post_url = telegram_post_url(post.get("source_name", ""), post.get("id"))
            link_cell = f"[open]({post_url})" if post_url else ""
            table_rows.append([
                str(post.get("id", "")),
                str(post.get("sender_id", "")),
                str(post.get("views", 0)),
                str(post.get("forwards", 0)),
                str(post.get("reactions", 0)),
                str(post.get("traction_score", 0)),
                excerpt,
                link_cell,
            ])
        lines.append(markdown_table(["ID", "Sender", "Views", "Forwards", "Reactions", "Traction", "Excerpt", "Telegram"], table_rows))
    else:
        lines.append("_No top posts available._")
    lines.append("")

    lines.append("## Source breakdown")
    lines.append("")
    for source in summary.get("source_summaries", []):
        source_name = source.get("source_name", "unknown")
        channel_url = telegram_channel_url(source_name)
        if channel_url:
            lines.append(f"### [{source_name}]({channel_url})")
        else:
            lines.append(f"### {source_name}")
        lines.append(f"- Messages: {source.get('message_count', 0)}")
        lines.append(f"- Unique senders: {source.get('unique_senders', 0)}")
        if source.get("top_posts"):
            lines.append("- Top posts:")
            for post in source.get("top_posts", [])[:3]:
                excerpt = textwrap.shorten(str(post.get("text") or "[no text]"), width=140, placeholder="...")
                post_url = telegram_post_url(source_name, post.get("id"))
                if post_url:
                    lines.append(f"  - [{excerpt}]({post_url}) (traction {post.get('traction_score', 0)})")
                else:
                    lines.append(f"  - {excerpt} (traction {post.get('traction_score', 0)})")
        if source.get("repeated_messages"):
            lines.append("- Repeated messages:")
            for item in source.get("repeated_messages", [])[:2]:
                lines.append(f"  - {item.get('count', 0)} occurrences: {textwrap.shorten(str(item.get('sample_text', '')), width=120, placeholder='...')}")
        if source.get("non_telegram_links"):
            lines.append("- External links:")
            for link in source.get("non_telegram_links", [])[:3]:
                lines.append(f"  - {link}")
        lines.append("")

    lines.append("## Deeper LLM analysis")
    lines.append("")
    lines.append(format_llm_section("Traction insights", llm_result.get("traction_insights") if isinstance(llm_result, dict) else None))
    lines.append("")
    lines.append(format_llm_section("Active member patterns", llm_result.get("active_member_patterns") if isinstance(llm_result, dict) else None))
    lines.append("")
    lines.append(format_llm_section("Repeated content observations", llm_result.get("repeated_content_observations") if isinstance(llm_result, dict) else None))
    lines.append("")
    lines.append(format_llm_section("Caveats", llm_result.get("caveats") if isinstance(llm_result, dict) else None))
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append("This report avoids psychological, clinical, or personality profiling of individuals.")

    return "\n".join(lines).strip() + "\n"


def derive_sidecar_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}")


def default_report_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "analysis.md"
    return input_path.with_suffix(".md")


def default_payload_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "analysis.llm_payload.json"
    return derive_sidecar_path(input_path, ".llm_payload.json")


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze exported Telegram channel JSON and summarize the content.")
    parser.add_argument("json_file", help="Path to an exported Telegram JSON file or a folder of exports")
    parser.add_argument("-o", "--output", help="Write the full analysis report to this Markdown file")
    parser.add_argument("--payload-output", help="Write the LLM request payload to this JSON file")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or "llama3", help="Ollama model name")
    parser.add_argument("--sample-limit", type=int, default=80, help="Maximum messages sent to the LLM")
    args = parser.parse_args()

    input_path = Path(args.json_file).expanduser().resolve()
    if not input_path.exists():
        print(f"[-] File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        input_files = discover_input_files(input_path)
    except Exception as exc:
        print(f"[-] Failed to discover input files: {exc}", file=sys.stderr)
        return 1

    if not input_files:
        print(f"[-] No JSON export files found in: {input_path}", file=sys.stderr)
        return 1

    records: list[MessageRecord] = []
    source_summaries: list[dict[str, Any]] = []

    for input_file in input_files:
        try:
            messages = load_messages(input_file)
        except Exception as exc:
            print(f"[-] Failed to load messages from {input_file}: {exc}", file=sys.stderr)
            return 1

        source_name = source_name_for_path(input_file)
        source_records = build_records(messages, source_name)
        records.extend(source_records)
        source_summaries.append(summarize_source_summary(source_records, source_name))

    summary = summarize_records(records)
    summary["input_scope"] = "directory" if input_path.is_dir() else "file"
    summary["source_count"] = len(input_files)
    summary["source_files"] = [str(item) for item in input_files]
    summary["source_summaries"] = source_summaries

    payload: dict[str, Any] | None = None
    payload_output_path: Path | None = None

    payload = build_llm_payload(records, summary, args.sample_limit)
    payload_output_path = Path(args.payload_output).expanduser().resolve() if args.payload_output else default_payload_output_path(input_path)
    write_text_file(payload_output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    try:
        llm_text = call_ollama_llm(model=args.model, payload=payload)
        llm_result: Any = parse_output(llm_text)
    except Exception as exc:
        llm_result = {"error": str(exc)}

    report = build_markdown_report(input_path, summary, llm_result, llm_provider="ollama", model_name=args.model)

    output_path = Path(args.output).expanduser().resolve() if args.output else default_report_output_path(input_path)
    if output_path.suffix.lower() != ".md":
        output_path = output_path.with_suffix(".md")
    write_text_file(output_path, report)
    print(f"[+] Report written to {output_path}")

    if payload_output_path is not None:
        print(f"[+] LLM payload written to {payload_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
