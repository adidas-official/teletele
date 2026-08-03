"""Analyze exported Telegram message JSON with local heuristics and Gemini.

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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
TELEGRAM_HOST_RE = re.compile(r"(^|\.)t\.me$|(^|\.)telegram\.me$|(^|\.)telegram\.org$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
MENTION_RE = re.compile(r"@\w+")


@dataclasses.dataclass(slots=True)
class MessageRecord:
    id: Any
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


def build_records(messages: list[dict[str, Any]]) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for message in messages:
        text = extract_text(message)
        records.append(
            MessageRecord(
                id=message.get("id"),
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


def summarize_records(records: list[MessageRecord]) -> dict[str, Any]:
    by_sender = collections.Counter()
    by_day = collections.Counter()
    duplicate_groups: dict[str, list[MessageRecord]] = collections.defaultdict(list)
    all_links: list[str] = []

    for record in records:
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
            repeated_messages.append(
                {
                    "count": len(group),
                    "message_ids": [item.id for item in group],
                    "sender_ids": [item.sender_id for item in group],
                    "sample_text": group[0].text[:300],
                }
            )
    repeated_messages.sort(key=lambda item: item["count"], reverse=True)

    top_posts = sorted(records, key=lambda item: item.traction_score, reverse=True)[:20]
    top_members = by_sender.most_common(20)

    unique_links = list(dict.fromkeys(all_links))
    non_telegram_links = unique_links

    return {
        "message_count": len(records),
        "unique_senders": len(by_sender),
        "messages_by_day": dict(by_day.most_common(30)),
        "top_members": [
            {"sender_id": sender_id, "message_count": count}
            for sender_id, count in top_members
        ],
        "top_posts": [
            {
                "id": record.id,
                "date": record.date,
                "sender_id": record.sender_id,
                "views": record.views,
                "forwards": record.forwards,
                "reactions": record.reactions,
                "traction_score": record.traction_score,
                "text": record.text[:500],
                "links": record.links,
            }
            for record in top_posts
        ],
        "repeated_messages": repeated_messages[:20],
        "non_telegram_links": non_telegram_links,
    }


def build_gemini_payload(records: list[MessageRecord], summary: dict[str, Any], sample_limit: int) -> dict[str, Any]:
    sample_messages = [
        {
            "id": record.id,
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
        "instructions": [
            "Summarize the main topics in the Telegram export.",
            "Highlight content patterns, recurring themes, and high-traction posts.",
            "Describe communication patterns for active members only in a non-clinical, non-diagnostic way.",
            "Do not infer mental health, personality disorders, emotions, or hidden motives.",
            "Return concise JSON with keys: topics, summary, traction_insights, active_member_patterns, repeated_content_observations, caveats.",
        ],
    }


def call_gemini(api_key: str, model: str, payload: dict[str, Any]) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(api_key)}"
    body = json.dumps(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": json.dumps(payload, ensure_ascii=False, indent=2)}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini API error: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API connection error: {exc.reason}") from exc

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini response did not contain any candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini response did not contain any content parts.")

    return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()


def parse_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return {"raw": stripped}


def render_report(summary: dict[str, Any], ai_result: Any, source_path: Path) -> dict[str, Any]:
    return {
        "source_file": str(source_path),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_summary": summary,
        "gemini_analysis": ai_result,
        "note": "This report avoids psychological, clinical, or personality profiling of individuals.",
    }


def derive_sidecar_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}")


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze exported Telegram channel JSON and summarize the content.")
    parser.add_argument("json_file", help="Path to exported Telegram messages JSON")
    parser.add_argument("-o", "--output", help="Write the full analysis report to this JSON file")
    parser.add_argument("--payload-output", help="Write the Gemini request payload to this JSON file")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--sample-limit", type=int, default=80, help="Maximum messages sent to Gemini")
    parser.add_argument("--no-gemini", action="store_true", help="Skip Gemini and return only local analysis")
    args = parser.parse_args()

    source_path = Path(args.json_file).expanduser().resolve()
    if not source_path.exists():
        print(f"[-] File not found: {source_path}", file=sys.stderr)
        return 1

    try:
        messages = load_messages(source_path)
    except Exception as exc:
        print(f"[-] Failed to load messages: {exc}", file=sys.stderr)
        return 1

    records = build_records(messages)
    summary = summarize_records(records)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ai_result: Any = {
        "skipped": True,
        "reason": "Gemini call disabled or no API key available.",
    }
    payload: dict[str, Any] | None = None
    payload_output_path: Path | None = None

    if not args.no_gemini:
        payload = build_gemini_payload(records, summary, args.sample_limit)
        payload_output_path = Path(args.payload_output).expanduser().resolve() if args.payload_output else derive_sidecar_path(source_path, ".gemini_payload.json")
        write_json_file(payload_output_path, payload)

        if not api_key:
            ai_result = {
                "skipped": True,
                "reason": "Set GEMINI_API_KEY or GOOGLE_API_KEY to enable Gemini analysis.",
            }
        else:
            try:
                ai_text = call_gemini(api_key=api_key, model=args.model, payload=payload)
                ai_result = parse_output(ai_text)
            except Exception as exc:
                ai_result = {"error": str(exc)}

    report = render_report(summary, ai_result, source_path)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        write_json_file(output_path, report)
        print(f"[+] Report written to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if payload_output_path is not None:
        print(f"[+] Gemini payload written to {payload_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
