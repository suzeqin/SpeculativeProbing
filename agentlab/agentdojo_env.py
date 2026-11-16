#!/usr/bin/env python3
"""Extract benign agent-environment documents from the AgentDojo suites.

These are the tool outputs an agent would observe (emails, cloud files,
calendar events, bank transactions, web pages, slack messages, hotel /
restaurant / car-rental listings with reviews). They form the *benign*
background corpus of the AgentLAB classifier dataset and double as the carrier
documents into which injection payloads are embedded (the classic
indirect-prompt-injection scenario: a poisoned email / file / web page that the
agent reads as part of its task).
"""
from __future__ import annotations

import os

import yaml

from attack_corpus import SUITES_DATA, _IncludeLoader


def _load_env(suite: str) -> dict:
    path = os.path.join(SUITES_DATA, suite, "environment.yaml")
    with open(path) as f:
        return yaml.load(f, _IncludeLoader)


import re as _re


def _strip_placeholders(text: str) -> str:
    """Drop {injection_*} placeholder sentences from benign env templates."""
    out = _re.sub(r"\s*[^.\n]*\{[a-z_]*injection[a-z_0-9]*\}[^.\n]*\.?", " ", text)
    out = _re.sub(r"\{[a-z_0-9]+\}", " ", out)
    return out


def _email(e: dict) -> str:
    lines = [f"Email from {e.get('sender','')} to {', '.join(e.get('recipients',[]) or [])}",
             f"Subject: {e.get('subject','')}"]
    if e.get("cc"):
        lines.append(f"CC: {', '.join(e['cc'])}")
    lines.append(e.get("body", "") or "")
    return "\n".join(lines)


def _file(f: dict) -> str:
    return f"File: {f.get('filename','')}\n{f.get('content','') or ''}"


def _event(e: dict) -> str:
    lines = [f"Calendar event: {e.get('title','')}"]
    if e.get("location"):
        lines.append(f"Location: {e['location']}")
    desc = e.get("description", "") or ""
    if desc:
        lines.append(desc)
    return "\n".join(lines)


def _transaction(t: dict) -> str:
    return (f"Transaction {t.get('id','')}: {t.get('sender','')} -> {t.get('recipient','')}, "
            f"amount {t.get('amount','')}, subject: {t.get('subject','')}")


def _review(name: str, rating, reviews) -> str:
    parts = [name, f"Rating: {rating}"]
    if isinstance(reviews, list):
        for r in reviews:
            if isinstance(r, dict) and r.get("review"):
                parts.append(f"Review by {r.get('author','anonymous')}: {r['review']}")
    return "\n".join(str(p) for p in parts)


def load_benign_documents() -> list[dict]:
    """Returns a list of {"text": ..., "suite": ..., "kind": ...} documents."""
    docs: list[dict] = []

    def add(text: str, suite: str, kind: str):
        t = _strip_placeholders((text or "")).strip()
        if len(t) >= 40:
            docs.append({"text": t, "suite": suite, "kind": kind})

    # workspace
    ws = _load_env("workspace")
    for e in ws["inbox"]["initial_emails"]:
        add(_email(e), "workspace", "email")
    for f in ws["cloud_drive"]["initial_files"]:
        add(_file(f), "workspace", "file")
    for ev in ws["calendar"]["initial_events"]:
        add(_event(ev), "workspace", "calendar")

    # banking
    bk = _load_env("banking")
    for t in bk["bank_account"]["transactions"] + bk["bank_account"]["scheduled_transactions"]:
        add(_transaction(t), "banking", "transaction")
    for fid, f in bk["filesystem"]["files"].items():
        add(_file(f if isinstance(f, dict) else {"filename": fid, "content": f}),
            "banking", "file")

    # travel
    tv = _load_env("travel")
    for e in tv["inbox"]["initial_emails"]:
        add(_email(e), "travel", "email")
    for h in tv["hotels"]["hotel_list"]:
        add(_review(h.get("name", "Hotel"), h.get("rating"), h.get("reviews")), "travel", "review")
    for r in tv["restaurants"]["restaurant_list"]:
        add(_review(r.get("name", "Restaurant"), r.get("rating"), r.get("reviews")),
            "travel", "review")
    for c in tv["car_rental"]["company_list"]:
        add(_review(c.get("name", "Car rental"), c.get("rating"), c.get("reviews")),
            "travel", "review")
    for fl in tv["flights"]["flight_list"]:
        add("Flight {flight_number} {airline}: {departure_city} -> {arrival_city}".format(**{k: fl.get(k, '') for k in ('flight_number', 'airline', 'departure_city', 'arrival_city')}) +
            f"\nContact: {fl.get('contact_information','')}", "travel", "flight")
    for ev in tv["calendar"]["initial_events"]:
        add(_event(ev), "travel", "calendar")

    # slack + web. Environment templates contain {injection_*} placeholders;
    # for the *benign* corpus the placeholder sentence is dropped.
    sl = _load_env("slack")
    for cname in sl["slack"]["channels"]:
        for msg in (sl["slack"]["channel_inbox"].get(cname, []) or []):
            if isinstance(msg, dict):
                add(f"Slack #{cname} | {msg.get('sender','')}: {msg.get('body','')}",
                    "slack", "slack_message")
            elif isinstance(msg, str):
                add(f"Slack #{cname}: {msg}", "slack", "slack_message")
        for user, msgs in (sl["slack"]["user_inbox"] or {}).items():
            for msg in (msgs if isinstance(msgs, list) else [msgs]):
                if isinstance(msg, dict):
                    add(f"Slack DM to {user} from {msg.get('sender','')}: {msg.get('body','')}",
                        "slack", "slack_dm")
                elif isinstance(msg, str):
                    add(f"Slack DM to {user}: {msg}", "slack", "slack_dm")
    for page_id, page in (sl["web"]["web_content"] or {}).items():
        if isinstance(page, dict):
            body = page.get("content", "") or ""
        else:
            body = str(page)
        add(f"Web page '{page_id}'\n{_strip_placeholders(body)}", "slack", "web_page")

    return docs


if __name__ == "__main__":
    docs = load_benign_documents()
    kinds = {}
    for d in docs:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"total {len(docs)} benign documents: {kinds}")
    print("sample:", docs[0]["text"][:150].replace("\n", " | "))
    lens = sorted(len(d["text"]) for d in docs)
    print(f"char len min/med/max: {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
