#!/usr/bin/env python3
"""Extract the shared attack-text corpus for the three new benchmarks.

All attack texts come from the AgentLAB repository clone
(``agentlab/AgentLAB_repo``) and its vendored AgentDojo, plus the upstream
AgentDojo clone (``agentlab/agentdojo_upstream``) that ships the environment
data. Everything is public benchmark material bundled in the repos.

Attack families:
  * injection_vectors  - AgentDojo injection payloads (embedded in emails /
                         files / web pages; the classic indirect-injection text)
  * injection_goals    - the GOAL string of every AgentDojo injection task
                         (what the hijacked agent is told to do)
  * agent_attacks      - Agent-SafetyBench instructions (attacks phrased as
                         agent user requests / embedded context)
  * hijack_descriptions - AgentLAB Intent-Hijacking task descriptions
  * horizon_payloads   - AgentLAB long-horizon injection payloads (emails /
                         system logs embedded in environment content)
"""
from __future__ import annotations

import ast
import json
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTLAB = os.path.join(ROOT, "agentlab", "AgentLAB_repo")
AGENTDOJO_UP = os.path.join(ROOT, "agentlab", "agentdojo_upstream")
SUITES_DATA = os.path.join(AGENTDOJO_UP, "src", "agentdojo", "data", "suites")
SUITES_SRC = os.path.join(AGENTDOJO_UP, "src", "agentdojo", "default_suites")


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader with ``!include`` support (agentdojo splits env YAMLs)."""

    def __init__(self, stream):
        self._root = os.path.dirname(stream.name)
        super().__init__(stream)


def _include(loader: _IncludeLoader, node):
    path = os.path.join(loader._root, loader.construct_scalar(node))
    with open(path) as f:
        return yaml.load(f, _IncludeLoader)


_IncludeLoader.add_constructor("!include", _include)


def _fmt(value) -> str:
    if isinstance(value, datetime_like := (type(None), int, float, str)):
        return str(value)
    return str(value)


def load_injection_vectors() -> dict[str, list[str]]:
    """suite -> list of injection payload texts."""
    out: dict[str, list[str]] = {}
    for suite in ("workspace", "banking", "travel", "slack"):
        path = os.path.join(SUITES_DATA, suite, "injection_vectors.yaml")
        data = yaml.safe_load(open(path))
        texts = []
        for name, vec in (data or {}).items():
            if isinstance(vec, str):
                t = vec.strip()
            elif isinstance(vec, dict):
                t = " ".join(str(v) for v in vec.values() if isinstance(v, str)).strip()
            else:
                t = str(vec).strip()
            if len(t) >= 40:
                texts.append(t)
        out[suite] = texts
    return out


def _substitute_fstring(expr: str, consts: dict[str, str]) -> str:
    """Resolve {name} placeholders of simple f-strings using class-level consts."""
    def repl(m):
        name = m.group(1).strip()
        return consts.get(name, m.group(0))

    return re.sub(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", repl, expr)


def _extract_string_assigns(src: str, names: tuple[str, ...]) -> list[str]:
    """Extract string constants assigned to the given attribute names.

    Handles both plain strings and f-strings whose placeholders refer to
    module/class-level string constants defined in the same file/class body.
    """
    tree = ast.parse(src)
    out: list[str] = []
    # collect all constant string assignments at module and class level
    const_map: dict[str, str] = {}

    def visit_assigns(body):
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant) \
                            and isinstance(node.value.value, str):
                        const_map[t.id] = node.value.value

    visit_assigns(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            visit_assigns(node.body)

    for node in ast.walk(tree):
        for assigns in ([node] if isinstance(node, (ast.Assign, ast.AnnAssign)) else []):
            targets = assigns.targets if isinstance(assigns, ast.Assign) else [assigns.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in names:
                    v = assigns.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        out.append(_substitute_fstring(v.value, const_map))
                    elif isinstance(v, ast.JoinedStr):  # f-string
                        parts = []
                        ok = True
                        for value in v.values:
                            if isinstance(value, ast.Constant):
                                parts.append(value.value)
                            elif isinstance(value, ast.FormattedValue) and \
                                    isinstance(value.value, ast.Name) and \
                                    value.value.id in const_map:
                                parts.append(const_map[value.value.id])
                            else:
                                ok = False
                                break
                        if ok and parts:
                            out.append("".join(parts))
    return out


def load_injection_goals() -> list[str]:
    """GOAL of every injection task across all AgentDojo suite versions."""
    out: list[str] = []
    for ver in sorted(os.listdir(SUITES_SRC)):
        for suite in ("workspace", "banking", "travel", "slack"):
            path = os.path.join(SUITES_SRC, ver, suite, "injection_tasks.py")
            if os.path.exists(path):
                out += [g for g in _extract_string_assigns(open(path).read(), ("GOAL",))
                        if len(g) >= 30]
    return out


def load_user_prompts() -> list[str]:
    """PROMPT of every benign user task across all AgentDojo suite versions."""
    out: list[str] = []
    for ver in sorted(os.listdir(SUITES_SRC)):
        for suite in ("workspace", "banking", "travel", "slack"):
            path = os.path.join(SUITES_SRC, ver, suite, "user_tasks.py")
            if os.path.exists(path):
                out += [p for p in _extract_string_assigns(open(path).read(), ("PROMPT",))
                        if len(p) >= 20]
    return out


def load_agent_attacks() -> list[dict]:
    """Agent-SafetyBench attack instructions (2000)."""
    path = os.path.join(AGENTLAB, "Agent_SafetyBench", "data", "released_data.json")
    data = json.load(open(path))
    out = []
    for r in data:
        ins = (r.get("instruction") or "").strip()
        if len(ins) >= 30:
            out.append({"text": ins, "risks": r.get("risks", []), "id": r.get("id")})
    return out


def load_hijack_descriptions() -> list[dict]:
    """AgentLAB Intent-Hijacking attack task descriptions (200)."""
    path = os.path.join(AGENTLAB, "data", "filtered_top_200_attacks.json")
    data = json.load(open(path))
    out = []
    for r in data:
        d = (r.get("task_description") or "").strip()
        if len(d) >= 30:
            out.append({"text": d, "id": r.get("original_id"),
                        "score": r.get("malicious_score")})
    return out


def load_horizon_payloads() -> list[str]:
    """Long-horizon injection payloads embedded in environment content.

    These live in AgentLAB's ``res/long_horizon/*.json`` as user/system turns
    of injected conversations, and in ``res/injection.json`` /
    ``rewritten_response_chat_1.json``.
    """
    res = os.path.join(AGENTLAB, "Task-Injection", "agentdojo", "res")
    out: list[str] = []
    for name in ("injection.json", "rewritten_response_chat_1.json"):
        path = os.path.join(res, name)
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict):
                for c in item.get("content", []) or []:
                    if isinstance(c, dict):
                        for role, text in c.items():
                            t = str(text).strip()
                            if len(t) >= 60:
                                out.append(t)
    for name in ("example_slack.json", "example_workspace.json", "example_travel.json",
                 "example_banking.json"):
        path = os.path.join(res, "long_horizon", name)
        if not os.path.exists(path):
            continue
        for item in json.load(open(path)):
            for c in item.get("content", []) or []:
                if isinstance(c, dict):
                    for role, text in c.items():
                        t = str(text).strip()
                        if len(t) >= 60:
                            out.append(t)
    return out


def load_all_attacks() -> dict:
    return {
        "injection_vectors": load_injection_vectors(),
        "injection_goals": load_injection_goals(),
        "agent_attacks": load_agent_attacks(),
        "hijack_descriptions": load_hijack_descriptions(),
        "horizon_payloads": load_horizon_payloads(),
        "user_prompts": load_user_prompts(),
    }


if __name__ == "__main__":
    c = load_all_attacks()
    for k, v in c.items():
        print(f"{k}: {len(v)}")
    print("sample injection vector:", c["injection_vectors"]["workspace"][0][:120])
    print("sample goal:", c["injection_goals"][0][:120])
    print("sample agent attack:", c["agent_attacks"][0]["text"][:120])
    print("sample hijack:", c["hijack_descriptions"][0]["text"][:120])
    print("sample horizon:", c["horizon_payloads"][0][:120])
    print("sample user prompt:", c["user_prompts"][0][:120])
