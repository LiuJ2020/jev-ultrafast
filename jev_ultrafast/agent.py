"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text
from .questions import MAX_STEPS

ADDRESS_BAR = {
    "label": "Browser address bar: the full https:// URL of the website to open for this goal",
    "role": "url",
}


def blocked_host(url):
    """Hosts listed in JEV_BLOCKED_HOSTS are never observed by the models or opened by the agent."""
    host = (urlparse(url).hostname or "").lower()
    domains = [d.strip().lower() for d in os.environ.get("JEV_BLOCKED_HOSTS", "").split(",") if d.strip()]
    return next((d for d in domains if host == d or host.endswith("." + d)), None)


def web_url(text):
    """Normalize typed or generated text into an http(s) URL, or refuse."""
    value = (text or "").strip()
    if value == "about:blank":
        return value
    refused = ValueError(f"Not an http(s) URL: {text!r}; nothing opened.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        # "javascript:…", "data:…", "mailto:…" have a scheme but no "//"; "localhost:8080" is a host and port.
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:(?!\d)", value):
            raise refused
        local = re.match(r"^(localhost|127\.0\.0\.1)(:\d+)?(/|$)", value)
        value = ("http://" if local else "https://") + value
    parsed = urlparse(value)
    try:
        parsed.port
    except ValueError:
        raise refused from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or re.search(r"\s", value):
        raise refused
    if domain := blocked_host(value):
        raise ValueError(f"{parsed.hostname} is blocked by JEV_BLOCKED_HOSTS ({domain}); nothing opened.")
    return value


class Agent:
    def __init__(self, url="about:blank", goals=None, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals or []).strip()
        plan = [task] if task else []
        self.pending_text = None
        self.browser = Browser(web_url(url))
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            turns=[{"goal": task, "status": "ready", "start": 0}] if task else [],
            turn_start=0,
            turn_decisions=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def turn_history(self):
        return self.state["history"][self.state.get("turn_start", 0) :]

    def finish_turn(self, status):
        self.state["status"] = status
        if self.state.get("turns"):
            self.state["turns"][-1].update(status=status, actions=len(self.turn_history()))

    def prompt(self, goal):
        """Start a new turn on the current page. Earlier turns stay in the trail but not in the model's history."""
        goal = (goal or "").strip()
        if not goal or len(goal) > 2000:
            raise ValueError("Enter 1–2,000 characters")
        state = self.state
        if state.get("turns") and state["turns"][-1]["status"] not in {"done", "blocked"}:
            state["turns"][-1].update(status="replaced", actions=len(self.turn_history()))
        state.update(
            goal=goal,
            decision=None,
            status="ready",
            turn_start=len(state["history"]),
            turn_decisions=len(state["decisions"]),
            started_at=None,
            elapsed_ms=0,
        )
        state["plan"].append(goal)
        state["plan_index"] = len(state["plan"]) - 1
        state.setdefault("turns", []).append({"goal": goal, "status": "ready", "start": len(state["history"])})
        self.pending_text = None
        state["page"] = state["browser"].observe(screenshot=self.screenshots)
        return self.snapshot()

    def goto(self, url):
        """A navigation you request directly; recorded in the trail, no model call."""
        url = web_url(url)
        state = self.state
        state["decision"] = None
        state["browser"].navigate(url)
        state["page"] = state["browser"].observe(screenshot=self.screenshots)
        state["history"].append(
            {
                "step": len(state["history"]) + 1,
                "action": f"You opened {url}",
                "kind": "navigate",
                "operation": "USER_NAVIGATE",
                "choice": "navigate",
                "probability": None,
                "confidence": None,
                "latency_ms": 0,
                "text": None,
                "page_changed": True,
                "url": state["page"]["url"],
            }
        )
        if state["status"] == "predicted":
            state["status"] = "ready"
        return self.snapshot()

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "prompt":
            return self.prompt(body.get("goal"))
        if name == "goto":
            return self.goto(body.get("url"))
        if name == "observe":
            # Re-read the page without a model call, e.g. after a form finishes rendering or you sign in.
            state["decision"] = None
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            if state["status"] == "predicted":
                state["status"] = "ready"
            return self.snapshot()
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if not state["goal"]:
                raise ValueError("Send Jev a prompt first")
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This turn has stopped. Send a new prompt to continue.")
            if domain := blocked_host(state["page"]["url"]):
                raise ValueError(f"This page is blocked by JEV_BLOCKED_HOSTS ({domain}); nothing was sent to Jev.")
            if len(state["decisions"]) - state.get("turn_decisions", 0) >= MAX_STEPS * 2:
                raise ValueError("Reached this turn's model-call budget")
            state["decision"] = choose(state["page"], state["goal"], self.turn_history())
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                self.finish_turn("done" if selected == "DONE" else "blocked")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(self.turn_history()) >= MAX_STEPS:
                self.finish_turn("blocked")
                raise ValueError(f"Stopped at the {MAX_STEPS}-action budget for this turn")
            text, helper = None, None
            if action["kind"] in {"fill", "navigate"}:
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                field = action if action["kind"] == "fill" else {**ADDRESS_BAR, "value": page["url"]}
                context = field_context(state["goal"], field, page, self.turn_history())
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": field["label"], "value": text})
                if action["kind"] == "navigate":
                    text = web_url(text)
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": "Open URL" if action["kind"] == "navigate" else action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            repeated = self.turn_history()[-3:]
            if len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated):
                self.finish_turn("blocked")
            else:
                state["status"] = "ready"
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
