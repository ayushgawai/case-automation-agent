"""Tools the agent may call, and the guardrails that run before any tool executes.

Guardrails live in the executor, not the planner, so they hold no matter what proposes the
action: the rule-based planner, an LLM, or text injected into the case body.
"""
import re
from dataclasses import dataclass, field

from .data import PRIVILEGED_SETS, STANDARD_SETS, Case, Org
from .retrieval import BM25Index

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

TOOL_SCHEMAS = [
    {"name": "search_kb", "description": "Search knowledge articles. Returns the best matching passages.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "search_logs", "description": "Find log lines for this case containing a pattern, such as an error code.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "unlock_account", "description": "Unlock a locked user. Requires verified identity.",
     "input_schema": {"type": "object", "properties": {"user": {"type": "string"}}, "required": ["user"]}},
    {"name": "reset_mfa", "description": "Reset a user's MFA registration. Requires verified identity.",
     "input_schema": {"type": "object", "properties": {"user": {"type": "string"}}, "required": ["user"]}},
    {"name": "grant_permission", "description": "Assign a standard permission set when the requester is the user's manager.",
     "input_schema": {"type": "object", "properties": {"user": {"type": "string"}, "permission_set": {"type": "string"}},
                      "required": ["user", "permission_set"]}},
    {"name": "request_approval", "description": "Send a privileged access request to the security team.",
     "input_schema": {"type": "object", "properties": {"user": {"type": "string"}, "permission_set": {"type": "string"}},
                      "required": ["user", "permission_set"]}},
    {"name": "ask_verification", "description": "Ask the user to verify their identity before account changes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "reply", "description": "Send the final reply and close the case as resolved.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "escalate", "description": "Route the case to a human with a reason.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
TERMINAL = {"reply": "resolved", "request_approval": "pending_approval", "ask_verification": "needs_verification",
            "escalate": "escalated"}


@dataclass
class ToolResult:
    ok: bool
    output: str
    blocked: str = ""  # guardrail that refused the call, if any


@dataclass
class Executor:
    org: Org
    kb: BM25Index
    audit: list = field(default_factory=list)

    def check(self, case: Case, name: str, args: dict) -> str:
        """Return the reason a call is refused, or an empty string if it may run."""
        if name not in TOOL_NAMES:
            return "unknown_tool"
        user = args.get("user")
        if user is not None and user != case.user:
            return "user_mismatch"  # actions only on the case's own subject user
        if name in ("unlock_account", "reset_mfa") and not case.identity_verified:
            return "identity_not_verified"
        if name == "grant_permission":
            perm = args.get("permission_set", "")
            if perm in PRIVILEGED_SETS:
                return "privileged_requires_approval"
            if perm not in STANDARD_SETS:
                return "unknown_permission_set"
            if not case.requester_is_manager:
                return "requester_not_manager"
        if name == "reply" and re.search(r"password\s*[:=]", args.get("text", ""), re.I):
            return "secret_in_reply"
        return ""

    def run(self, case: Case, name: str, args: dict) -> ToolResult:
        reason = self.check(case, name, args)
        if reason:
            res = ToolResult(False, f"refused: {reason}", blocked=reason)
        else:
            res = ToolResult(True, self._execute(case, name, args))
        self.audit.append({"case": case.case_id, "tool": name, "args": args, "ok": res.ok, "blocked": res.blocked})
        return res

    def _execute(self, case: Case, name: str, args: dict) -> str:
        if name == "search_kb":
            hits = self.kb.search(args["query"], k=3)
            return "\n".join(f"[{c.article}] {c.text}" for c, _ in hits) or "no matching article"
        if name == "search_logs":
            prefix = f"case={case.case_id[1:].lstrip('0') or '0'} "
            lines = [l for l in self.org.logs if l.startswith(prefix) and args["pattern"] in l]
            return "\n".join(EMAIL.sub("<redacted>", l) for l in lines) or "no matching log lines"
        if name == "unlock_account":
            self.org.locked.discard(args["user"])
            return f"{args['user']} unlocked"
        if name == "reset_mfa":
            self.org.mfa_registered.discard(args["user"])
            return f"MFA reset for {args['user']}"
        if name == "grant_permission":
            self.org.permissions.setdefault(args["user"], set()).add(args["permission_set"])
            return f"granted {args['permission_set']} to {args['user']}"
        if name == "request_approval":
            return f"approval requested for {args['permission_set']} on {args['user']}"
        return "ok"
