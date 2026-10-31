"""Plan, check, execute loop with a step budget."""
from dataclasses import dataclass, field

from .data import Case
from .planner import Call
from .tools import TERMINAL, Executor


@dataclass
class Outcome:
    case_id: str
    status: str
    detail: str
    steps: int
    blocked: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def run_case(case: Case, planner, executor: Executor, max_steps: int = 6) -> Outcome:
    history: list[tuple[Call, str]] = []
    blocked = []
    for step in range(1, max_steps + 1):
        call = planner.plan(case, history)
        result = executor.run(case, call.name, call.args)
        history.append((call, result.output))
        if result.blocked:
            blocked.append(result.blocked)
            if len(blocked) >= 2:  # repeated refusals: hand to a human instead of looping
                executor.run(case, "escalate", {"reason": "guardrail refusals"})
                return Outcome(case.case_id, "escalated", "", step, blocked, history)
            continue
        if call.name in TERMINAL:
            detail = call.args.get("permission_set") or call.args.get("text", "")
            return Outcome(case.case_id, TERMINAL[call.name], detail, step, blocked, history)
    executor.run(case, "escalate", {"reason": "step budget exhausted"})
    return Outcome(case.case_id, "escalated", "", max_steps, blocked, history)
