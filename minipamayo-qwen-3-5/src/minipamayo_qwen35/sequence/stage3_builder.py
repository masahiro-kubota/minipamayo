"""Reasoning and prompt builders for the Qwen3.5 Stage 3 path."""

from __future__ import annotations

from typing import Final

ACTION_SECTION_HEADER: Final[str] = "[Action Tokens]"

DEFAULT_STAGE3_USER_PROMPT: Final[str] = (
    "Current ego speed: {v0:.2f} m/s.\n"
    "Analyze the driving scene, provide the structured reasoning format below, "
    "then output the action tokens under the action section."
)

TRACE_TEMPLATES: Final[dict[tuple[str, str], str]] = {
    ("go_straight", "lane_keeping"): (
        "The route command indicates lane following and the planner remains in nominal cruise. "
        "The ego vehicle should continue forward while keeping the current lane."
    ),
    ("stop", "lane_keeping"): (
        "The planner state indicates a stopping behavior. "
        "The ego vehicle should brake to a stop while staying in the current lane."
    ),
    ("yield", "lane_keeping"): (
        "The planner state indicates a yielding behavior. "
        "The ego vehicle should slow down and keep the current lane while giving way."
    ),
    ("follow_lead", "lane_keeping"): (
        "The planner state indicates following behavior. "
        "The ego vehicle should keep the lane and adapt speed to the lead vehicle."
    ),
}


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_driving_decision(command: str, planner_state: str) -> dict[str, str]:
    """Infer simplified Alpamayo-style decisions from CARLA planner labels."""

    command_label = normalize_label(command)
    planner_label = normalize_label(planner_state)

    if "lane_change_left" in command_label or (
        "change" in command_label and "left" in command_label
    ):
        lateral = "lane_change_left"
    elif "lane_change_right" in command_label or (
        "change" in command_label and "right" in command_label
    ):
        lateral = "lane_change_right"
    elif "left" in command_label:
        lateral = "turn_left"
    elif "right" in command_label:
        lateral = "turn_right"
    else:
        lateral = "lane_keeping"

    if "yield" in planner_label or "yield" in command_label:
        longitudinal = "yield"
    elif "stop" in planner_label or "stop" in command_label or "red" in planner_label:
        longitudinal = "stop"
    elif "follow" in planner_label or "follow" in command_label:
        longitudinal = "follow_lead"
    else:
        longitudinal = "go_straight"

    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
    }


def _default_trace(command: str, planner_state: str, decision: dict[str, str]) -> str:
    longitudinal = decision["longitudinal"]
    lateral = decision["lateral"]
    template = TRACE_TEMPLATES.get((longitudinal, lateral))
    if template is not None:
        return template
    return (
        f"The route command is {command} and the planner state is {planner_state}. "
        f"The ego vehicle should execute {longitudinal} while keeping the lateral intent {lateral}."
    )


def build_reasoning_text(
    command: str,
    planner_state: str,
    decision: dict[str, str] | None = None,
) -> str:
    """Build a deterministic structured reasoning target from planner labels."""

    resolved_decision = decision or infer_driving_decision(command, planner_state)
    trace = _default_trace(command, planner_state, resolved_decision)
    return (
        "[Driving Decision]\n"
        f"longitudinal: {resolved_decision['longitudinal']}\n"
        f"lateral: {resolved_decision['lateral']}\n\n"
        "[Critical Components]\n"
        f"- route_command: {command}\n"
        f"- planner_state: {planner_state}\n\n"
        "[CoC Trace]\n"
        f"{trace}"
    )


def build_stage3_user_prompt(v0: float) -> str:
    return DEFAULT_STAGE3_USER_PROMPT.format(v0=float(v0))


def build_chat_prompt_text(processor, user_prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_stage3_prompt_text(processor, v0: float) -> str:
    return build_chat_prompt_text(processor, user_text=build_stage3_user_prompt(v0))


def build_stage3_target_text(reasoning_text: str, action_text: str) -> str:
    return f"{reasoning_text}\n\n{ACTION_SECTION_HEADER}\n{action_text}"
