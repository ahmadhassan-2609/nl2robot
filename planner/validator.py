VALID_ACTIONS = {"move_to", "grasp", "lift", "place_on", "release", "move_home"}
VALID_OBJECTS = {"red_block", "blue_block", "green_block"}


def validate_plan(plan: dict) -> tuple[bool, str]:
    """
    Validates a parsed action plan from the LLM.
    Returns (True, "OK") on success or (False, reason) on failure.
    """
    if not plan.get("feasible", False):
        return False, f"LLM marked task infeasible: {plan.get('reason', 'no reason given')}"

    if "steps" not in plan or not plan["steps"]:
        return False, "Plan contains no steps"

    for i, step in enumerate(plan["steps"]):
        action = step.get("action")
        args   = step.get("args", {})

        if action not in VALID_ACTIONS:
            return False, f"Step {i}: Unknown action '{action}'"

        if "object" in args and args["object"] not in VALID_OBJECTS:
            return False, f"Step {i}: Unknown object '{args['object']}'"

        if action == "lift":
            h = args.get("height", 0)
            if not (0 < h <= 0.3):
                return False, f"Step {i}: Lift height {h} out of safe range (0, 0.3]"

    return True, "OK"
