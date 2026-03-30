ACTION_SYMBOLS = {
    "move_to":   "->",
    "grasp":     "[grip]",
    "lift":      "^",
    "place_on":  "v",
    "release":   "[open]",
    "move_home": "[home]",
}


def log_plan(plan: dict):
    """Pretty-prints the LLM's reasoning and action plan to the terminal."""
    print("\n" + "=" * 50)
    print("  ROBOT PLAN")
    print("=" * 50)

    reasoning = plan.get("reasoning", "").strip()
    if reasoning:
        print(f"\nReasoning:\n  {reasoning}\n")

    steps = plan.get("steps", [])
    print(f"Steps ({len(steps)} total):")
    for i, step in enumerate(steps, start=1):
        action = step.get("action", "?")
        args   = step.get("args", {})
        symbol = ACTION_SYMBOLS.get(action, "•")

        # Format args nicely
        if args:
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            line = f"  {i:2}. {symbol}  {action}({arg_str})"
        else:
            line = f"  {i:2}. {symbol}  {action}()"

        print(line)

    print("=" * 50)
