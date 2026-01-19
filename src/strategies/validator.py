"""Strategy definition validator for rule-based strategies."""

from typing import Any

from src.strategies.indicators import INDICATOR_PARAMS

# Valid condition types
CONDITION_TYPES = {"crossover", "crossunder", "above", "below", "between"}

# Built-in data references
BUILTIN_REFS = {"close", "open", "high", "low", "volume"}

# Logic operators
LOGIC_OPERATORS = {"and", "or"}


def validate_strategy_definition(definition: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a strategy definition JSON.

    Args:
        definition: Strategy definition dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Check required top-level fields
    if not isinstance(definition, dict):
        return False, ["Definition must be a dictionary"]

    if "name" not in definition:
        errors.append("Missing required field: 'name'")
    elif not isinstance(definition["name"], str) or not definition["name"].strip():
        errors.append("Field 'name' must be a non-empty string")

    if "indicators" not in definition:
        errors.append("Missing required field: 'indicators'")

    if "rules" not in definition:
        errors.append("Missing required field: 'rules'")

    # If basic structure is invalid, return early
    if errors:
        return False, errors

    # Validate indicators
    indicator_ids = set()
    indicators = definition.get("indicators", [])

    if not isinstance(indicators, list):
        errors.append("Field 'indicators' must be a list")
    else:
        for i, indicator in enumerate(indicators):
            indicator_errors = _validate_indicator(indicator, i, indicator_ids)
            errors.extend(indicator_errors)
            if "id" in indicator:
                indicator_ids.add(indicator["id"])

    # Validate rules
    rules = definition.get("rules", {})
    if not isinstance(rules, dict):
        errors.append("Field 'rules' must be a dictionary")
    else:
        # Valid references include built-ins, indicator IDs, and MACD/BB sub-refs
        valid_refs = _get_valid_references(indicator_ids, definition.get("indicators", []))

        if "buy" in rules:
            buy_errors = _validate_rule(rules["buy"], "buy", valid_refs)
            errors.extend(buy_errors)

        if "sell" in rules:
            sell_errors = _validate_rule(rules["sell"], "sell", valid_refs)
            errors.extend(sell_errors)

        if "buy" not in rules and "sell" not in rules:
            errors.append("Rules must contain at least 'buy' or 'sell'")

    return len(errors) == 0, errors


def _validate_indicator(
    indicator: Any, index: int, existing_ids: set[str]
) -> list[str]:
    """Validate a single indicator definition."""
    errors: list[str] = []
    prefix = f"indicators[{index}]"

    if not isinstance(indicator, dict):
        return [f"{prefix}: Must be a dictionary"]

    # Check required fields
    if "id" not in indicator:
        errors.append(f"{prefix}: Missing required field 'id'")
    elif not isinstance(indicator["id"], str) or not indicator["id"].strip():
        errors.append(f"{prefix}: Field 'id' must be a non-empty string")
    elif indicator["id"] in existing_ids:
        errors.append(f"{prefix}: Duplicate indicator id '{indicator['id']}'")
    elif indicator["id"] in BUILTIN_REFS:
        errors.append(
            f"{prefix}: Indicator id '{indicator['id']}' conflicts with built-in reference"
        )

    if "type" not in indicator:
        errors.append(f"{prefix}: Missing required field 'type'")
    elif indicator["type"] not in INDICATOR_PARAMS:
        errors.append(
            f"{prefix}: Unknown indicator type '{indicator['type']}'. "
            f"Valid types: {', '.join(INDICATOR_PARAMS.keys())}"
        )
    else:
        # Validate parameters for this indicator type
        ind_type = indicator["type"]
        params = indicator.get("params", {})
        param_errors = _validate_indicator_params(ind_type, params, prefix)
        errors.extend(param_errors)

    return errors


def _validate_indicator_params(
    ind_type: str, params: Any, prefix: str
) -> list[str]:
    """Validate indicator parameters against bounds."""
    errors: list[str] = []

    if not isinstance(params, dict):
        return [f"{prefix}.params: Must be a dictionary"]

    expected_params = INDICATOR_PARAMS.get(ind_type, {})

    for param_name, bounds in expected_params.items():
        if param_name not in params:
            # Use defaults from the indicator functions
            continue

        value = params[param_name]
        param_type = bounds.get("type", "int")

        # Type check
        if param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    f"{prefix}.params.{param_name}: Must be an integer, got {type(value).__name__}"
                )
                continue
        elif param_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(
                    f"{prefix}.params.{param_name}: Must be a number, got {type(value).__name__}"
                )
                continue

        # Bounds check
        min_val = bounds.get("min")
        max_val = bounds.get("max")

        if min_val is not None and value < min_val:
            errors.append(
                f"{prefix}.params.{param_name}: Value {value} is below minimum {min_val}"
            )
        if max_val is not None and value > max_val:
            errors.append(
                f"{prefix}.params.{param_name}: Value {value} exceeds maximum {max_val}"
            )

    return errors


def _get_valid_references(
    indicator_ids: set[str], indicators: list[dict]
) -> set[str]:
    """Get all valid references including indicator sub-components."""
    valid_refs = BUILTIN_REFS.copy()
    valid_refs.update(indicator_ids)

    # Add sub-references for multi-output indicators
    for indicator in indicators:
        if not isinstance(indicator, dict):
            continue

        ind_id = indicator.get("id", "")
        ind_type = indicator.get("type", "")

        if ind_type == "macd":
            valid_refs.add(f"{ind_id}.line")
            valid_refs.add(f"{ind_id}.signal")
            valid_refs.add(f"{ind_id}.histogram")
        elif ind_type == "bollinger_bands":
            valid_refs.add(f"{ind_id}.upper")
            valid_refs.add(f"{ind_id}.middle")
            valid_refs.add(f"{ind_id}.lower")

    return valid_refs


def _validate_rule(rule: Any, rule_name: str, valid_refs: set[str]) -> list[str]:
    """Validate a buy/sell rule definition."""
    errors: list[str] = []
    prefix = f"rules.{rule_name}"

    if not isinstance(rule, dict):
        return [f"{prefix}: Must be a dictionary"]

    # Check logic operator
    logic = rule.get("logic", "and")
    if logic not in LOGIC_OPERATORS:
        errors.append(
            f"{prefix}.logic: Invalid logic operator '{logic}'. Valid: {', '.join(LOGIC_OPERATORS)}"
        )

    # Check conditions
    conditions = rule.get("conditions", [])
    if not isinstance(conditions, list):
        errors.append(f"{prefix}.conditions: Must be a list")
    elif len(conditions) == 0:
        errors.append(f"{prefix}.conditions: Must contain at least one condition")
    else:
        for i, condition in enumerate(conditions):
            cond_errors = _validate_condition(
                condition, f"{prefix}.conditions[{i}]", valid_refs
            )
            errors.extend(cond_errors)

    return errors


def _validate_condition(
    condition: Any, prefix: str, valid_refs: set[str]
) -> list[str]:
    """Validate a single condition."""
    errors: list[str] = []

    if not isinstance(condition, dict):
        return [f"{prefix}: Must be a dictionary"]

    # Check condition type
    cond_type = condition.get("type")
    if cond_type is None:
        errors.append(f"{prefix}: Missing required field 'type'")
    elif cond_type not in CONDITION_TYPES:
        errors.append(
            f"{prefix}.type: Invalid condition type '{cond_type}'. "
            f"Valid types: {', '.join(CONDITION_TYPES)}"
        )

    # Check left reference (required for all conditions)
    left = condition.get("left")
    if left is None:
        errors.append(f"{prefix}: Missing required field 'left'")
    elif not isinstance(left, str):
        errors.append(f"{prefix}.left: Must be a string")
    elif left not in valid_refs:
        errors.append(
            f"{prefix}.left: Invalid reference '{left}'. "
            f"Valid refs: {', '.join(sorted(valid_refs))}"
        )

    # Check right reference (required for crossover, crossunder, above, below)
    if cond_type in {"crossover", "crossunder", "above", "below"}:
        right = condition.get("right")
        if right is None:
            errors.append(f"{prefix}: Missing required field 'right' for type '{cond_type}'")
        elif not isinstance(right, str):
            # Allow numeric values for comparison
            if not isinstance(right, (int, float)):
                errors.append(f"{prefix}.right: Must be a string or number")
        elif right not in valid_refs:
            errors.append(
                f"{prefix}.right: Invalid reference '{right}'. "
                f"Valid refs: {', '.join(sorted(valid_refs))}"
            )

    # Check bounds for 'between' condition
    if cond_type == "between":
        lower = condition.get("lower")
        upper = condition.get("upper")

        if lower is None:
            errors.append(f"{prefix}: Missing required field 'lower' for type 'between'")
        elif not isinstance(lower, (int, float)):
            errors.append(f"{prefix}.lower: Must be a number")

        if upper is None:
            errors.append(f"{prefix}: Missing required field 'upper' for type 'between'")
        elif not isinstance(upper, (int, float)):
            errors.append(f"{prefix}.upper: Must be a number")

        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            if lower >= upper:
                errors.append(f"{prefix}: 'lower' must be less than 'upper'")

    return errors
