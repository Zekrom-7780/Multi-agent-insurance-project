"""Default tool registry with all tools registered."""

from app.tools.billing_tools import get_billing_info, get_payment_history
from app.tools.claims_tools import get_claim_status
from app.tools.policy_tools import get_auto_policy_details, get_policy_details
from app.tools.registry import ToolRegistry, ToolSpec


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_policy_details",
            description="Fetch a customer's policy details by policy number",
            parameters={
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string", "description": "e.g. POL000001"}
                },
                "required": ["policy_number"],
            },
            fn=get_policy_details,
        )
    )

    registry.register(
        ToolSpec(
            name="get_auto_policy_details",
            description="Get auto-specific policy details including vehicle info and deductibles",
            parameters={
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string", "description": "e.g. POL000004"}
                },
                "required": ["policy_number"],
            },
            fn=get_auto_policy_details,
        )
    )

    registry.register(
        ToolSpec(
            name="get_billing_info",
            description="Get billing information including current balance and due dates",
            parameters={
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string"},
                    "customer_id": {"type": "string"},
                },
                "required": [],
            },
            fn=get_billing_info,
        )
    )

    registry.register(
        ToolSpec(
            name="get_payment_history",
            description="Get recent payment history for a policy",
            parameters={
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string"}
                },
                "required": ["policy_number"],
            },
            fn=get_payment_history,
        )
    )

    registry.register(
        ToolSpec(
            name="get_claim_status",
            description="Get claim status and details by claim_id or policy_number",
            parameters={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "policy_number": {"type": "string"},
                },
                "required": [],
            },
            fn=get_claim_status,
        )
    )

    return registry


default_registry = build_default_registry()
