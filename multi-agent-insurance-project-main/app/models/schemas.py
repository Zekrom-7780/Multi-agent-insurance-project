from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request / Response ───────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    policy_number: Optional[str] = None
    claim_id: Optional[str] = None


class AgentStep(BaseModel):
    agent: str
    action: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    agent_trace: list[AgentStep] = []
    requires_human: bool = False


class AgentEventSSE(BaseModel):
    """Payload sent over the SSE stream."""
    conversation_id: str
    agent_name: str
    event_type: str
    payload: dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Tool output wrappers ────────────────────────────────────────

class PolicyDetail(BaseModel):
    policy_number: str
    customer_id: str
    policy_type: str
    start_date: str
    premium_amount: float
    billing_frequency: str
    status: str
    first_name: str = ""
    last_name: str = ""

    model_config = {"from_attributes": True}


class AutoPolicyDetailSchema(BaseModel):
    policy_number: str
    vehicle_vin: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    liability_limit: float
    collision_deductible: float
    comprehensive_deductible: float
    uninsured_motorist: int
    rental_car_coverage: int
    policy_type: str = ""
    premium_amount: float = 0.0

    model_config = {"from_attributes": True}


class BillingInfo(BaseModel):
    bill_id: str
    policy_number: str
    billing_date: str
    due_date: str
    amount_due: float
    status: str
    premium_amount: float = 0.0
    billing_frequency: str = ""

    model_config = {"from_attributes": True}


class PaymentRecord(BaseModel):
    payment_date: str
    amount: float
    status: str
    payment_method: str

    model_config = {"from_attributes": True}


class ClaimInfo(BaseModel):
    claim_id: str
    policy_number: str
    claim_date: str
    incident_type: str
    estimated_loss: float
    status: str
    policy_type: str = ""

    model_config = {"from_attributes": True}
