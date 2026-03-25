from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str] = mapped_column(String(20))
    date_of_birth: Mapped[str] = mapped_column(String(30))
    state: Mapped[str] = mapped_column(String(20))

    policies: Mapped[list["Policy"]] = relationship(back_populates="customer")


class Policy(Base):
    __tablename__ = "policies"

    policy_number: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("customers.customer_id")
    )
    policy_type: Mapped[str] = mapped_column(String(50))
    start_date: Mapped[str] = mapped_column(String(30))
    premium_amount: Mapped[float] = mapped_column(Float)
    billing_frequency: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))

    customer: Mapped["Customer"] = relationship(back_populates="policies")
    auto_detail: Mapped["AutoPolicyDetail | None"] = relationship(
        back_populates="policy"
    )
    bills: Mapped[list["Bill"]] = relationship(back_populates="policy")
    claims: Mapped[list["Claim"]] = relationship(back_populates="policy")


class AutoPolicyDetail(Base):
    __tablename__ = "auto_policy_details"

    policy_number: Mapped[str] = mapped_column(
        String(20), ForeignKey("policies.policy_number"), primary_key=True
    )
    vehicle_vin: Mapped[str] = mapped_column(String(50))
    vehicle_make: Mapped[str] = mapped_column(String(50))
    vehicle_model: Mapped[str] = mapped_column(String(50))
    vehicle_year: Mapped[int] = mapped_column(Integer)
    liability_limit: Mapped[float] = mapped_column(Float)
    collision_deductible: Mapped[float] = mapped_column(Float)
    comprehensive_deductible: Mapped[float] = mapped_column(Float)
    uninsured_motorist: Mapped[int] = mapped_column(Integer)
    rental_car_coverage: Mapped[int] = mapped_column(Integer)

    policy: Mapped["Policy"] = relationship(back_populates="auto_detail")


class Bill(Base):
    __tablename__ = "billing"

    bill_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    policy_number: Mapped[str] = mapped_column(
        String(20), ForeignKey("policies.policy_number")
    )
    billing_date: Mapped[str] = mapped_column(String(30))
    due_date: Mapped[str] = mapped_column(String(30))
    amount_due: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))

    policy: Mapped["Policy"] = relationship(back_populates="bills")
    payments: Mapped[list["Payment"]] = relationship(back_populates="bill")


class Payment(Base):
    __tablename__ = "payments"

    payment_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    bill_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("billing.bill_id")
    )
    payment_date: Mapped[str] = mapped_column(String(30))
    amount: Mapped[float] = mapped_column(Float)
    payment_method: Mapped[str] = mapped_column(String(50))
    transaction_id: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))

    bill: Mapped["Bill"] = relationship(back_populates="payments")


class Claim(Base):
    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    policy_number: Mapped[str] = mapped_column(
        String(20), ForeignKey("policies.policy_number")
    )
    claim_date: Mapped[str] = mapped_column(String(30))
    incident_type: Mapped[str] = mapped_column(String(100))
    estimated_loss: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))

    policy: Mapped["Policy"] = relationship(back_populates="claims")


class AgentEvent(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(50), index=True)
    agent_name: Mapped[str] = mapped_column(String(50))
    event_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
