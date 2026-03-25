"""Seed the SQLite database with synthetic insurance data.

Usage: uv run python -m scripts.seed_db
"""

import asyncio
import random
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import settings
from app.models.db import Base


def generate_sample_data(random_state: int = 42) -> dict:
    """Generate enriched sample data matching the notebook schema."""
    random.seed(random_state)

    first_names = [
        "John", "Jane", "Robert", "Maria", "David", "Lisa", "Michael", "Sarah",
        "James", "Emily", "William", "Emma", "Joseph", "Olivia", "Charles", "Ava",
        "Thomas", "Isabella", "Daniel", "Mia", "Matthew", "Sophia", "Anthony",
        "Charlotte", "Christopher", "Amelia", "Andrew", "Harper", "Joshua", "Evelyn",
        "Ryan", "Abigail", "Brandon", "Ella", "Justin", "Scarlett", "Tyler", "Grace",
        "Alexander", "Chloe", "Kevin", "Victoria", "Jason", "Lily", "Brian", "Hannah",
        "Eric", "Aria", "Kyle", "Zoey",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
        "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
        "Mitchell", "Carter", "Roberts",
    ]
    states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"]

    customers = []
    for i in range(1, 1001):
        dob = datetime(1980, 1, 1) + timedelta(days=random.randint(0, 10000))
        customers.append({
            "customer_id": f"CUST{str(i).zfill(5)}",
            "first_name": random.choice(first_names),
            "last_name": random.choice(last_names),
            "email": f"user{i}@example.com",
            "phone": f"555-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "state": random.choice(states),
        })

    policies = []
    for i in range(1, 1501):
        sd = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        policies.append({
            "policy_number": f"POL{str(i).zfill(6)}",
            "customer_id": f"CUST{str(random.randint(1, 1000)).zfill(5)}",
            "policy_type": random.choice(["auto", "home", "life"]),
            "start_date": sd.strftime("%Y-%m-%d"),
            "premium_amount": round(random.uniform(50, 500), 2),
            "billing_frequency": random.choice(["monthly", "quarterly", "annual"]),
            "status": random.choice(["active", "active", "active", "cancelled"]),
        })

    auto_details = []
    for p in policies:
        if p["policy_type"] == "auto":
            auto_details.append({
                "policy_number": p["policy_number"],
                "vehicle_vin": f"VIN{random.randint(10000000000000000, 99999999999999999)}",
                "vehicle_make": random.choice(["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"]),
                "vehicle_model": random.choice(["Camry", "Civic", "F-150", "Malibu", "Altima"]),
                "vehicle_year": random.randint(2015, 2023),
                "liability_limit": random.choice([50000, 100000, 300000]),
                "collision_deductible": random.choice([250, 500, 1000]),
                "comprehensive_deductible": random.choice([250, 500, 1000]),
                "uninsured_motorist": random.choice([0, 1]),
                "rental_car_coverage": random.choice([0, 1]),
            })

    bills = []
    policy_numbers = [p["policy_number"] for p in policies]
    for i in range(1, 5001):
        bd = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        dd = datetime(2024, 1, 15) + timedelta(days=random.randint(0, 90))
        bills.append({
            "bill_id": f"BILL{str(i).zfill(6)}",
            "policy_number": random.choice(policy_numbers),
            "billing_date": bd.strftime("%Y-%m-%d"),
            "due_date": dd.strftime("%Y-%m-%d"),
            "amount_due": round(random.uniform(100, 1000), 2),
            "status": random.choice(["paid", "pending", "overdue"]),
        })

    bill_ids = [b["bill_id"] for b in bills]
    payments = []
    for i in range(1, 4001):
        pd_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        payments.append({
            "payment_id": f"PAY{str(i).zfill(6)}",
            "bill_id": random.choice(bill_ids),
            "payment_date": pd_date.strftime("%Y-%m-%d"),
            "amount": round(random.uniform(50, 500), 2),
            "payment_method": random.choice(["credit_card", "debit_card", "bank_transfer"]),
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "status": random.choice(["completed", "pending", "failed"]),
        })

    claims = []
    for i in range(1, 301):
        cd = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        claims.append({
            "claim_id": f"CLM{str(i).zfill(6)}",
            "policy_number": random.choice(policy_numbers),
            "claim_date": cd.strftime("%Y-%m-%d"),
            "incident_type": random.choice(["collision", "theft", "property_damage", "medical", "liability"]),
            "estimated_loss": round(random.uniform(500, 20000), 2),
            "status": random.choice(["submitted", "under_review", "approved", "paid", "denied"]),
        })

    return {
        "customers": customers,
        "policies": policies,
        "auto_policy_details": auto_details,
        "billing": bills,
        "payments": payments,
        "claims": claims,
    }


async def seed():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    data = generate_sample_data()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        for table_name, rows in data.items():
            if not rows:
                continue
            cols = list(rows[0].keys())
            col_str = ", ".join(cols)
            param_str = ", ".join(f":{c}" for c in cols)
            stmt = text(f"INSERT INTO {table_name} ({col_str}) VALUES ({param_str})")
            await session.execute(stmt, rows)
        await session.commit()

    await engine.dispose()
    print(f"Database seeded: {', '.join(f'{k}={len(v)}' for k, v in data.items())}")


if __name__ == "__main__":
    asyncio.run(seed())
