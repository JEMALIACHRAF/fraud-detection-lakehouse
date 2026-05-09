"""
scripts/generate_synthetic_transactions.py

Generate 100k realistic synthetic banking transactions with fraud patterns.

Fraud patterns simulated:
1. Card testing — many small transactions in short time
2. Account takeover — sudden large transaction after long inactivity
3. Geographic anomaly — transaction in unusual country
4. Night fraud — high-value transaction between 23:00-05:00
5. Velocity fraud — many transactions in 1 hour

Usage:
    python scripts/generate_synthetic_transactions.py --rows 100000 --output-dir data
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

COUNTRIES = {
    "FR": 0.35, "DE": 0.20, "ES": 0.12, "IT": 0.10,
    "GB": 0.10, "BE": 0.05, "NL": 0.04, "US": 0.04,
}

MERCHANT_CATEGORIES = {
    "grocery":        0.25,
    "restaurant":     0.15,
    "gas_station":    0.10,
    "electronics":    0.08,
    "clothing":       0.08,
    "pharmacy":       0.07,
    "travel":         0.06,
    "entertainment":  0.05,
    "online_retail":  0.08,
    "atm_withdrawal": 0.05,
    "luxury":         0.03,
}

TRANSACTION_TYPES = {
    "purchase":   0.65,
    "withdrawal": 0.15,
    "transfer":   0.10,
    "payment":    0.07,
    "refund":     0.03,
}

CHANNELS = {
    "pos":    0.40,
    "online": 0.35,
    "atm":    0.15,
    "mobile": 0.08,
    "wire":   0.02,
}

CURRENCIES = {
    "EUR": 0.60, "USD": 0.20, "GBP": 0.10,
    "CHF": 0.05, "CAD": 0.03, "JPY": 0.02,
}

# Amount distribution per category (mean, std)
AMOUNT_PARAMS = {
    "grocery":        (45.0,   30.0),
    "restaurant":     (35.0,   25.0),
    "gas_station":    (60.0,   20.0),
    "electronics":    (250.0,  200.0),
    "clothing":       (80.0,   60.0),
    "pharmacy":       (30.0,   20.0),
    "travel":         (400.0,  300.0),
    "entertainment":  (50.0,   40.0),
    "online_retail":  (70.0,   80.0),
    "atm_withdrawal": (150.0,  100.0),
    "luxury":         (800.0,  500.0),
}

CITIES = {
    "FR": ["PARIS", "LYON", "MARSEILLE", "BORDEAUX", "TOULOUSE"],
    "DE": ["BERLIN", "MUNICH", "HAMBURG", "FRANKFURT", "COLOGNE"],
    "ES": ["MADRID", "BARCELONA", "SEVILLE", "VALENCIA", "BILBAO"],
    "IT": ["ROME", "MILAN", "NAPLES", "TURIN", "FLORENCE"],
    "GB": ["LONDON", "MANCHESTER", "BIRMINGHAM", "LEEDS", "GLASGOW"],
    "BE": ["BRUSSELS", "ANTWERP", "GHENT"],
    "NL": ["AMSTERDAM", "ROTTERDAM", "THE_HAGUE"],
    "US": ["NEW_YORK", "LOS_ANGELES", "CHICAGO", "HOUSTON", "MIAMI"],
}


# ── Weighted random choice ─────────────────────────────────────────────────────

def weighted_choice(options: dict) -> str:
    keys = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]


# ── Account pool ──────────────────────────────────────────────────────────────

def generate_accounts(n: int = 5000) -> list[dict]:
    """Generate a pool of realistic bank accounts."""
    accounts = []
    for _ in range(n):
        country = weighted_choice(COUNTRIES)
        accounts.append({
            "account_id":   f"ACC{str(uuid.uuid4())[:8].upper()}",
            "home_country": country,
            "risk_profile": random.choices(
                ["low", "medium", "high"],
                weights=[0.75, 0.20, 0.05]
            )[0],
            "avg_monthly_tx": random.randint(5, 80),
        })
    return accounts


# ── Fraud pattern generators ──────────────────────────────────────────────────

def generate_card_testing_burst(
    account_id: str,
    base_time: datetime,
    merchant_id: str,
) -> list[dict]:
    """Card testing: 5-10 small transactions within 10 minutes."""
    txs = []
    for i in range(random.randint(5, 10)):
        txs.append({
            "_fraud_pattern": "card_testing",
            "account_id":     account_id,
            "merchant_id":    merchant_id,
            "merchant_category": "online_retail",
            "transaction_type": "purchase",
            "channel":        "online",
            "amount":         round(random.uniform(0.50, 5.00), 2),
            "timestamp":      (base_time + timedelta(minutes=i)).isoformat(),
            "is_fraud":       True,
        })
    return txs


def generate_account_takeover(
    account_id: str,
    base_time: datetime,
    home_country: str,
) -> dict:
    """Account takeover: large transaction from foreign country."""
    foreign = random.choice([c for c in COUNTRIES if c != home_country])
    return {
        "_fraud_pattern": "account_takeover",
        "account_id":     account_id,
        "merchant_id":    f"M{str(uuid.uuid4())[:6].upper()}",
        "merchant_category": "electronics",
        "transaction_type": "purchase",
        "channel":        "online",
        "country_code":   foreign,
        "city":           random.choice(CITIES.get(foreign, ["UNKNOWN"])),
        "amount":         round(random.uniform(800.0, 3000.0), 2),
        "timestamp":      base_time.isoformat(),
        "is_fraud":       True,
    }


# ── Normal transaction generator ─────────────────────────────────────────────

def generate_normal_transaction(
    account: dict,
    timestamp: datetime,
) -> dict:
    """Generate a single normal (non-fraud) transaction."""
    category = weighted_choice(MERCHANT_CATEGORIES)
    mean_amount, std_amount = AMOUNT_PARAMS[category]
    amount = max(0.5, round(random.gauss(mean_amount, std_amount), 2))
    country = account["home_country"]

    # Occasional legitimate foreign transaction
    if random.random() < 0.05:
        country = weighted_choice(COUNTRIES)

    return {
        "_fraud_pattern": None,
        "account_id":     account["account_id"],
        "merchant_id":    f"M{str(uuid.uuid4())[:6].upper()}",
        "merchant_category": category,
        "transaction_type": weighted_choice(TRANSACTION_TYPES),
        "channel":        weighted_choice(CHANNELS),
        "country_code":   country,
        "city":           random.choice(CITIES.get(country, ["UNKNOWN"])),
        "currency":       weighted_choice(CURRENCIES),
        "amount":         amount,
        "timestamp":      timestamp.isoformat(),
        "is_fraud":       False,
    }


# ── Main generator ────────────────────────────────────────────────────────────

def generate_transactions(
    n_rows: int = 100_000,
    fraud_rate: float = 0.008,  # ~0.8% fraud — realistic for banking
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict]:
    """
    Generate n_rows realistic banking transactions with embedded fraud patterns.

    Args:
        n_rows: Total number of transactions to generate
        fraud_rate: Target fraud rate (default 0.8%)
        start_date: Start of time range (default 6 months ago)
        end_date: End of time range (default today)

    Returns:
        List of transaction dicts
    """
    random.seed(42)

    end = end_date or datetime.now(timezone.utc)
    start = start_date or (end - timedelta(days=180))
    time_range_seconds = int((end - start).total_seconds())

    accounts = generate_accounts(5000)
    n_fraud_target = int(n_rows * fraud_rate)

    print(f"Generating {n_rows:,} transactions...")
    print(f"  Time range: {start.date()} → {end.date()}")
    print(f"  Accounts: {len(accounts):,}")
    print(f"  Target fraud: {n_fraud_target:,} ({fraud_rate:.1%})")

    transactions = []

    # ── Generate fraud transactions ───────────────────────────────────────────
    fraud_count = 0
    fraud_accounts = random.sample(accounts, min(200, len(accounts)))

    for account in fraud_accounts:
        if fraud_count >= n_fraud_target:
            break

        base_time = start + timedelta(
            seconds=random.randint(0, time_range_seconds)
        )
        pattern = random.choice(["card_testing", "account_takeover", "night_fraud"])

        if pattern == "card_testing":
            burst = generate_card_testing_burst(
                account["account_id"],
                base_time,
                f"M{str(uuid.uuid4())[:6].upper()}",
            )
            transactions.extend(burst)
            fraud_count += len(burst)

        elif pattern == "account_takeover":
            tx = generate_account_takeover(
                account["account_id"],
                base_time,
                account["home_country"],
            )
            transactions.append(tx)
            fraud_count += 1

        elif pattern == "night_fraud":
            # High-value transaction between 23:00 and 05:00
            night_time = base_time.replace(
                hour=random.choice([23, 0, 1, 2, 3, 4]),
                minute=random.randint(0, 59),
            )
            tx = generate_normal_transaction(account, night_time)
            tx["amount"] = round(random.uniform(500.0, 2000.0), 2)
            tx["is_fraud"] = True
            tx["_fraud_pattern"] = "night_fraud"
            transactions.append(tx)
            fraud_count += 1

    # ── Generate normal transactions ──────────────────────────────────────────
    n_normal = n_rows - len(transactions)
    print(f"  Generating {n_normal:,} normal transactions...")

    for _ in range(n_normal):
        account = random.choice(accounts)
        timestamp = start + timedelta(
            seconds=random.randint(0, time_range_seconds)
        )
        transactions.append(generate_normal_transaction(account, timestamp))

    # ── Enrich with IDs and metadata ─────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    enriched = []
    for tx in transactions:
        # Fill missing fields
        if "country_code" not in tx:
            account_map = {a["account_id"]: a for a in accounts}
            acc = account_map.get(tx["account_id"], accounts[0])
            tx["country_code"] = acc["home_country"]
        if "city" not in tx:
            tx["city"] = random.choice(
                CITIES.get(tx["country_code"], ["UNKNOWN"])
            )
        if "currency" not in tx:
            tx["currency"] = weighted_choice(CURRENCIES)

        enriched.append({
            "transaction_id":    f"TX{str(uuid.uuid4())[:12].upper()}",
            "account_id":        tx["account_id"],
            "merchant_id":       tx.get("merchant_id", f"M{str(uuid.uuid4())[:6].upper()}"),
            "merchant_category": tx["merchant_category"],
            "transaction_type":  tx["transaction_type"],
            "channel":           tx["channel"],
            "country_code":      tx["country_code"],
            "city":              tx["city"],
            "currency":          tx["currency"],
            "amount":            tx["amount"],
            "timestamp":         tx["timestamp"],
            "is_fraud":          tx["is_fraud"],
            "created_at":        tx["timestamp"],
            "_ingested_at":      now,
            "_source":           "synthetic_banking",
            "_fraud_pattern":    tx.get("_fraud_pattern"),
        })

    # Shuffle to mix fraud and normal
    random.shuffle(enriched)

    actual_fraud = sum(1 for t in enriched if t["is_fraud"])
    print(f"\nGeneration complete:")
    print(f"  Total:      {len(enriched):,}")
    print(f"  Fraud:      {actual_fraud:,} ({actual_fraud/len(enriched):.2%})")
    print(f"  Normal:     {len(enriched)-actual_fraud:,}")

    return enriched


# ── Write to Bronze ───────────────────────────────────────────────────────────

def write_bronze(
    transactions: list[dict],
    output_dir: str,
    batch_size: int = 10_000,
) -> None:
    """Write transactions to Bronze layer as JSON Lines, partitioned by day."""

    # Group by date for realistic partitioning
    by_date: dict[str, list] = {}
    for tx in transactions:
        date_str = tx["timestamp"][:10] if tx["timestamp"] else "2024-01-15"
        if date_str not in by_date:
            by_date[date_str] = []
        by_date[date_str].append(tx)

    print(f"\nWriting to Bronze layer: {output_dir}")
    total_files = 0

    for date_str, txs in sorted(by_date.items()):
        year, month, day = date_str.split("-")
        partition_path = Path(output_dir) / "bronze" / "transactions" \
            / f"year={year}" / f"month={month}" / f"day={day}"
        partition_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, i in enumerate(range(0, len(txs), batch_size)):
            batch = txs[i:i + batch_size]
            output_file = partition_path / f"batch_{batch_idx:04d}.jsonl"
            with open(output_file, "w") as f:
                for tx in batch:
                    f.write(json.dumps(tx) + "\n")
            total_files += 1

    print(f"  Partitions: {len(by_date):,} days")
    print(f"  Files:      {total_files:,}")
    print(f"  Path:       {Path(output_dir) / 'bronze' / 'transactions'}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic banking transactions for fraud detection"
    )
    parser.add_argument("--rows",       type=int,   default=100_000, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.008,   help="Fraud rate (default 0.8%%)")
    parser.add_argument("--output-dir", type=str,   default="data",  help="Output directory")
    parser.add_argument("--batch-size", type=int,   default=10_000,  help="Rows per file")
    args = parser.parse_args()

    transactions = generate_transactions(
        n_rows=args.rows,
        fraud_rate=args.fraud_rate,
    )

    write_bronze(
        transactions=transactions,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
