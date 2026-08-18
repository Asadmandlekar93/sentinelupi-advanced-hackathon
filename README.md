# SentinelUPI Advanced
## Intelligent UPI Fraud Detection & Prevention — Hackathon MVP

This is a **simulation/prototype**, not a live UPI/bank integration.

### What is new in this version
- Random Forest fraud classifier trained at startup on deterministic synthetic behavioural transaction data
- Isolation Forest anomaly detector
- Hybrid risk score: ML probability + anomaly score + rule-based behavioural signals
- Explainable risk factors with contribution scores
- User behavioural profile and deviation analysis
- Velocity/burst detection
- New device / new beneficiary risk
- Unusual time and location risk
- Impossible-travel / geo-velocity simulation
- Risk-based intervention: ALLOW, MONITOR, STEP-UP VERIFICATION, BLOCK
- Transaction decision ledger in SQLite
- Live monitoring dashboard
- Risk distribution, fraud-rate and intervention statistics
- Transaction replay/simulation mode
- Model health panel and feature importance
- API endpoint for external transaction simulators
- Seeded demo data for a convincing dashboard

### Important
The system does NOT access PhonePe, Google Pay, Paytm, a bank account, or the UPI network.
The transaction stream is simulated for a hackathon demonstration.

### Run

```powershell
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
python app.py
```

Open:
http://127.0.0.1:5000

### API
POST `/api/analyze`

Example JSON:

```json
{
  "user_id": "U1007",
  "amount": 50000,
  "avg_amount": 1500,
  "std_amount": 600,
  "hour": 2,
  "frequency_5m": 8,
  "device": "NEW",
  "location": "UNUSUAL",
  "beneficiary": "NEW",
  "beneficiary_risk": 0.72,
  "distance_km": 850,
  "minutes_since_last": 4,
  "failed_attempts": 0
}
```

The response includes:
- fraud probability
- anomaly score
- hybrid risk score
- risk level
- intervention
- explanation factors
- model confidence
- transaction ID
