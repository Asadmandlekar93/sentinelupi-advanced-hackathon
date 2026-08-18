from flask import Flask, render_template, request, jsonify
import sqlite3, os, json, math, random, joblib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

BASE = os.path.dirname(__file__)
DB = os.path.join(BASE, "sentinelupi.db")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)

FEATURES = [
    "amount_ratio","amount_z","hour_risk","velocity_5m","new_device",
    "new_beneficiary","location_risk","beneficiary_risk","geo_velocity",
    "minutes_since_last","failed_attempts","account_age_days"
]

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id TEXT UNIQUE, user_id TEXT, amount REAL, fraud_prob REAL,
            anomaly REAL, risk REAL, level TEXT, intervention TEXT,
            decision TEXT, reasons TEXT, features TEXT, created_at TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS users(
            user_id TEXT PRIMARY KEY, avg_amount REAL, std_amount REAL,
            home_lat REAL, home_lon REAL, known_devices INTEGER,
            account_age_days INTEGER, trust_score REAL
        )""")
        con.commit()

def make_training_data(n=14000, seed=42):
    rng = np.random.default_rng(seed)
    amount_ratio = np.exp(rng.normal(0.0, 0.65, n))
    amount_z = rng.normal(0, 1, n)
    hour = rng.integers(0,24,n)
    hour_risk = np.where((hour<5)|(hour>=23),1.0,np.where((hour<7),.55,0.05))
    velocity = rng.poisson(1.3,n)+1
    new_device = rng.binomial(1,0.09,n)
    new_beneficiary = rng.binomial(1,0.14,n)
    location_risk = rng.beta(1.2,7,n)
    beneficiary_risk = rng.beta(1.5,8,n)
    geo_velocity = np.maximum(0,rng.normal(25,45,n))
    mins = np.maximum(1,rng.exponential(55,n))
    failed = rng.poisson(.15,n)
    account_age = np.maximum(5,rng.gamma(3,120,n))
    raw = (
        1.0*(amount_ratio>5) + 0.8*(amount_ratio>10) +
        .7*(np.abs(amount_z)>3) + .8*hour_risk +
        .55*(velocity>=5) + .8*new_device + .65*new_beneficiary +
        1.1*location_risk + .9*beneficiary_risk +
        .8*(geo_velocity>300) + .6*(mins<5) + .45*(failed>=2) +
        .35*(account_age<30)
    )
    p = 1/(1+np.exp(-(raw-2.8)))
    y = rng.binomial(1,p)
    X = pd.DataFrame({
        "amount_ratio":amount_ratio,"amount_z":amount_z,"hour_risk":hour_risk,
        "velocity_5m":velocity,"new_device":new_device,
        "new_beneficiary":new_beneficiary,"location_risk":location_risk,
        "beneficiary_risk":beneficiary_risk,"geo_velocity":geo_velocity,
        "minutes_since_last":mins,"failed_attempts":failed,
        "account_age_days":account_age
    })
    return X, y

def train_models():
    X,y = make_training_data()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(
        n_estimators=260,max_depth=12,min_samples_leaf=3,
        class_weight="balanced_subsample",random_state=42,n_jobs=-1
    )
    clf.fit(Xtr,ytr)
    iso = IsolationForest(n_estimators=180,contamination=.08,random_state=42)
    iso.fit(Xtr)
    prob = clf.predict_proba(Xte)[:,1]
    pred = (prob>=.5).astype(int)
    metrics = {
        "auc": round(float(roc_auc_score(yte,prob)),3),
        "precision": round(float(precision_score(yte,pred,zero_division=0)),3),
        "recall": round(float(recall_score(yte,pred,zero_division=0)),3),
        "samples": len(X)
    }
    joblib.dump(clf, os.path.join(MODEL_DIR,"fraud_rf.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR,"anomaly_iso.joblib"))
    with open(os.path.join(MODEL_DIR,"metrics.json"),"w") as f: json.dump(metrics,f)
    return clf,iso,metrics

def load_models():
    try:
        clf=joblib.load(os.path.join(MODEL_DIR,"fraud_rf.joblib"))
        iso=joblib.load(os.path.join(MODEL_DIR,"anomaly_iso.joblib"))
        with open(os.path.join(MODEL_DIR,"metrics.json")) as f: metrics=json.load(f)
        return clf,iso,metrics
    except Exception:
        return train_models()

CLF, ISO, MODEL_METRICS = load_models()

def seed_users():
    users = [
        ("U1001",1200,450,19.076,72.877,2,420,91),
        ("U1002",2500,900,19.218,72.978,1,780,86),
        ("U1007",1500,600,18.989,73.117,2,540,88),
        ("U1022",800,300,19.033,73.029,1,180,74),
    ]
    with sqlite3.connect(DB) as con:
        for u in users:
            con.execute("INSERT OR REPLACE INTO users VALUES(?,?,?,?,?,?,?,?)",u)
        con.commit()

def get_user(uid):
    with sqlite3.connect(DB) as con:
        row=con.execute("SELECT * FROM users WHERE user_id=?",(uid,)).fetchone()
    if row: return dict(zip(["user_id","avg_amount","std_amount","home_lat","home_lon","known_devices","account_age_days","trust_score"],row))
    return {"user_id":uid,"avg_amount":1500,"std_amount":600,"home_lat":19.076,"home_lon":72.877,"known_devices":1,"account_age_days":365,"trust_score":80}

def clamp(v,a=0,b=1): return max(a,min(b,v))

def feature_vector(d):
    avg=max(float(d.get("avg_amount",1500)),1)
    std=max(float(d.get("std_amount",600)),50)
    amount=float(d.get("amount",0))
    ratio=amount/avg
    z=(amount-avg)/std
    hour=int(d.get("hour",14))
    hour_risk=1.0 if hour<5 or hour>=23 else (.55 if hour<7 else .05)
    velocity=int(d.get("frequency_5m",1))
    new_device=1 if str(d.get("device","KNOWN")).upper()=="NEW" else 0
    new_ben=1 if str(d.get("beneficiary","KNOWN")).upper()=="NEW" else 0
    location=str(d.get("location","NORMAL")).upper()
    location_risk=0.85 if location=="UNUSUAL" else .05
    ben_risk=float(d.get("beneficiary_risk",.08))
    distance=float(d.get("distance_km",10))
    mins=max(float(d.get("minutes_since_last",60)),1)
    geo_velocity=distance/mins*60
    failed=int(d.get("failed_attempts",0))
    age=int(d.get("account_age_days",365))
    values=[ratio,z,hour_risk,velocity,new_device,new_ben,location_risk,ben_risk,geo_velocity,mins,failed,age]
    return np.array(values,dtype=float), {
        "amount_ratio":ratio,"amount_z":z,"hour_risk":hour_risk,"velocity_5m":velocity,
        "new_device":new_device,"new_beneficiary":new_ben,"location_risk":location_risk,
        "beneficiary_risk":ben_risk,"geo_velocity":geo_velocity,"minutes_since_last":mins,
        "failed_attempts":failed,"account_age_days":age
    }

def explain(f):
    candidates=[]
    if f["amount_ratio"]>=10: candidates.append(("Extreme amount deviation",min(.99,.32+f["amount_ratio"]/100)))
    elif f["amount_ratio"]>=4: candidates.append(("High amount vs normal behaviour",.21))
    elif abs(f["amount_z"])>=2: candidates.append(("Amount is statistically unusual",.14))
    if f["new_device"]: candidates.append(("New device detected",.19))
    if f["new_beneficiary"]: candidates.append(("New beneficiary detected",.13))
    if f["hour_risk"]>=.9: candidates.append(("High-risk transaction time",.17))
    elif f["hour_risk"]>.4: candidates.append(("Unusual transaction hour",.10))
    if f["velocity_5m"]>=8: candidates.append(("Transaction burst detected",.18))
    elif f["velocity_5m"]>=4: candidates.append(("Elevated transaction velocity",.10))
    if f["location_risk"]>.5: candidates.append(("Location anomaly",.13))
    if f["geo_velocity"]>300: candidates.append(("Impossible-travel / geo-velocity signal",.18))
    elif f["geo_velocity"]>120: candidates.append(("Unusual travel velocity",.08))
    if f["beneficiary_risk"]>.6: candidates.append(("Beneficiary risk elevated",.16))
    if f["failed_attempts"]>=2: candidates.append(("Multiple failed authentication attempts",.10))
    if f["account_age_days"]<30: candidates.append(("New account risk",.08))
    candidates.sort(key=lambda x:x[1],reverse=True)
    return [{"label":a,"weight":round(b,2)} for a,b in candidates[:6]]

def analyze(d):
    x,f=feature_vector(d)
    X=pd.DataFrame([f],columns=FEATURES)
    fraud_prob=float(CLF.predict_proba(X)[0,1])
    raw_anom=float(-ISO.score_samples(X)[0])
    anom=clamp((raw_anom-.35)/.35)
    factors=explain(f)
    rule_score=sum(x["weight"] for x in factors)
    hybrid=clamp(.58*fraud_prob+.22*anom+.20*min(rule_score,1))
    # Make the output a risk score, not a claim of actual fraud.
    risk=round(hybrid*100,1)
    if risk>=85: level,intervention,decision="CRITICAL","BLOCK & STEP-UP VERIFY","BLOCKED"
    elif risk>=65: level,intervention,decision="HIGH RISK","STEP-UP VERIFICATION","VERIFICATION"
    elif risk>=40: level,intervention,decision="SUSPICIOUS","MONITOR + VERIFY","MONITORED"
    else: level,intervention,decision="SAFE","ALLOW","ALLOWED"
    confidence=round(min(99,max(52,50+abs(fraud_prob-.5)*100)),1)
    return {"fraud_probability":round(fraud_prob*100,1),"anomaly_score":round(anom*100,1),
            "risk":risk,"level":level,"intervention":intervention,"decision":decision,
            "confidence":confidence,"factors":factors,"features":f}

def seed_demo_transactions():
    with sqlite3.connect(DB) as con:
        count=con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if count>0: return
    base=datetime.now()
    demos=[
        {"user_id":"U1001","amount":650,"avg_amount":1200,"std_amount":450,"hour":18,"frequency_5m":1,"device":"KNOWN","location":"NORMAL","beneficiary":"KNOWN","beneficiary_risk":.04,"distance_km":5,"minutes_since_last":120,"failed_attempts":0,"account_age_days":420},
        {"user_id":"U1002","amount":4200,"avg_amount":2500,"std_amount":900,"hour":21,"frequency_5m":3,"device":"KNOWN","location":"NORMAL","beneficiary":"KNOWN","beneficiary_risk":.12,"distance_km":8,"minutes_since_last":30,"failed_attempts":0,"account_age_days":780},
        {"user_id":"U1007","amount":28000,"avg_amount":1500,"std_amount":600,"hour":2,"frequency_5m":9,"device":"NEW","location":"UNUSUAL","beneficiary":"NEW","beneficiary_risk":.78,"distance_km":850,"minutes_since_last":4,"failed_attempts":2,"account_age_days":540},
        {"user_id":"U1022","amount":18000,"avg_amount":800,"std_amount":300,"hour":3,"frequency_5m":7,"device":"NEW","location":"UNUSUAL","beneficiary":"KNOWN","beneficiary_risk":.35,"distance_km":600,"minutes_since_last":8,"failed_attempts":1,"account_age_days":180},
        {"user_id":"U1001","amount":1100,"avg_amount":1200,"std_amount":450,"hour":13,"frequency_5m":1,"device":"KNOWN","location":"NORMAL","beneficiary":"KNOWN","beneficiary_risk":.04,"distance_km":4,"minutes_since_last":90,"failed_attempts":0,"account_age_days":420},
    ]
    with sqlite3.connect(DB) as con:
        for d in demos:
            r=analyze(d); tx="TX-DEMO-"+str(random.randint(10000,99999))
            con.execute("""INSERT INTO transactions(tx_id,user_id,amount,fraud_prob,anomaly,risk,level,intervention,decision,reasons,features,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(tx,d["user_id"],d["amount"],r["fraud_probability"],r["anomaly_score"],r["risk"],r["level"],r["intervention"],r["decision"],json.dumps(r["factors"]),json.dumps(r["features"]),(base-timedelta(minutes=random.randint(2,180))).isoformat(timespec="seconds")))
        con.commit()

init_db(); seed_users(); seed_demo_transactions()

@app.route("/")
def index(): return render_template("index.html")

@app.get("/api/model")
def model():
    imp=dict(sorted(zip(FEATURES,CLF.feature_importances_),key=lambda x:x[1],reverse=True)[:7])
    labels={"amount_ratio":"Amount deviation","amount_z":"Amount z-score","hour_risk":"Time anomaly","velocity_5m":"Velocity","new_device":"New device","new_beneficiary":"New beneficiary","location_risk":"Location anomaly","beneficiary_risk":"Beneficiary risk","geo_velocity":"Geo-velocity","minutes_since_last":"Time since last TX","failed_attempts":"Failed attempts","account_age_days":"Account age"}
    return jsonify({"metrics":MODEL_METRICS,"importance":[{"label":labels[k],"value":round(v*100,1)} for k,v in imp.items()]})

@app.post("/api/analyze")
def api_analyze():
    d=request.get_json(force=True)
    r=analyze(d)
    tx="TX-"+datetime.now().strftime("%H%M%S")+"-"+str(random.randint(10,99))
    with sqlite3.connect(DB) as con:
        con.execute("""INSERT INTO transactions VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tx,d.get("user_id","U1007"),float(d.get("amount",0)),r["fraud_probability"],r["anomaly_score"],r["risk"],r["level"],r["intervention"],r["decision"],json.dumps(r["factors"]),json.dumps(r["features"]),datetime.now().isoformat(timespec="seconds")))
        con.commit()
    r["tx_id"]=tx
    return jsonify(r)

@app.get("/api/stats")
def stats():
    with sqlite3.connect(DB) as con:
        rows=con.execute("SELECT tx_id,user_id,amount,fraud_prob,anomaly,risk,level,intervention,decision,created_at FROM transactions ORDER BY id DESC LIMIT 100").fetchall()
    total=len(rows)
    safe=sum(x[6]=="SAFE" for x in rows); suspicious=sum(x[6]=="SUSPICIOUS" for x in rows); high=sum(x[6] in ("HIGH RISK","CRITICAL") for x in rows)
    blocked=sum(x[8]=="BLOCKED" for x in rows); verify=sum(x[8]=="VERIFICATION" for x in rows)
    avg=round(sum(x[5] for x in rows)/total,1) if total else 0
    fraud_rate=round((high/total)*100,1) if total else 0
    return jsonify({"total":total,"safe":safe,"suspicious":suspicious,"high":high,"blocked":blocked,"verify":verify,"avg_risk":avg,"fraud_rate":fraud_rate,
        "transactions":[{"tx_id":x[0],"user_id":x[1],"amount":x[2],"fraud_prob":x[3],"anomaly":x[4],"risk":x[5],"level":x[6],"intervention":x[7],"decision":x[8],"created_at":x[9]} for x in rows[:15]]})

@app.post("/api/reset")
def reset():
    with sqlite3.connect(DB) as con: con.execute("DELETE FROM transactions"); con.commit()
    seed_demo_transactions()
    return jsonify({"ok":True})

if __name__=="__main__":
    app.run(debug=True,port=5000)
