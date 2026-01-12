
# Local Setup & Execution Guide


This guide explains how to set up and run the **Fake Job Detection System** locally, covering all milestones from data preparation to authenticated web application.

---

## Prerequisites

Ensure the following are installed on your system:

* Python **3.10+** (tested on Python 3.12)
* Node.js **18+**
* npm (comes with Node.js)
* Git (optional)
* 8 GB RAM or higher

---

## Project Structure (Final)

```
project/
│
├── src/
│   ├── api.py
│   ├── model_utils.py
│   ├── deep_learning_model.py
│
├── models/
│   ├── bilstm_model_v1.h5
│   └── tokenizer.pkl
│
├── processed_data/
│   └── processed_jobs.csv
│
├── frontend/          # React (Vite) app
│
├── requirements.txt
├── README.md
└── guide.md
```

---

# Milestone 1 — Environment Setup & Project Initialization

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Milestone 1 is complete when:

* Project structure is created
* Virtual environment is active
* Dependencies are installed successfully

---

# Milestone 2 — Model Training & Evaluation

This milestone includes classical ML models and a deep learning BiLSTM model.

### 1. Ensure processed dataset exists

```
processed_data/processed_jobs.csv
```

Required columns:

* `cleaned_text`
* `fraudulent`

### 2. Train Deep Learning model (BiLSTM)

From project root:

```bash
python src/deep_learning_model.py
```

This will:

* Train an optimized BiLSTM (8 GB RAM safe)
* Evaluate accuracy and F1 score
* Save model to `models/`

  * `bilstm_model_v1.h5`
  * `tokenizer.pkl`

Milestone 2 is complete when:

* Training finishes without errors
* Model files are saved

---

# Milestone 3 — API & Basic Frontend

### 1. Start FastAPI backend

```bash
uvicorn src.api:app --reload
```

Backend will run at:

```
http://127.0.0.1:8000
```

### 2. Verify API

Open in browser:

```
http://127.0.0.1:8000/docs
```

You should see:

* `/login`
* `/predict` (secured endpoint)

Milestone 3 is complete when:

* API starts successfully
* Swagger UI loads

---

# Milestone 4 — Authentication & React UI (Vite)

## Backend (Authentication)

Authentication is implemented using **JWT (Bearer Token)**.

### Test login (via Swagger or frontend)

* Username: `admin`
* Password: `admin123`

---

## Frontend (React + Vite)

### 1. Navigate to frontend folder

```bash
cd frontend
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Start React app

```bash
npm run dev
```

Frontend will run at:

```
http://127.0.0.1:5173
```

### 4. Application Flow

1. Click **Login**
2. JWT token is generated
3. Enter job description
4. Click **Check Job**
5. Prediction is displayed

Milestone 4 is complete when:

* Login works
* Protected `/predict` endpoint is accessible
* Prediction result is shown in UI

---

# Running Order (Quick Reference)

```bash
# Activate environment
venv\Scripts\activate

# Backend
uvicorn src.api:app --reload

# Frontend (new terminal)
cd frontend
npm run dev
```


## How to Test (Quick Reminder)

In Swagger UI (`/docs`) or from your frontend, send text to your prediction field (e.g., `job_description`).

---

# ❌ FAKE JOB POSTING EXAMPLES (3)

### Fake Job – Example 1

```
We are urgently hiring candidates to work from home and earn up to ₹50,000 per week. No experience required. Registration fee is mandatory to activate your job ID. Immediate joining guaranteed. Limited slots available, apply now.
```

**Why fake indicators exist**

* Unrealistic salary
* Urgency pressure
* Registration fee
* No company details

---

### Fake Job – Example 2

```
Congratulations! You have been shortlisted for an online data entry position. Earn money instantly by completing simple tasks. Payment will be released daily. Submit your bank details and ID proof to get started today.
```

**Why fake indicators exist**

* “Congratulations” without interview
* Requests sensitive information
* Vague job responsibilities

---

### Fake Job – Example 3

```
This is a verified international opportunity. Earn dollars from home by sharing links on social media. No skills required. Investment of ₹2,000 is required to activate your account. High profit guaranteed.
```

**Why fake indicators exist**

* Investment required
* Guaranteed profits
* No employer identity

---

# ✅ REAL JOB POSTING EXAMPLES (3)

### Real Job – Example 1

```
We are hiring a Software Engineer with 2–4 years of experience in Python and REST APIs. Responsibilities include developing backend services, writing unit tests, and collaborating with frontend teams. Candidates should have experience with FastAPI or Django. Location: Bengaluru (Hybrid).
```

**Why real indicators exist**

* Clear role
* Skills required
* Location
* No monetary requests

---

### Real Job – Example 2

```
ABC Technologies Pvt Ltd is looking for a Data Analyst to join our analytics team. The role involves data cleaning, dashboard creation, and report generation using Python and SQL. Bachelor’s degree required. Full-time position with benefits.
```

**Why real indicators exist**

* Company name
* Defined responsibilities
* Educational requirements
* Professional tone

---

### Real Job – Example 3

```
We are seeking a Marketing Executive responsible for campaign planning, market research, and performance analysis. Prior experience in digital marketing tools is preferred. Salary as per industry standards. Office location: Mumbai.
```

**Why real indicators exist**

* Structured description
* Realistic salary phrasing
* No urgency or fees

---

## Expected Model Behavior (High-Level)

| Input Type | Expected Prediction |
| ---------- | ------------------- |
| Fake job   | Fake / Fraudulent   |
| Real job   | Real / Legitimate   |

(Exact labels depend on your model output format.)

---