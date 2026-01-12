import { useState } from "react";
import "./App.css";

function App() {
  const [token, setToken] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [validationErrors, setValidationErrors] = useState({});
  const [showGuide, setShowGuide] = useState(false);

  function validateLogin() {
    const errors = {};
    
    if (!username.trim()) {
      errors.username = "Username is required";
    } else if (username.trim().length < 3) {
      errors.username = "Username must be at least 3 characters";
    }
    
    if (!password.trim()) {
      errors.password = "Password is required";
    } else if (password.trim().length < 3) {
      errors.password = "Password must be at least 3 characters";
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  }

  async function login() {
    setError("");
    setValidationErrors({});
    
    if (!validateLogin()) {
      return;
    }

    setLoginLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password: password.trim() })
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Login failed");
      }
      
      const data = await res.json();
      setToken(data.access_token);
      setError("");
      setUsername("");
      setPassword("");
      setValidationErrors({});
    } catch (err) {
      setError(err.message || "Failed to login. Please try again.");
    } finally {
      setLoginLoading(false);
    }
  }

  function handleLoginKeyPress(e) {
    if (e.key === "Enter") {
      login();
    }
  }

  async function predict() {
    if (!text.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ description: text })
      });

      if (!res.ok) {
        if (res.status === 401) {
          setToken("");
          throw new Error("Session expired. Please login again.");
        }
        const errorData = await res.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to analyze job. Please try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    setToken("");
    setText("");
    setResult(null);
    setError("");
    setUsername("");
    setPassword("");
    setValidationErrors({});
  }

  const isFraudulent = result?.prediction === "Fraudulent";
  const probability = result?.fraud_probability || 0;

  return (
    <div className="app-container">
      <div className="app-card">
        <header className="app-header">
          <h1>🔍 Aslam's Jobcheck</h1>
          <p className="subtitle">AI-powered job posting analyzer</p>
        </header>

        {!token ? (
          <div className="login-section">
            <div className="login-card">
              <h2>Welcome</h2>
              <p>Login to start analyzing job postings</p>
              
              {error && <div className="error-message">{error}</div>}
              
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  type="text"
                  className={`form-input ${validationErrors.username ? "input-error" : ""}`}
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => {
                    setUsername(e.target.value);
                    if (validationErrors.username) {
                      setValidationErrors({ ...validationErrors, username: "" });
                    }
                  }}
                  onKeyPress={handleLoginKeyPress}
                  disabled={loginLoading}
                  autoComplete="username"
                />
                {validationErrors.username && (
                  <div className="field-error">{validationErrors.username}</div>
                )}
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  type="password"
                  className={`form-input ${validationErrors.password ? "input-error" : ""}`}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    if (validationErrors.password) {
                      setValidationErrors({ ...validationErrors, password: "" });
                    }
                  }}
                  onKeyPress={handleLoginKeyPress}
                  disabled={loginLoading}
                  autoComplete="current-password"
                />
                {validationErrors.password && (
                  <div className="field-error">{validationErrors.password}</div>
                )}
              </div>

              <button 
                className="btn btn-primary btn-large" 
                onClick={login}
                disabled={loginLoading}
              >
                {loginLoading ? (
                  <>
                    <span className="spinner"></span>
                    Logging in...
                  </>
                ) : (
                  "Login"
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="main-section">
            <div className="header-actions">
              <button 
                className="how-to-toggle" 
                onClick={() => setShowGuide(!showGuide)}
                aria-expanded={showGuide}
              >
                <span>{showGuide ? '📖' : '❓'}</span>
                {showGuide ? 'Hide Guide' : 'How to Use'}
              </button>
              <button className="btn btn-secondary" onClick={logout}>
                Logout
              </button>
            </div>

            {showGuide && (
              <div className="how-to-guide">
                <h3>How to Use This App</h3>
                <ol className="how-to-steps">
                  <li>
                    <strong>Login:</strong> Use your credentials to access the analyzer. 
                    Default credentials are provided for testing.
                  </li>
                  <li>
                    <strong>Paste Job Description:</strong> Copy and paste the complete job 
                    posting description into the text area. The more details you include, 
                    the more accurate the analysis will be.
                  </li>
                  <li>
                    <strong>Analyze:</strong> Click the "Analyze Job Posting" button to 
                    process the description through our AI-powered fraud detection system.
                  </li>
                  <li>
                    <strong>Review Results:</strong> The system will display:
                    <ul style={{ marginTop: '8px', paddingLeft: '20px', listStyle: 'disc' }}>
                      <li>Prediction (Legitimate or Fraudulent)</li>
                      <li>Fraud probability percentage</li>
                      <li>Confidence score</li>
                      <li>Detailed analysis message</li>
                    </ul>
                  </li>
                  <li>
                    <strong>Make Informed Decisions:</strong> Use the analysis results along 
                    with your own judgment to evaluate job opportunities safely.
                  </li>
                </ol>
                <div className="how-to-tips">
                  <strong>💡 Tips for Best Results:</strong>
                  <p>
                    Include the full job description with all details like company information, 
                    job requirements, salary details, and contact information. The AI analyzes 
                    patterns, language, and red flags that are common in fraudulent postings.
                  </p>
                </div>
              </div>
            )}

            <div className="input-section">
              <label htmlFor="job-description">Job Description</label>
              <textarea
                id="job-description"
                className="job-textarea"
                rows="8"
                placeholder="Paste the job description here to analyze if it's legitimate or fraudulent..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={loading}
              />
              <div className="char-count">{text.length} characters</div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button
              className="btn btn-primary btn-large"
              onClick={predict}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                "Analyze Job Posting"
              )}
            </button>

            {result && (
              <div className={`result-card ${isFraudulent ? "fraudulent" : "legitimate"}`}>
                <div className="result-header">
                  <span className="result-icon">
                    {isFraudulent ? "⚠️" : "✅"}
                  </span>
                  <h2 className="result-title">{result.prediction}</h2>
                </div>
                
                <div className="result-details">
                  <div className="probability-bar">
                    <div className="probability-label">
                      Fraud Probability: {(probability * 100).toFixed(1)}%
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="confidence-score">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="result-message">
                  {isFraudulent
                    ? "⚠️ This job posting shows signs of being fraudulent. Exercise caution."
                    : "✅ This job posting appears to be legitimate."}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
