"""
Fake Job Detection API

This API uses a trained BiLSTM deep learning model to predict if job postings are fraudulent.

IMPORTANT: Model Configuration
- MAX_LEN = 150 (must match deep_learning_model.py)
- MAX_WORDS = 5000 (must match deep_learning_model.py)
- Preprocessing must match data_preprocessing.py exactly
- Model expects cleaned text (after preprocessing pipeline)

The model was trained with class weights to handle imbalanced data and uses
an improved BiLSTM architecture for better fraud detection.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError

SECRET_KEY = "milestone4_secret_key"
ALGORITHM = "HS256"

security = HTTPBearer()

FAKE_USER = {
    "username": "admin",
    "password": "admin123"
}


import pickle
import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

MAX_LEN = 150  # Match deep_learning_model.py
MAX_WORDS = 5000  # Match deep_learning_model.py

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Add custom stop words (matching data_preprocessing.py)
custom_stop_words = {
    'job', 'work', 'company', 'position', 'apply', 'experience',
    'candidate', 'role', 'opportunity', 'team', 'looking'
}
stop_words.update(custom_stop_words)

# Load model and tokenizer with error handling
# Priority: bilstm_model_best.h5 (best validation performance) > bilstm_model_v1.h5 (final state)
# The "best" model is saved by ModelCheckpoint during training and has the lowest validation loss
model_path = None
model_version = None
if os.path.exists("models/bilstm_model_best.h5"):
    model_path = "models/bilstm_model_best.h5"
    model_version = "bilstm_model_best.h5 (best validation performance)"
    print(f"Using {model_version}")
elif os.path.exists("models/bilstm_model_v1.h5"):
    model_path = "models/bilstm_model_v1.h5"
    model_version = "bilstm_model_v1.h5 (final model state)"
    print(f"Using {model_version} (fallback)")

try:
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Expected: models/bilstm_model_best.h5 or models/bilstm_model_v1.h5")
    # Load model (it should be saved with compilation)
    model = load_model(model_path, compile=True)
    print(f"✓ BiLSTM MODEL LOADED SUCCESSFULLY: {model_version}")
except Exception as e:
    try:
        # Try loading without compilation and then compile
        model = load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        print(f"✓ BiLSTM MODEL LOADED AND COMPILED SUCCESSFULLY: {model_version}")
    except Exception as e2:
        model = None
        print(f"✗ Error loading model: {e2}")

try:
    tokenizer_path = "models/tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ TOKENIZER LOADED SUCCESSFULLY")
except Exception as e:
    tokenizer = None
    print(f"✗ Error loading tokenizer: {e}")

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data (matching data_preprocessing.py).
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        # Fallback if tokenization fails
        tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def predict_text(text: str) -> float:
    """
    Preprocess text and make prediction using the BiLSTM model.
    This function matches the exact preprocessing pipeline used during training.
    
    Args:
        text: Raw input text (job description, title, requirements, etc.)
        
    Returns:
        Probability of being fraudulent (0-1), where > 0.5 indicates fraudulent
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or tokenizer not loaded. Please check server logs and ensure models are trained."
        )
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty"
        )
    try:
        # CRITICAL: Preprocess text to match training data format exactly
        # This must match the clean_text function in data_preprocessing.py
        cleaned_text = clean_text(text)
        
        if not cleaned_text or cleaned_text.strip() == '':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text became empty after preprocessing. Please provide more meaningful text with actual words."
            )
        
        # Tokenize using the saved tokenizer (matching training pipeline)
        # The tokenizer was fit on MAX_WORDS=5000 with oov_token="<OOV>"
        seq = tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad sequences to MAX_LEN=150 (matching training configuration)
        # Using "post" padding and truncation to match training
        padded = pad_sequences(
            seq, 
            maxlen=MAX_LEN, 
            padding="post", 
            truncating="post"
        )
        
        # Make prediction
        # The model outputs a single value between 0 and 1 (sigmoid activation)
        prob = model.predict(padded, verbose=0)[0][0]
        return float(prob)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )

app = FastAPI(title="Fake Job Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_token(username: str):
    return jwt.encode({"sub": username}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
class LoginInput(BaseModel):
    username: str
    password: str

@app.post("/")
def root():
    return {"message": "Welcome to the Fake Job Detection API"}

@app.post("/login")
def login(data: LoginInput):
    if data.username == FAKE_USER["username"] and data.password == FAKE_USER["password"]:
        token = create_token(data.username)
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

class JobInput(BaseModel):
    description: str

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API and model status.
    Returns detailed information about model availability and configuration.
    """
    status_info = {
        "status": "healthy" if (model is not None and tokenizer is not None) else "degraded",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_version": model_version if model is not None else None,
        "model_config": {
            "max_sequence_length": MAX_LEN,
            "max_vocab_size": MAX_WORDS
        }
    }
    
    if model is None:
        status_info["error"] = "Model not loaded. Please train the model first using deep_learning_model.py"
    if tokenizer is None:
        status_info["error"] = "Tokenizer not loaded. Please train the model first using deep_learning_model.py"
    
    return status_info

@app.post("/predict")
def predict(job: JobInput, user: str = Depends(verify_token)):
    """
    Predict if a job posting is fraudulent or legitimate.
    
    The model uses an improved BiLSTM architecture trained with class weights
    to handle imbalanced data (fraudulent vs legitimate jobs).
    
    Args:
        job: JobInput containing the job description text
        user: Authenticated user (from JWT token)
    
    Returns:
        Dictionary with prediction results including:
        - prediction: "Fraudulent" or "Legitimate"
        - fraud_probability: Probability score (0-1)
        - confidence: Confidence in the prediction (0-1)
        - interpretation: Human-readable explanation
    """
    if not job.description or not job.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty"
        )
    
    probability = predict_text(job.description)
    label = "Fraudulent" if probability > 0.5 else "Legitimate"
    confidence = abs(probability - 0.5) * 2  # Convert to 0-1 confidence scale
    
    # Provide interpretation based on probability
    if probability >= 0.7:
        interpretation = "High likelihood of being fraudulent"
    elif probability >= 0.5:
        interpretation = "Moderate likelihood of being fraudulent"
    elif probability >= 0.3:
        interpretation = "Moderate likelihood of being legitimate"
    else:
        interpretation = "High likelihood of being legitimate"

    return {
        "prediction": label,
        "fraud_probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "interpretation": interpretation
    }
