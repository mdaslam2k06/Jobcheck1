import numpy as np
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("="*60)
print("Loading and Preparing Data")
print("="*60)

df = pd.read_csv("processed_data/processed_jobs.csv")

# Check for empty or null cleaned_text
df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.len() > 0)]

texts = df["cleaned_text"].values
labels = df["fraudulent"].values

print(f"Total samples: {len(texts)}")
print(f"Class distribution: Real={np.sum(labels==0)}, Fraudulent={np.sum(labels==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights_list = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights_list[0], 1: class_weights_list[1]}

print(f"\nClass weights: {class_weights}")

# Optimized settings for 8 GB RAM
MAX_WORDS = 5000  # Reduced to prevent overfitting on small dataset
MAX_LEN = 150  # Reduced sequence length

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print(f"\nVocabulary size: {len(tokenizer.word_index) + 1}")
print(f"Training sequences shape: {X_train_pad.shape}")
print(f"Test sequences shape: {X_test_pad.shape}")

print("\n" + "="*60)
print("Building BiLSTM Model")
print("="*60)

# Improved model architecture for better feature learning
model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,  # Increased embedding dimension
        input_length=MAX_LEN,
        mask_zero=True
    ),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Dropout(0.3),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    Dropout(0.4),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"]
)

model.summary()

# Enhanced callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/bilstm_model_best.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("\n" + "="*60)
print("Training Model")
print("="*60)

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=30,  # More epochs with early stopping
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights,  # Apply class weights
    verbose=1
)

print("\n" + "="*60)
print("Evaluating Model")
print("="*60)

y_pred_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError:
    roc_auc = 0.0

print(f"\nBiLSTM Model Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fraudulent']))

# Save final model
model.save("models/bilstm_model_v1.h5")
print("\n✓ Model saved to models/bilstm_model_v1.h5")

# Save tokenizer
import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved to models/tokenizer.pkl")
