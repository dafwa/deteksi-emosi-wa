import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import joblib
import os

# Pylance entah kenapa perlu # type: ignore

# Create output folder if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. LOAD & CLEAN DATA
# ==========================================
print("--- Memuat Data ---")
df = pd.read_csv('labeled_dataset.csv') 

df['clean_message'] = df['clean_message'].fillna('')
df['emoji_used'] = df['emoji_used'].fillna('')

# [INFO] DISTRIBUSI KELAS
print("\n" + "="*40)
print("[INFO] DISTRIBUSI KELAS EMOSI")
print("="*40)
print(f"Total Jumlah Data: {len(df)}")
print(df['emotion'].value_counts())
print("="*40 + "\n")

X_text = df['clean_message'].astype(str)
X_emoji_str = df['emoji_used'].astype(str) 
X_numeric = df[['capital_ratio', 'hour', 'emoji_count']]
y = df['emotion']

# ==========================================
# 2. PREPROCESSING & SPLIT
# ==========================================
print("--- Preprocessing ---")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

indices = np.arange(len(df))
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    indices, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_dict = dict(enumerate(class_weights))

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("--- Ekstraksi Fitur ---")

# ngram_range=(1, 2) artinya membaca 1 kata DAN pasangan 2 kata
# Contoh: "tidak suka" akan dianggap sebagai fitur tersendiri
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2)) 
X_text_tfidf = tfidf.fit_transform(X_text).toarray()

emoji_vect = CountVectorizer(token_pattern=r'[^\s]') 
X_emoji_vect = emoji_vect.fit_transform(X_emoji_str).toarray()

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

X_train_text = X_text_tfidf[X_train_idx]
X_test_text = X_text_tfidf[X_test_idx]
X_train_emoji = X_emoji_vect[X_train_idx]
X_test_emoji = X_emoji_vect[X_test_idx]
X_train_num = X_numeric_scaled[X_train_idx]
X_test_num = X_numeric_scaled[X_test_idx]

# ==========================================
# 4. TRAINING
# ==========================================
print("--- Membangun & Melatih Model ---")

input_text = Input(shape=(X_train_text.shape[1],), name='input_text')
dense_text = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_text) # Naik dari 64 ke 128
dense_text = Dropout(0.5)(dense_text) # Naik dari 0.3 ke 0.5 (Agar tidak overfitting)

# Input Emoji
input_emoji = Input(shape=(X_train_emoji.shape[1],), name='input_emoji')
dense_emoji = Dense(16, activation='relu')(input_emoji)

# Input Numerik
input_num = Input(shape=(X_train_num.shape[1],), name='input_numeric')
dense_num = Dense(8, activation='relu')(input_num)

# Fusi
merged = Concatenate()([dense_text, dense_emoji, dense_num])

x = Dense(32, activation='relu')(merged)
x = Dropout(0.4)(x) # Naik dari 0.2 ke 0.4
output = Dense(3, activation='softmax')(x)

model = Model(inputs=[input_text, input_emoji, input_num], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]

history = model.fit(
    [X_train_text, X_train_emoji, X_train_num], y_train,
    validation_data=([X_test_text, X_test_emoji, X_test_num], y_test),
    epochs=50, batch_size=32, class_weight=class_weights_dict, callbacks=callbacks, verbose=1
)

# ==========================================
# 5. SIMPAN EVALUASI KE GAMBAR
# ==========================================
print("--- Menyimpan Grafik Evaluasi ---")

plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

# Plot 2: Confusion Matrix
y_pred_prob = model.predict([X_test_text, X_test_emoji, X_test_num])
y_pred_class = np.argmax(y_pred_prob, axis=1)
y_true_class = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true_class, y_pred_class)

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix'); plt.xlabel('Prediksi'); plt.ylabel('Asli')

plt.tight_layout()
plt.savefig('output/hasil_evaluasi.png') 
print("Grafik disimpan sebagai 'hasil_evaluasi.png'")

# ==========================================
# 6. SIMPAN HASIL KE EXCEL/CSV
# ==========================================
print("--- Menyimpan Data Evaluasi ke Excel ---")

df_history = pd.DataFrame(history.history)
df_history.index.name = 'epoch'
df_history.reset_index(inplace=True)
df_history['epoch'] += 1 

report_dict = classification_report(y_true_class, y_pred_class, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose() 

df_cm = pd.DataFrame(cm, index=le.classes_, columns=[f"Pred_{c}" for c in le.classes_])
df_cm.index.name = 'Actual'

nama_file_excel = 'output/laporan_hasil_training.xlsx'

try:
    with pd.ExcelWriter(nama_file_excel, engine='openpyxl') as writer:
        df_report.to_excel(writer, sheet_name='Metrics Report')
        df_cm.to_excel(writer, sheet_name='Confusion Matrix')
        df_history.to_excel(writer, sheet_name='Training History', index=False)
    print(f"Sukses! Data hasil tersimpan di '{nama_file_excel}'")
except ImportError:
    print("Library 'openpyxl' belum terinstall. Menyimpan ke CSV terpisah...")
    df_report.to_csv('output/hasil_metrics.csv')
    df_cm.to_csv('output/hasil_confusion_matrix.csv')
    df_history.to_csv('output/hasil_training_history.csv')
    print("Sukses! Data tersimpan di 3 file CSV terpisah.")

# ==========================================
# 7. SIMPAN MODEL & ASET
# ==========================================
print("--- Menyimpan Model & Aset ---")
model.save('output/model_emosi_whatsapp.h5')
joblib.dump(tfidf, 'output/tfidf_vectorizer.pkl')
joblib.dump(emoji_vect, 'output/emoji_vectorizer.pkl')
joblib.dump(scaler, 'output/scaler_numeric.pkl')
joblib.dump(le, 'output/label_encoder.pkl')
print("SELESAI. Semua file berhasil disimpan.")

print("\n[DETAIL] Classification Report:")
print(classification_report(y_true_class, y_pred_class, target_names=le.classes_))