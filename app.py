import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import re
import emoji

# --- 1. CONFIG & LOAD MODEL ---
st.set_page_config(page_title="Deteksi Emosi WA", page_icon="💬")

@st.cache_resource # Agar model tidak di-load berulang kali setiap klik
def load_assets():
    model = tf.keras.models.load_model('data/model_emosi_whatsapp.h5')
    tfidf = joblib.load('data/tfidf_vectorizer.pkl')
    emoji_vect = joblib.load('data/emoji_vectorizer.pkl')
    scaler = joblib.load('data/scaler_numeric.pkl')
    le = joblib.load('data/label_encoder.pkl')
    return model, tfidf, emoji_vect, scaler, le

# Load data
try:
    model, tfidf, emoji_vect, scaler, le = load_assets()
    st.success("Sistem Siap! Model berhasil dimuat.")
except Exception as e:
    st.error(f"Error memuat model. Pastikan file .h5 dan .pkl ada di folder yang sama. Error: {e}")
    st.stop()

# --- 2. FUNGSI PREPROCESSING (Sama dengan Training) ---
def preprocess_input(message, hour_val):
    # A. Ekstraksi Emoji
    # Mengambil karakter emoji saja dari pesan
    extracted_emojis = ''.join([c for c in message if c in emoji.EMOJI_DATA])
    
    # B. Hitung Fitur Numerik
    caps_len = sum(1 for c in message if c.isupper())
    total_len = len(message)
    caps_ratio = caps_len / total_len if total_len > 0 else 0
    emoji_count = len(extracted_emojis)
    
    # C. Bersihkan Teks
    clean_txt = re.sub(r'[^\w\s]', '', message.lower())
    
    # D. Transformasi ke Vector
    vec_text = tfidf.transform([clean_txt]).toarray()
    vec_emoji = emoji_vect.transform([extracted_emojis]).toarray()
    vec_num = scaler.transform([[caps_ratio, hour_val, emoji_count]])
    
    return [vec_text, vec_emoji, vec_num]

# --- 3. TAMPILAN UI (Streamlit) ---
st.title("💬 Analisis Emosi Percakapan WhatsApp")
st.write("Menggunakan Jaringan Syaraf Tiruan (MLP) dengan Fusi Fitur.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_message = st.text_area("Masukkan Pesan WhatsApp:", height=100, placeholder="Contoh: Selamat ulang tahun! 🎉")
    
    with col2:
        user_time = st.time_input("Waktu Pengiriman:", value=None)
        
    submit_btn = st.form_submit_button("Prediksi Emosi")

# Logic Prediksi
if submit_btn and user_message:
    # Ambil Jam dari input waktu
    if user_time:
        hour_val = user_time.hour
    else:
        hour_val = 12 # Default
        
    # Proses Data
    input_data = preprocess_input(user_message, hour_val)
    
    # Prediksi
    pred_prob = model.predict(input_data, verbose=0)
    pred_class_idx = np.argmax(pred_prob)
    pred_label = le.inverse_transform([pred_class_idx])[0]
    confidence = pred_prob[0][pred_class_idx] * 100
    
    # Tampilkan Hasil
    st.divider()
    st.subheader("Hasil Analisis")
    
    # Warna hasil
    color_map = {'positif': 'green', 'negatif': 'red', 'netral': 'gray'}
    result_color = color_map.get(pred_label, 'blue')
    
    st.markdown(f"Emosi Terdeteksi: <h2 style='color:{result_color};'>{pred_label.upper()}</h2>", unsafe_allow_html=True)
    st.write(f"Confidence Level: **{confidence:.2f}%**")
    
    # Tampilkan Bar Chart Probabilitas
    st.write("Detail Probabilitas:")
    prob_dict = {label: round(prob, 4) for label, prob in zip(le.classes_, pred_prob[0])}
    st.bar_chart(prob_dict)

elif submit_btn and not user_message:
    st.warning("Mohon masukkan pesan teks terlebih dahulu.")