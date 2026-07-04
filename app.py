import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧", layout="wide")
st.title("Sistem Klasifikasi Spam pada Pesan Email Berbahasa Indonesia")
st.write("Menggunakan Model IndoBERT berbasis Web")

@st.cache_resource
def load_model():
    model_path = "./indobert-spam-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_email(text):
    cleaned_text = clean_text(text)
    
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = F.softmax(outputs.logits, dim=-1)
    conf_score, prediction = torch.max(probs, dim=1)
    
    label = "Spam" if prediction.item() == 1 else "Ham (Bukan Spam)"
    return label, conf_score.item()

# Tab untuk Single dan Bulk Prediction
tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction (CSV)"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.subheader("Cek Satu Email")
    user_input = st.text_area("Masukkan teks isi email berbahasa Indonesia di sini:", height=150)
    
    if st.button("Deteksi Spam"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks email terlebih dahulu!")
        else:
            with st.spinner('Sedang menganalisis...'):
                label, score = predict_email(user_input)
                
                st.write("### Hasil Klasifikasi:")
                if label == "Spam":
                    st.error(f"**Kategori:** {label}")
                else:
                    st.success(f"**Kategori:** {label}")
                    
                st.info(f"**Confidence Score:** {score:.4f}")

# --- TAB 2: BULK PREDICTION ---
with tab2:
    st.subheader("Cek Banyak Email Sekaligus")
    st.write("Unggah file CSV yang memiliki kolom bernama **Pesan** (berisi teks email).")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'Pesan' not in df.columns:
                st.error("File CSV harus memiliki kolom bernama 'Pesan'!")
            else:
                st.write("Memproses data...")
                # Membuat progress bar
                progress_bar = st.progress(0)
                
                results = []
                scores = []
                total_rows = len(df)
                
                for i, row in df.iterrows():
                    label, score = predict_email(row['Pesan'])
                    results.append(label)
                    scores.append(score)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_rows)
                
                df['Hasil_Prediksi'] = results
                df['Confidence_Score'] = scores
                
                st.success("Prediksi selesai!")
                st.dataframe(df[['Pesan', 'Hasil_Prediksi', 'Confidence_Score']].head(10))
                
                # Tombol untuk mengunduh hasil
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh Hasil Prediksi (CSV)",
                    data=csv,
                    file_name='hasil_prediksi_spam.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
