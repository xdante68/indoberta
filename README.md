# Indonesian Email Spam Classification with IndoBERT

Sistem klasifikasi email spam berbahasa Indonesia berbasis IndoBERT dan Streamlit. Proyek ini melakukan fine-tuning model IndoBERT untuk mengklasifikasikan isi email ke dalam dua kelas, yaitu `Spam` dan `Ham (Bukan Spam)`, lalu menyediakan antarmuka web untuk prediksi satu email maupun prediksi massal melalui file CSV.

## Project Summary

Email masih menjadi media komunikasi utama, tetapi juga menjadi salah satu kanal umum untuk spam, promosi tidak diinginkan, phishing, dan pesan berbahaya. Proyek ini membangun sistem deteksi spam otomatis untuk teks email berbahasa Indonesia menggunakan pendekatan Natural Language Processing berbasis Transformer.

Model utama yang digunakan adalah `indobenchmark/indobert-base-p1`, yaitu varian BERT yang telah dilatih pada korpus bahasa Indonesia. Model kemudian di-fine-tune untuk tugas klasifikasi biner email `spam` dan `ham`.

## Key Features

- Klasifikasi email berbahasa Indonesia menjadi `Spam` atau `Ham`.
- Prediksi satu email melalui input teks.
- Prediksi banyak email melalui upload file CSV.
- Confidence score untuk setiap hasil prediksi.
- Model IndoBERT lokal yang dimuat dari folder `indobert-spam-model`.
- Aplikasi web sederhana menggunakan Streamlit.

## Tech Stack

- Python
- Pandas
- PyTorch
- Hugging Face Transformers
- IndoBERT
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Dataset

Dataset yang digunakan adalah dataset email spam berbahasa Indonesia dengan dua kolom utama:

| Kolom | Deskripsi |
| --- | --- |
| `Kategori` | Label email, yaitu `spam` atau `ham` |
| `Pesan` | Isi teks email |

Ringkasan data:

- Jumlah data awal: 2.636 baris
- Label awal: 1.368 spam dan 1.268 ham
- Missing value: tidak ada
- Duplikasi pesan/baris: 16
- Setelah drop duplikat: 2.620 baris
- Label setelah drop duplikat: 1.368 spam dan 1.252 ham

## Methodology

Alur pengembangan model mengikuti referensi proyek/paper:

1. Load dataset email spam Indonesia.
2. Cek kualitas data, missing value, distribusi label, dan duplikasi.
3. Drop data duplikat untuk mengurangi risiko data leakage.
4. Preprocessing teks:
   - lowercase
   - hapus URL
   - hapus angka, tanda baca, simbol, dan karakter non-alfabet
   - hapus spasi berlebih
5. Encoding label:
   - `ham = 0`
   - `spam = 1`
6. Split dataset:
   - 80% training
   - 12% validation
   - 8% testing
7. Tokenisasi menggunakan IndoBERT tokenizer.
8. Fine-tuning IndoBERT menggunakan Hugging Face `Trainer`.
9. Evaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.
10. Save model untuk digunakan pada aplikasi Streamlit.

## Model Configuration

Konfigurasi utama training:

| Parameter | Nilai |
| --- | --- |
| Base model | `indobenchmark/indobert-base-p1` |
| Task | Binary text classification |
| Number of labels | 2 |
| Learning rate | `3e-5` |
| Epoch | 10 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Best model metric | F1-score |

## Evaluation Result

Hasil evaluasi pada data testing:

| Metric | Score |
| --- | ---: |
| Accuracy | 98.57% |
| Precision | 98.20% |
| Recall | 99.09% |
| F1-score | 98.64% |
| Eval loss | 0.1311 |

Confusion matrix:

| Actual / Predicted | Ham | Spam |
| --- | ---: | ---: |
| Ham | 98 | 2 |
| Spam | 1 | 109 |

Interpretasi:

- Model berhasil mendeteksi 109 dari 110 email spam pada data testing.
- Hanya 1 email spam yang salah diprediksi sebagai ham.
- Hanya 2 email ham yang salah diprediksi sebagai spam.
- Hasil menunjukkan model memiliki performa yang sangat baik untuk deteksi spam email berbahasa Indonesia.

## Overfitting Analysis

Learning curve menunjukkan indikasi overfitting ringan: training loss turun sangat rendah, sementara validation loss tidak selalu turun secara konsisten. Namun, dampaknya tidak terlihat signifikan pada data testing karena model tetap menghasilkan F1-score yang tinggi.

Untuk pengembangan lanjutan, stabilitas model dapat ditingkatkan dengan:

- early stopping
- weight decay
- tuning learning rate
- tuning batch size
- penambahan data atau augmentasi teks

## Repository Structure

```text
.
├── app.py
├── training.ipynb
├── email_spam_indo.csv
├── deskripsi_proyek.pdf
├── referensi_paper.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

Folder model tidak disertakan di repository karena ukuran file model besar.

Struktur model yang dibutuhkan:

```text
indobert-spam-model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Letakkan folder `indobert-spam-model` sejajar dengan `app.py` sebelum menjalankan aplikasi.

## Installation

Clone repository:

```bash
git clone https://github.com/username/indobert-email-spam-classification.git
cd indobert-email-spam-classification
```

Buat virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Untuk Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

Karena file model berukuran besar, folder `indobert-spam-model` tidak diupload langsung ke GitHub. Download atau salin folder model hasil training, lalu letakkan pada root project:

```text
.
├── app.py
└── indobert-spam-model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

Model disimpan di Google Drive.

```text
Model download link: <https://drive.google.com/drive/folders/1hLbGun4_YopiNfVNkn86t7Sk_i8xz9b5?usp=drive_link>
```

## Run the Application

Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

Setelah server berjalan, buka URL lokal yang diberikan Streamlit, biasanya:

```text
http://localhost:8501
```

## CSV Format for Bulk Prediction

Untuk prediksi massal, upload file CSV dengan kolom bernama `Pesan`.

Contoh:

```csv
Pesan
"Selamat anda mendapatkan hadiah gratis, klik link berikut sekarang"
"Halo, berikut jadwal rapat untuk hari Senin"
```

Output aplikasi akan menambahkan:

- `Hasil_Prediksi`
- `Confidence_Score`

## How It Works

1. User memasukkan teks email atau upload CSV.
2. Teks dibersihkan menggunakan fungsi preprocessing yang sama dengan training.
3. Teks ditokenisasi menggunakan IndoBERT tokenizer.
4. Model IndoBERT melakukan prediksi kelas.
5. Probabilitas dihitung menggunakan softmax.
6. Aplikasi menampilkan label prediksi dan confidence score.

## Portfolio Highlight

Proyek ini menunjukkan kemampuan dalam:

- membangun pipeline NLP end-to-end
- melakukan fine-tuning Transformer model untuk bahasa Indonesia
- melakukan data cleaning dan validasi dataset
- mengevaluasi model klasifikasi menggunakan metrik standar
- menganalisis overfitting melalui learning curve
- mengintegrasikan model machine learning ke aplikasi web interaktif
- membangun fitur single prediction dan bulk CSV prediction

## Reference

Proyek ini mengacu pada pendekatan dalam paper:

Supono, Riza Adriati, and Muhammad Irgi Imani. "Implementasi Machine Learning untuk Klasifikasi Email Spam Menggunakan Indobert, Hugging Face Transfomers dan Streamlit." Jurnal Sosial dan Teknologi (SOSTECH), vol. 6, no. 1, 1 Januari 2026.

## Notes

- Model besar tidak disimpan langsung di GitHub.
- File `indobert-spam-model/` diabaikan melalui `.gitignore`.
- Untuk training, jalankan `training.ipynb` dari awal.
- Untuk deployment, pastikan resource server cukup untuk memuat model IndoBERT.

