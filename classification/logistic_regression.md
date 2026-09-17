# Modul Pembelajaran: Logistic Regression

> **Kategori:** Classification
> **Level:** Beginner → Intermediate
> **Prasyarat:** Python dasar, konsep dasar Machine Learning (fitur, label, train-test split)

---

## 1. Overview

**Logistic Regression** adalah algoritma **supervised learning** yang digunakan untuk **classification** (klasifikasi), bukan regresi — meskipun namanya mengandung kata "regression".

- **Tipe pembelajaran:** Supervised
- **Tugas:** Classification (umumnya biner: 0/1, ya/tidak, spam/bukan spam)
- **Output:** Probabilitas suatu data termasuk ke kelas tertentu (nilai antara 0 dan 1), lalu dikonversi menjadi label kelas.

### Masalah yang bisa diselesaikan
- Deteksi email spam vs bukan spam
- Prediksi apakah pasien menderita penyakit tertentu (positif/negatif)
- Prediksi apakah mahasiswa lulus atau tidak berdasarkan jam belajar
- Prediksi apakah pelanggan akan churn (berhenti berlangganan)

### Gambaran sederhana cara kerja
Logistic Regression mencari **garis pemisah (decision boundary)** antara dua kelas. Bedanya dengan Linear Regression: output linear diubah menjadi **probabilitas** melalui fungsi **sigmoid**, sehingga hasilnya selalu berada di rentang [0, 1].

### Contoh kasus dunia nyata
Sebuah bank ingin memprediksi apakah nasabah akan **gagal bayar (default)** berdasarkan pendapatan, umur, dan riwayat kredit. Model menghasilkan probabilitas gagal bayar, misalnya 0.82 → diklasifikasikan sebagai "berisiko tinggi".

---

## 2. Intuition

Bayangkan Anda seorang dosen yang ingin memprediksi apakah mahasiswa **lulus (1)** atau **tidak lulus (0)** berdasarkan **jumlah jam belajar** per minggu.

Data mini:

| Jam Belajar | Lulus? |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |
| 6 | 1 |

Jika kita gambar:

```
Lulus
 1 |         ●   ●   ●
   |
 0 |  ●   ●   ●
   +------------------------> Jam Belajar
      1   2   3   4   5   6
```

Kalau kita paksakan garis lurus (Linear Regression), garis itu bisa memprediksi nilai seperti −0.3 atau 1.7 → tidak masuk akal untuk "probabilitas".

**Solusi Logistic Regression:** gunakan kurva berbentuk **"S"** (sigmoid) yang:
- Selalu berada di antara 0 dan 1
- Meningkat tajam di sekitar titik keputusan (misalnya 3.5 jam)
- Cocok untuk merepresentasikan probabilitas

```
P(lulus)
 1.0 |                _____●●●
     |             _/
 0.5 |............/........... ← threshold
     |         _/
 0.0 |●●●___/
     +------------------------> Jam Belajar
```

Aturan keputusan: jika P(lulus) ≥ 0.5 → prediksi lulus.

---

## 3. How It Works

### Training (belajar dari data)

```
Input: data (X, y)
        ↓
Step 1: Inisialisasi bobot w dan bias b (biasanya 0 atau random kecil)
        ↓
Step 2: Hitung skor linear: z = w·x + b
        ↓
Step 3: Ubah z menjadi probabilitas: p = sigmoid(z)
        ↓
Step 4: Hitung loss (log loss) antara p dan y sebenarnya
        ↓
Step 5: Update w dan b menggunakan Gradient Descent
        ↓
Step 6: Ulangi Step 2–5 hingga loss cukup kecil / iterasi habis
        ↓
Output: model dengan w dan b optimal
```

### Prediction (memprediksi data baru)

```
Input: data baru x_new
        ↓
Step 1: Hitung z = w·x_new + b
        ↓
Step 2: Hitung p = sigmoid(z)
        ↓
Step 3: Jika p ≥ threshold (biasanya 0.5) → kelas 1, else kelas 0
        ↓
Output: label kelas (dan/atau probabilitas)
```

---

## 4. Mathematical Foundation

### 4.1 Konsep Dasar: Fungsi Linear

Logistic Regression dimulai dari fungsi linear:

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

- $x_i$ = fitur ke-i
- $w_i$ = bobot fitur ke-i (menunjukkan pengaruh fitur)
- $b$ = bias (intercept)
- $z$ = skor linear (bisa bernilai apa saja, dari −∞ sampai +∞)

**Masalah:** $z$ bukan probabilitas. Kita butuh cara mengubahnya menjadi angka antara 0 dan 1.

### 4.2 Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- $e$ = bilangan Euler (≈ 2.718)
- $\sigma(z)$ selalu bernilai antara 0 dan 1

**Intuisi:**
- Jika $z$ sangat besar (misal 10) → $\sigma(z) \approx 1$
- Jika $z$ sangat kecil (misal −10) → $\sigma(z) \approx 0$
- Jika $z = 0$ → $\sigma(z) = 0.5$

Inilah kurva "S" yang tadi dibahas.

Model akhirnya:

$$P(y=1 \mid x) = \sigma(w \cdot x + b)$$

### 4.3 Loss Function: Log Loss (Binary Cross-Entropy)

Kita perlu ukuran seberapa "salah" prediksi model. Untuk klasifikasi, digunakan **log loss**:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

- $N$ = jumlah data
- $y_i$ = label sebenarnya (0 atau 1)
- $p_i$ = probabilitas prediksi

**Intuisi:**
- Jika $y = 1$ dan $p = 0.99$ → loss kecil (bagus).
- Jika $y = 1$ dan $p = 0.01$ → loss besar (model sangat salah).
- Log loss memberi **penalti besar** pada prediksi yang salah dengan keyakinan tinggi.

**Kenapa tidak pakai MSE?** Karena dengan sigmoid, MSE menghasilkan permukaan loss yang tidak convex → sulit dioptimasi. Log loss menghasilkan **convex loss** → gradient descent dijamin konvergen ke minimum global.

### 4.4 Gradient Descent

Untuk menemukan $w$ dan $b$ terbaik, kita minimumkan $L$ dengan gradient descent:

$$w := w - \eta \frac{\partial L}{\partial w}, \quad b := b - \eta \frac{\partial L}{\partial b}$$

- $\eta$ = learning rate (seberapa besar langkah update)

Turunan log loss (fakta yang perlu diketahui, tanpa derivasi):

$$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i) x_{ij}$$

Bentuk ini sederhana: **selisih prediksi dan label**, dikalikan fitur.

### 4.5 Regularization (opsional tapi penting)

Untuk mencegah overfitting, tambahkan penalti pada bobot:

- **L2 (Ridge):** $L + \lambda \sum w_j^2$
- **L1 (Lasso):** $L + \lambda \sum |w_j|$

$\lambda$ mengontrol kekuatan regularisasi.

---

## 5. Small Manual Example

Dataset: prediksi lulus (1) / tidak (0) berdasarkan jam belajar.

| $x$ (jam) | $y$ |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 4 | 1 |
| 5 | 1 |

Misalkan setelah training kita mendapat: **$w = 1.5$, $b = -4.5$**.

**Hitung untuk $x = 2$:**
- $z = 1.5 \times 2 - 4.5 = -1.5$
- $p = \frac{1}{1 + e^{1.5}} = \frac{1}{1 + 4.48} \approx 0.182$
- Karena $p < 0.5$ → prediksi **0** ✓ (sesuai label)

**Hitung untuk $x = 5$:**
- $z = 1.5 \times 5 - 4.5 = 3.0$
- $p = \frac{1}{1 + e^{-3.0}} = \frac{1}{1 + 0.0498} \approx 0.953$
- Karena $p \geq 0.5$ → prediksi **1** ✓

**Titik keputusan (decision boundary):** $z = 0 \Rightarrow x = 3$.
Artinya siapa pun yang belajar > 3 jam diprediksi lulus.

---

## 6. Important Hyperparameters

| Hyperparameter | Fungsi | Dampak jika dinaikkan | Dampak jika diturunkan |
|---|---|---|---|
| `C` (invers dari $\lambda$) | Mengontrol kekuatan regularisasi | Regularisasi lemah → risiko overfitting | Regularisasi kuat → risiko underfitting |
| `penalty` | Jenis regularisasi (`l1`, `l2`, `elasticnet`, `none`) | L1 → sparse (banyak bobot = 0), L2 → shrinkage halus | — |
| `solver` | Algoritma optimasi (`lbfgs`, `liblinear`, `saga`) | Berpengaruh pada kecepatan & dukungan penalty | — |
| `max_iter` | Jumlah maksimum iterasi optimasi | Lebih lama training, tapi konvergen lebih baik | Risiko belum konvergen |
| `class_weight` | Bobot kelas (untuk data tidak seimbang) | `balanced` membantu kelas minoritas | Bias ke kelas mayoritas |

**Catatan:**
- `C` kecil → model sederhana → cenderung **underfitting**.
- `C` besar → model bebas → cenderung **overfitting**.
- Waktu training relatif cepat karena optimasinya convex.

---

## 7. Python Implementation

Kita gunakan dataset **Breast Cancer** dari scikit-learn (klasifikasi biner: tumor jinak vs ganas).

```python
# 1. Import library
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

# 2. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

# 3. Exploratory check sederhana
print(X.shape)          # (569, 30)
print(y.value_counts()) # cek balance kelas

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessing: standardisasi (WAJIB untuk logistic regression
#    karena optimasi lebih cepat dan regularisasi lebih adil)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit HANYA di train
X_test_scaled  = scaler.transform(X_test)        # test hanya di-transform

# 6. Model initialization
model = LogisticRegression(C=1.0, penalty='l2',
                           solver='lbfgs', max_iter=1000)

# 7. Training
model.fit(X_train_scaled, y_train)

# 8. Prediction
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 9. Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))
print(confusion_matrix(y_test, y_pred))

# 10. Interpretasi: koefisien mana yang paling berpengaruh
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
print(coef.head(5))   # 5 fitur paling menurunkan probabilitas kelas 1
print(coef.tail(5))   # 5 fitur paling menaikkan probabilitas kelas 1
```

**Penjelasan penting:**
- `stratify=y` menjaga proporsi kelas di train & test.
- `fit_transform` hanya dilakukan di data train untuk **mencegah data leakage**.
- Koefisien $w$ bisa langsung diinterpretasi (setelah standardisasi): semakin besar |w|, semakin besar pengaruh fitur.

---

## 8. Evaluation

Karena Logistic Regression menghasilkan **probabilitas**, kita bisa menggunakan berbagai metric.

| Metric | Apa yang diukur | Kapan digunakan |
|---|---|---|
| **Accuracy** | (TP+TN)/total | Data seimbang |
| **Precision** | TP/(TP+FP) | Ketika False Positive mahal (misal: spam) |
| **Recall** | TP/(TP+FN) | Ketika False Negative mahal (misal: deteksi kanker) |
| **F1-score** | Harmonic mean precision & recall | Data tidak seimbang |
| **ROC-AUC** | Kemampuan model membedakan kelas di berbagai threshold | Butuh evaluasi tanpa bergantung threshold |

### Kesalahan interpretasi yang umum
- **Terlalu mengandalkan accuracy** pada data tidak seimbang (misal 95% kelas 0 → prediksi selalu 0 sudah accuracy 95%, tapi model tidak berguna).
- **Menyamakan probabilitas dengan keyakinan mutlak**: $p = 0.9$ berarti "9 dari 10 kasus serupa akan berlabel 1", bukan "pasti 1".
- **Menggunakan threshold 0.5 tanpa berpikir**: threshold bisa disesuaikan dengan konteks bisnis (misal 0.3 untuk deteksi penyakit agar lebih sensitif).

---

## 9. Strengths and Limitations

| Aspek | Penjelasan |
|---|---|
| **Kelebihan** | Sederhana, cepat, menghasilkan probabilitas, mudah diinterpretasi |
| **Kekurangan** | Hanya cocok untuk hubungan (kurang-lebih) linear antara fitur dan log-odds; tidak menangkap interaksi non-linear tanpa feature engineering |
| **Interpretability** | Tinggi — koefisien menunjukkan arah & besar pengaruh fitur |
| **Training cost** | Rendah (optimasi convex) |
| **Prediction cost** | Sangat rendah (hanya perkalian vektor) |
| **Sensitivity** | Sensitif terhadap skala fitur (perlu standardisasi), multikolinearitas, dan outlier |
| **Scalability** | Baik untuk dataset besar; dengan solver `saga` bisa menangani jutaan sampel |

---

## 10. When Should I Use It?

**Gunakan Logistic Regression ketika:**
- Masalahnya klasifikasi biner (atau multi-kelas dengan strategi one-vs-rest / softmax).
- Anda butuh **probabilitas**, bukan hanya label.
- Anda butuh model yang **mudah dijelaskan** ke stakeholder non-teknis.
- Dataset relatif linear separable atau bisa dibuat linear dengan feature engineering.
- Anda butuh **baseline model** sebelum mencoba yang lebih kompleks.

**Pertimbangkan algoritma lain ketika:**
- Hubungan antara fitur dan target sangat **non-linear** dan sulit dilinearkan → coba Random Forest, Gradient Boosting, atau Neural Network.
- Fitur banyak berinteraksi kompleks → tree-based models lebih cocok.
- Data berupa gambar, teks mentah, atau audio → gunakan deep learning atau representasi khusus.

---

## 11. Comparison

| Aspek | K-Nearest Neighbors | **Logistic Regression** | Decision Tree |
|---|---|---|---|
| Cara kerja | Cari K tetangga terdekat, voting | Cari garis pemisah linear via probabilitas | Split data secara rekursif berdasar fitur |
| Interpretability | Rendah | **Tinggi** (koefisien) | Sedang–Tinggi (aturan if-else) |
| Training | Sangat cepat (lazy) | Cepat | Sedang |
| Prediction | Lambat (hitung jarak ke semua data) | **Sangat cepat** | Cepat |
| Overfitting | Sensitif jika K kecil | Rendah (dengan regularisasi) | Tinggi jika tidak di-pruning |
| Data yang cocok | Data kecil dengan struktur lokal | Data linear separable dengan fitur numerik terstandarisasi | Data campuran, non-linear, ada interaksi |

---

## 12. Common Mistakes

1. **Tidak melakukan standardisasi fitur.** Fitur berskala besar akan mendominasi optimasi & regularisasi.
2. **Data leakage:** melakukan `fit` scaler / encoder di seluruh dataset sebelum split.
3. **Salah pilih metric:** menggunakan accuracy pada data tidak seimbang.
4. **Menggunakan threshold 0.5 secara buta** tanpa mempertimbangkan biaya kesalahan.
5. **Menganggap koefisien besar = fitur "penting"** padahal fitur belum distandarisasi (skala berbeda).
6. **Overfitting** karena `C` terlalu besar dan tidak melakukan cross-validation.
7. **Mengharapkan performa tinggi pada data non-linear** tanpa feature engineering.
8. **Multikolinearitas:** dua fitur sangat berkorelasi membuat koefisien tidak stabil dan sulit diinterpretasi.

---

## 13. Practical Mini Project

### Problem
Sebuah perusahaan telekomunikasi ingin memprediksi pelanggan yang berpotensi **churn** (berhenti berlangganan) agar dapat ditawari promo retensi.

### Dataset
[Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — sekitar 7.000 baris, fitur numerik & kategorikal (tenure, monthly charges, contract type, dll.), target: `Churn` (Yes/No).

### Objective
Membangun model Logistic Regression yang bisa memprediksi churn dengan **recall tinggi** (agar sedikit pelanggan berisiko yang terlewat), sambil menjaga precision agar promo tidak salah sasaran.

### Steps
1. Load dataset, cek missing values dan tipe data.
2. Encode variabel kategorikal (`pd.get_dummies` atau `OneHotEncoder`).
3. Split data (stratified) menjadi train dan test.
4. Standardisasi fitur numerik.
5. Latih Logistic Regression dengan `class_weight='balanced'`.
6. Evaluasi: confusion matrix, precision, recall, F1, ROC-AUC.
7. Coba beberapa nilai threshold (0.3, 0.4, 0.5) dan analisis trade-off precision-recall.
8. Interpretasi 5 koefisien terbesar dan terkecil → fitur apa yang mendorong churn?

### Expected Learning Outcome
- Memahami alur end-to-end klasifikasi.
- Merasakan efek `class_weight` dan threshold tuning pada data tidak seimbang.
- Mampu menerjemahkan koefisien model menjadi **insight bisnis**.

---

## 14. Summary

1. **Logistic Regression** adalah algoritma **classification** meski namanya "regression".
2. Menyelesaikan masalah klasifikasi (terutama biner) dengan menghasilkan **probabilitas**.
3. Cara kerja: hitung skor linear $z = w \cdot x + b$, lalu ubah menjadi probabilitas via **sigmoid**.
4. Konsep matematika inti: **sigmoid function** + **log loss** + **gradient descent**.
5. Hyperparameter terpenting: `C` (regularisasi), `penalty`, `class_weight`.
6. Kelebihan: sederhana, cepat, interpretable, menghasilkan probabilitas.
7. Kekurangan: hanya cocok untuk hubungan (kurang-lebih) linear; sensitif terhadap skala & multikolinearitas.
8. Gunakan ketika butuh **baseline** yang mudah dijelaskan dan probabilitas.
9. Jangan gunakan jika masalah sangat non-linear tanpa feature engineering.
10. Sebelum lanjut ke algoritma berikutnya, pahami: **sigmoid, log loss, gradient descent, regularization, dan interpretasi koefisien**.

---

## 15. Exercises

### Level 1 — Concept
1. Mengapa Logistic Regression disebut algoritma klasifikasi, bukan regresi?
2. Apa arti nilai output $p = 0.7$ dari model Logistic Regression?
3. Jelaskan mengapa kita menggunakan **log loss**, bukan **MSE**.
4. Apa yang terjadi pada model jika nilai `C` sangat kecil?
5. Mengapa fitur perlu distandarisasi sebelum dilatih?

### Level 2 — Implementation
1. Latih Logistic Regression pada dataset `load_iris` (ambil hanya 2 kelas pertama agar biner). Bandingkan akurasi dengan dan tanpa `StandardScaler`.
2. Gunakan `load_breast_cancer`. Coba `C = [0.01, 0.1, 1, 10, 100]`. Catat accuracy train vs test untuk tiap nilai. Nilai mana yang overfitting?
3. Implementasi manual sigmoid + prediksi, tanpa scikit-learn:
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   ```
   Ambil `model.coef_` dan `model.intercept_` dari model terlatih, hitung probabilitas sendiri, bandingkan dengan `predict_proba`.
4. Ganti threshold default 0.5 menjadi 0.3 secara manual dari `predict_proba`. Bandingkan precision dan recall.
5. Tulis Logistic Regression dari nol dengan NumPy: gradient descent, `max_iter=1000`, `learning_rate=0.1`. Bandingkan bobot akhir dengan scikit-learn.

### Level 3 — Analysis
1. Buat dataset tidak seimbang (90% kelas 0). Latih dua model: default dan `class_weight='balanced'`. Analisis perubahan confusion matrix. Metric mana yang menyesatkan di sini?
2. Bandingkan `penalty='l1'` vs `penalty='l2'` (solver `liblinear` atau `saga`) pada dataset breast cancer. Berapa koefisien yang menjadi 0 pada L1? Apa artinya?
3. Buat data non-linear (misal `sklearn.datasets.make_circles`). Latih Logistic Regression. Mengapa gagal? Tambahkan fitur $x_1^2 + x_2^2$ lalu latih ulang. Jelaskan hasilnya.
4. Ambil dataset dengan dua fitur yang sangat berkorelasi (korelasi > 0.95). Latih model, lihat koefisiennya. Hapus satu fitur, latih ulang. Jelaskan perubahan koefisien dan kaitannya dengan multikolinearitas.
5. Plot kurva ROC dan Precision-Recall untuk model Anda. Pada kasus deteksi penyakit, kurva mana yang lebih informatif? Kenapa?

---

### Optional / Further Learning
- **Multinomial Logistic Regression (Softmax)** untuk lebih dari 2 kelas
- **Interpretasi odds ratio**: $e^{w_j}$ = perubahan odds per unit kenaikan fitur
- **Elastic Net** (kombinasi L1 + L2)
- **Calibration curve** untuk memeriksa apakah probabilitas model realistis
- **Newton-Raphson / IRLS** sebagai alternatif gradient descent
