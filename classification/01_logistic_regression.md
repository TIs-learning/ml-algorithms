## 1. Overview

**Logistic Regression** adalah algoritma *supervised learning* yang digunakan khusus untuk menyelesaikan masalah **klasifikasi** (paling umum *binary classification* atau klasifikasi dua kelas).

Meskipun memiliki kata "Regression" pada namanya, Logistic Regression **bukan** digunakan untuk memprediksi nilai kontinu (seperti harga rumah atau suhu), melainkan untuk memprediksi **probabilitas** keberadaan suatu kejadian atau keanggotaan kelas (misalnya $0$ atau $1$, "Ya" atau "Tidak").

```
  [ Fitur Input X ] ---> [ Logistic Regression ] ---> [ Probabilitas P(Y=1) ] ---> [ Class Label (0/1) ]
```

### Masalah yang Diselesaikan
Logistic Regression memetakan hubungan antara satu atau beberapa variabel bebas (*features*) dengan variabel terikat diskrit (*target class*). Algoritma ini menjawab pertanyaan seperti: *"Berapa peluang pelanggan ini akan berhenti berlangganan (churn)?"* atau *"Apakah email ini tergolong spam atau bukan?"*

### Contoh Kasus Dunia Nyata
**Deteksi Transaksi Kartu Kredit Palsu (Fraud Detection)**
* **Input (X):** Nominal transaksi, jarak lokasi transaksi dari rumah, waktu transaksi.
* **Output (Y):** `0` (Transaksi Normal) atau `1` (Transaksi Mencurigakan/Fraud).

---

## 2. Intuition

Bayangkan Anda ingin memprediksi apakah seorang mahasiswa akan **Lulus ($1$)** atau **Gagal ($0$)** dalam suatu ujian berdasarkan **Jumlah Jam Belajar**.

Jika kita menggunakan **Linear Regression** biasa, model akan menarik garis lurus:

```
Hasil Linear Regression:
Nilai Output (y)
   ^
 1.5 |                                      / (Prediksi > 1.0 ?)
 1.0 |-------------------------------------/--- (Lulus)
 0.5 |                                /
 0.0 |-------------------------------/--------- (Gagal)
-0.5 |                      / (Prediksi < 0.0 ?)
     +------------------------------------------> Jam Belajar
```

**Masalah Linear Regression untuk Klasifikasi:**
1. **Prediksi di Luar Batas:** Untuk mahasiswa yang belajar 20 jam, garis lurus mungkin menghasilkan prediksi $y = 1.5$. Padahal, nilai probabilitas tidak boleh melebihi $1.0$ ($100\%$) atau kurang dari $0.0$ ($0\%$).
2. **Sensitif Terhadap Outlier:** Keberadaan data ekstrem akan menggeser posisi garis lurus secara signifikan sehingga garis keputusan (*decision boundary*) menjadi berantakan.

**Solusi Logistic Regression:**
Logistic Regression mengambil garis lurus tersebut, lalu **"membengkokkannya"** menggunakan fungsi khusus dinamakan **Fungsi Sigmoid** sehingga membentuk kurva berbentuk huruf **"S"**:

```
Hasil Logistic Regression (Kurva S):
Probabilitas P(Y=1)
 1.0 |                                 .----''''
     |                               /'
 0.5 |--------------- Threshold ----/----------------
     |                            /'
 0.0 |....----''''---------------'
     +------------------------------------------> Jam Belajar
```

* Jika jam belajar sangat sedikit, nilai probabilitas mendekati $0$.
* Jika jam belajar sangat banyak, nilai probabilitas mendekati $1$.
* Di titik tengah (misalnya jam belajar = 4 jam), probabilitas berada tepat di angka $0.5$ (ambang batas keputusan).

---

## 3. How It Works

Mekanisme Logistic Regression dibagi menjadi dua alur utama: **Training Phase** dan **Prediction Phase**.

### Process Diagram

```
[ Input Features (X) ]
          ↓
[ Linear Combination: z = w·X + b ]
          ↓
[ Sigmoid Function: σ(z) ]
          ↓
[ Predicted Probability (ŷ) ]
          ↓
┌───────────────────────────┴───────────────────────────┐
│                                                       │
▼ (Saat Training)                                       ▼ (Saat Prediction)
[ Compute Loss (Binary Cross-Entropy) ]                 [ Apply Threshold (default 0.5) ]
          ↓                                                       ↓
[ Gradient Descent Update (w & b) ]                     [ Output Class: 0 or 1 ]
```

### A. Training Phase (Proses Pembelajaran)
1. **Linear Score Calculation:** Model menghitung kombinasi linear dari input $X$ menggunakan bobot ($w$) dan bias ($b$).
2. **Sigmoid Transformation:** Nilai kombinasi linear diubah menjadi rentang probabilitas $0$ sampai $1$.
3. **Loss Computation:** Hitung seberapa jauh prediksi model dari label asli ($y$) menggunakan *Binary Cross-Entropy Loss*.
4. **Optimization:** Bobot ($w$) dan bias ($b$) diperbarui secara bertahap menggunakan algoritma *Gradient Descent* hingga nilai loss seminimal mungkin.

### B. Prediction Phase (Proses Inference)
1. Masukkan data $X$ baru ke dalam model yang sudah dilatih ($w$ dan $b$ sudah optimal).
2. Hitung nilai probabilitas $\hat{y} = \sigma(w \cdot X + b)$.
3. Tentukan kelas keluaran berdasarkan ambang batas (*threshold*):
   * Jika $\hat{y} \ge 0.5 \rightarrow \text{Kelas } 1$
   * Jika $\hat{y} < 0.5 \rightarrow \text{Kelas } 0$

---

## 4. Mathematical Foundation

### 1. Persamaan Linear (Score Line)
Sebelum diubah menjadi probabilitas, Logistic Regression terlebih dahulu menghitung persentase skor linear ($z$):

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

* $z$: Skor kombinasi linear (*logit*).
* $x_i$: Fitur masukan ke-$i$.
* $w_i$: Bobot (*weight*) untuk fitur ke-$i$.
* $b$: Bias.

### 2. Fungsi Sigmoid (Activation Function)
Untuk memetakan skor $z$ (rentang $-\infty$ hingga $+\infty$) menjadi nilai probabilitas di rentang $[0, 1]$, digunakan fungsi Sigmoid:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* $\sigma(z)$: Nilai output probabilitas $\hat{y} \in (0, 1)$.
* $e$: Bilangan Euler ($\approx 2.71828$).
* $z$: Skor linear dari tahap sebelumnya.

**Sifat Penting Sigmoid:**
* Jika $z \to +\infty$, maka $e^{-z} \to 0$, sehingga $\sigma(z) \to 1$.
* Jika $z \to -\infty$, maka $e^{-z} \to +\infty$, sehingga $\sigma(z) \to 0$.
* Jika $z = 0$, maka $\sigma(0) = \frac{1}{1 + 1} = 0.5$.

### 3. Probability Formulation
Prediksi probabilitas dinyatakan sebagai:

$$P(Y=1|X) = \hat{y} = \sigma(z)$$

$$P(Y=0|X) = 1 - \hat{y}$$

### 4. Loss Function: Binary Cross-Entropy (Log Loss)
Kita **tidak bisa** menggunakan *Mean Squared Error (MSE)* seperti pada Linear Regression karena fungsi Sigmoid membuat kalkulasi MSE menjadi *non-convex* (banyak *local minima*). 

Sebagai gantinya, digunakan **Binary Cross-Entropy (Log Loss)**:

$$L(y, \hat{y}) = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right)$$

* $y$: Label aktual ($0$ atau $1$).
* $\hat{y}$: Probabilitas hasil prediksi model.

**Intuisi Log Loss:**
* Jika $y = 1$: Loss = $-\log(\hat{y})$. Jika $\hat{y} \to 1$, loss mendekati $0$. Namun jika $\hat{y} \to 0$, loss meledak mendekati $\infty$.
* Jika $y = 0$: Loss = $-\log(1 - \hat{y})$. Jika $\hat{y} \to 0$, loss mendekati $0$. Namun jika $\hat{y} \to 1$, loss meledak mendekati $\infty$.

Untuk seluruh dataset ($m$ data sampel), Cost Function $J(w, b)$ adalah rata-rata dari seluruh loss:

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 5. Gradient Descent Update Rule
Untuk meminimalkan $J(w, b)$, kita memperbarui $w$ dan $b$ dengan menghitung turunan parsial (*gradient*):

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

Pembaruan bobot dan bias dilakukan secara simultan:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

* $\alpha$: *Learning rate* (kecepatan iterasi pembaruan).

---

## 5. Small Manual Example

Misalkan kita memiliki dataset sederhana berisi 3 mahasiswa:

| Mahasiswa | Jam Belajar ($x$) | Status Lulus ($y$) |
|---|---|---|
| A | 1 | 0 |
| B | 3 | 0 |
| C | 5 | 1 |

### Parameter Awal (Inisialisasi):
* $w = 0.5$
* $b = -1.5$
* $\alpha = 0.1$ (*Learning rate*)

### Iterasi 1 (Forward Pass & Update):

**Sampel 1: Mahasiswa A ($x = 1, y = 0$)**
1. Skor Linear: $z = (0.5 \times 1) + (-1.5) = -1.0$
2. Sigmoid: $\hat{y} = \frac{1}{1 + e^{-(-1.0)}} = \frac{1}{1 + 2.718} \approx 0.269$
3. Error: $(\hat{y} - y) = 0.269 - 0 = 0.269$

**Sampel 2: Mahasiswa B ($x = 3, y = 0$)**
1. Skor Linear: $z = (0.5 \times 3) + (-1.5) = 0.0$
2. Sigmoid: $\hat{y} = \frac{1}{1 + e^{0}} = 0.500$
3. Error: $(\hat{y} - y) = 0.500 - 0 = 0.500$

**Sampel 3: Mahasiswa C ($x = 5, y = 1$)**
1. Skor Linear: $z = (0.5 \times 5) + (-1.5) = 1.0$
2. Sigmoid: $\hat{y} = \frac{1}{1 + e^{-(1.0)}} \approx 0.731$
3. Error: $(\hat{y} - y) = 0.731 - 1 = -0.269$

---

### Perhitungan Gradient Kompleks (Batch Mean):

Rata-rata Gradient untuk $w$:
$$\frac{\partial J}{\partial w} = \frac{(0.269 \times 1) + (0.500 \times 3) + (-0.269 \times 5)}{3} = \frac{0.269 + 1.500 - 1.345}{3} = \frac{0.424}{3} \approx 0.1413$$

Rata-rata Gradient untuk $b$:
$$\frac{\partial J}{\partial b} = \frac{0.269 + 0.500 + (-0.269)}{3} = \frac{0.500}{3} \approx 0.1667$$

### Pembaruan Bobot (Update Weights):
$$w_{baru} = 0.5 - (0.1 \times 0.1413) = 0.5 - 0.01413 = 0.48587$$
$$b_{baru} = -1.5 - (0.1 \times 0.1667) = -1.5 - 0.01667 = -1.51667$$

---

## 6. Important Hyperparameters

Berikut hyperparameter utama dalam `scikit-learn` (`LogisticRegression`):

| Hyperparameter | Fungsi | Dampak jika dinaikkan | Dampak jika diturunkan |
|---|---|---|---|
| `C` | Kebalikan dari kekuatan regularisasi ($C = \frac{1}{\lambda}$). | Regularisasi melemah (model lebih kompleks, risiko **overfitting**). | Regularisasi menguat (model lebih sederhana, risiko **underfitting**). |
| `penalty` | Jenis regularisasi (`'l2'`, `'l1'`, `'elasticnet'`, `'none'`). | `'l1'` dapat membuat bobot fitur irrelevant menjadi $0$ (Fitur Selection). | `'l2'` mengecilkan bobot tanpa membuatnya benar-benar nol. |
| `solver` | Algoritma optimasi (`'lbfgs'`, `'liblinear'`, `'saga'`). | `'saga'` cocok untuk dataset besar; `'liblinear'` bagus untuk dataset kecil. | Pilihan solver mempengaruhi waktu training dan dukungan penalty. |
| `max_iter` | Jumlah iterasi maksimum algoritma optimasi. | Memberikan waktu lebih banyak untuk konvergensi (waktu training lebih lama). | Model mungkin berhenti sebelum mencapai titik loss minimum (belum konvergen). |

---

## 7. Python Implementation

Berikut implementasi lengkap menggunakan Python, `numpy`, `pandas`, dan `scikit-learn`.

```python
# 1. Import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 2. Load / Create Dataset Sederhana
data = {
    'Jam_Belajar': [1.5, 2.0, 3.0, 3.5, 4.5, 5.0, 6.0, 6.5, 7.5, 8.0, 8.5, 9.0],
    'Nilai_Tugas': [40, 45, 50, 60, 55, 65, 70, 75, 85, 80, 90, 95],
    'Lulus':       [0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1]
}
df = pd.DataFrame(data)

# 3. Exploratory Check Sederhana
print("--- Data Sample ---")
print(df.head())

X = df[['Jam_Belajar', 'Nilai_Tugas']]
y = df['Lulus']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Preprocessing (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model Initialization
model = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)

# 7. Training
model.fit(X_train_scaled, y_train)

# 8. Prediction
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# 9. Evaluation
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 10. Interpretasi Bobot (Coefficients)
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"Koefisien {feature}: {coef:.4f}")
print(f"Intercept (b): {model.intercept_[0]:.4f}")
```

### Penjelasan Langkah Kode:
1. **StandardScaler:** Fitur `Nilai_Tugas` (rentang 40-95) berukuran jauh lebih besar dibanding `Jam_Belajar` (1.5-9.0). Tanpa *scaling*, gradient descent akan mendominasi fitur bernilai besar.
2. **`predict()` vs `predict_proba()`:** `predict()` langsung menghasilkan label `0` atau `1`. `predict_proba()` menghasilkan probabilitas $P(Y=0)$ dan $P(Y=1)$.

---

## 8. Evaluation

Untuk mengevaluasi performa klasifikasi pada Logistic Regression, digunakan matriks evaluasi berbasis **Confusion Matrix**:

```
                  Aktual Positif (1)    Aktual Negatif (0)
Prediksi Positif       True Positive (TP)    False Positive (FP)
Prediksi Negatif       False Negative (FN)   True Negative (TN)
```

### 1. Accuracy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

* **Kapan digunakan:** Dataset seimbang (*balanced data*).
* **Kesalahan umum:** Menggunakan akurasi pada dataset yang sangat timpang (*imbalanced dataset*).

### 2. Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

* **Kapan digunakan:** Ketika biaya kesalahan **False Positive** tinggi (misal: Email Spam).

### 3. Recall (Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

* **Kapan digunakan:** Ketika biaya kesalahan **False Negative** sangat mahal (misal: Deteksi Kanker).

### 4. F1-Score
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 5. ROC-AUC
Mengukur kemampuan model membedakan antara kelas 0 dan kelas 1 pada berbagai tingkatan *threshold*.
* **Nilai 1.0:** Model sempurna.
* **Nilai 0.5:** Prediksi acak (sama seperti melempar koin).

---

## 9. Strengths and Limitations

| Aspek | Penjelasan |
|---|---|
| **Kelebihan** | Sederhana, cepat dihitung, tidak membutuhkan komputasi berat, keluaran berupa probabilitas yang mudah diinterpretasikan. |
| **Kekurangan** | Hanya dapat mempelajari **linear decision boundary**. Gagal menangkap hubungan kompleks non-linear tanpa *feature engineering*. |
| **Interpretability** | Sangat tinggi. Nilai koefisien $w$ dapat diubah menjadi *Odds Ratio* untuk mengukur dampak tiap fitur. |
| **Training Cost** | Sangat Rendah ($O(m \cdot n)$ di mana $m$ = sampel, $n$ = fitur). |
| **Prediction Cost** | Sangat Rendah ($O(n)$ per sampel baru). |
| **Sensitivity** | Sangat sensitif terhadap *outlier*, *multicollinearity*, dan *unscaled data*. |
| **Scalability** | Sangat baik untuk dataset berukuran besar (*large-scale data*). |

---

## 10. When Should I Use It?

### **Gunakan Logistic Regression ketika:**
* Masalahnya adalah klasifikasi biner.
* Membutuhkan model *baseline* yang cepat dan sederhana sebelum mencoba model kompleks.
* Bisnis/stakeholder membutuhkan transparansi penuh tentang *bagaimana* keputusan diambil (interpretabilitas tinggi).
* Fitur-fitur input memiliki hubungan yang cenderung linier terhadap log-odds target.
* Menginginkan luaran berupa nilai **probabilitas**, bukan sekadar label kaku.

### **Pertimbangkan Algoritma Lain ketika:**
* Hubungan antara fitur dan target sangat kompleks dan bersifat non-linear (Gunakan *Decision Tree*, *Random Forest*, atau *Neural Networks*).
* Fitur saling berinteraksi secara rumit tanpa pola linear.
* Dataset memiliki tingkat *multicollinearity* yang sangat parah.

---

## 11. Comparison

| Aspek | Linear Regression | Logistic Regression | Decision Tree Classifier |
|---|---|---|---|
| **Tujuan utama** | Memprediksi Nilai Kontinu | Memprediksi Probabilitas / Kelas | Memprediksi Kelas / Nilai Kontinu |
| **Bentuk Output** | Kontinu ($-\infty$ s.d. $+\infty$) | Probabilitas ($0$ s.d. $1$) | Diskrit / Kategori |
| **Decision Boundary** | Garis/Hyperplane Lurus | Garis/Hyperplane Lurus | Aksis-Paralel (Kotak-kotak) |
| **Interpretability** | Sangat Tinggi | Sangat Tinggi | Tinggi (jika pohon dangkal) |
| **Feature Scaling** | Sangat Disarankan | **Wajib** | Tidak Perlu |
| **Penanganan Non-Linear** | Lemah | Lemah | Sangat Kuat |

---

## 12. Common Mistakes

1. **Lupa Melakukan Feature Scaling:**
   Logistic Regression menggunakan optimasi berbasis gradient. Jika skala fitur berbeda jauh, proses optimasi akan terhambat.
2. **Mengabaikan Multicollinearity:**
   Jika dua fitur input sangat berkorelasi tinggi, nilai koefisien $w$ menjadi tidak stabil dan tidak bisa diinterpretasikan.
3. **Mengabaikan Imbalanced Data:**
   Jika kelas positif hanya 1% dari data, Logistic Regression akan cenderung memprediksi semua sampel sebagai kelas mayoritas (0). Gunakan `class_weight='balanced'`.
4. **Langsung Menggunakan Threshold Default (0.5) Tanpa Analisis:**
   Ambang batas harus disesuaikan berdasarkan trade-off antara Precision dan Recall sesuai kebutuhan kasus bisnis.

---

## 13. Practical Mini Project

### Problem
Sebuah perusahaan telekomunikasi mengalami tingkat kehilangan pelanggan (*customer churn*) yang tinggi. Manajemen ingin memprediksi pelanggan mana yang berpotensi berhenti berlangganan bulan depan.

### Dataset
Dataset sintetis berisi 1.000 baris data pelanggan dengan atribut:
* `Masa_Berlangganan` (bulan)
* `Tagihan_Bulanan` (Ribuan Rupiah)
* `Jumlah_Keluhan` (kali)
* `Churn` (Target: 0 = Tetap, 1 = Churn)

### Objective
Membangun model Logistic Regression untuk mengidentifikasi pelanggan risiko tinggi *churn* dengan nilai **Recall $\ge 80\%$** untuk kelas positif.

### Steps
1. Load dataset menggunakan pandas.
2. Lakukan pemeriksaan korelasi antar fitur.
3. Pisahkan dataset menjadi Training (80%) dan Testing (20%).
4. Terapkan `StandardScaler` pada fitur numerik.
5. Inisialisasi `LogisticRegression(class_weight='balanced')`.
6. Latih model pada data training.
7. Evaluasi model menggunakan Confusion Matrix, Precision, Recall, dan F1-Score pada data testing.
8. Eksperimen dengan mengubah ambang batas (*threshold*) dari $0.5$ ke $0.3$.

### Expected Learning Outcome
Mahasiswa mampu menerapkan workflow klasifikasi lengkap, melakukan fitur scaling, menangani data imbalanced, serta menyesuaikan decision threshold sesuai kebutuhan bisnis.

---

## 14. Summary

1. **Definisi:** Logistic Regression adalah algoritma supervised learning untuk masalah klasifikasi biner.
2. **Output:** Memprediksi nilai probabilitas keberadaan suatu kelas (rentang $0$ hingga $1$).
3. **Fungsi Utama:** Menggunakan fungsi **Sigmoid** $\sigma(z) = \frac{1}{1 + e^{-z}}$ untuk memetakan kombinasi linear ke skala probabilitas.
4. **Decision Boundary:** Bersifat linear (garis lurus atau hyperplane pembatas).
5. **Loss Function:** Menggunakan **Binary Cross-Entropy (Log Loss)**, bukan MSE.
6. **Optimasi:** Menggunakan algoritma **Gradient Descent** untuk memperbarui bobot ($w$) dan bias ($b$).
7. **Hyperparameter Kunci:** `C` (kekuatan regularisasi) dan `penalty` (`l1`, `l2`).
8. **Kelebihan Utama:** Cepat, efisien, ringan, dan hasil koefisiennya mudah diinterpretasikan.
9. **Keterbatasan Utama:** Tidak mampu mempelajari pola hubungan non-linear yang rumit.
10. **Langkah Sebelum Lanjut:** Pastikan telah memahami *Feature Scaling*, *Confusion Matrix*, dan *Trade-off Precision-Recall*.

---

## 15. Exercises

### Level 1 — Concept
1. Mengapa fungsi *Mean Squared Error (MSE)* tidak cocok digunakan sebagai Loss Function pada Logistic Regression? Jelaskan secara intuitif!
2. Jika nilai skor linear $z = 0$, berapakah nilai probabilitas $\hat{y}$ yang dihasilkan oleh fungsi Sigmoid?
3. Apa perbedaan dampak antara parameter $C = 0.001$ dan $C = 1000$ pada model `LogisticRegression` scikit-learn?

### Level 2 — Implementation
1. Tulis skrip Python dari awal (*from scratch*) tanpa scikit-learn untuk menghitung fungsi Sigmoid dari sebuah NumPy Array: `z = np.array([-2, -1, 0, 1, 2])`.
2. Gunakan `scikit-learn` pada dataset Iris (ambil 2 kelas saja untuk biner). Terapkan `StandardScaler`, latih model `LogisticRegression`, lalu tampilkan probabilitas prediksi untuk 5 sampel pertama data testing menggunakan `predict_proba()`.

### Level 3 — Analysis
1. Diberikan kasus deteksi kanker payudara. Mana yang lebih berbahaya untuk kesehatan pasien: kesalahan **False Positive** atau **False Negative**? Berdasarkan analisis tersebut, apakah Anda akan menaikkan atau menurunkan *decision threshold* dari angka $0.5$? Jelaskan alasan akademis Anda!