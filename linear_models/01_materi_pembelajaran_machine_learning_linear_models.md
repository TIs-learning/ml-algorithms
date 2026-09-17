# ML Algorithms

Machine Learning (ML) adalah cabang dari kecerdasan buatan (Artificial Intelligence) yang berfokus pada pengembangan sistem yang mampu belajar dari data untuk membuat prediksi, keputusan, atau menemukan pola tanpa diatur secara eksplisit menggunakan pemrograman berbasis aturan (rule-based programming).

Secara umum, algoritma Machine Learning dibagi menjadi beberapa paradigma utama:
1. **Supervised Learning (Pembelajaran Terawasi):** Model dilatih menggunakan dataset yang memiliki input (fitur) dan output yang benar (label/target).
2. **Unsupervised Learning (Pembelajaran Tanpa Pengawasan):** Model mencari struktur atau pola tersembunyi dari data tanpa label.
3. **Semi-Supervised Learning & Reinforcement Learning:** Kombinasi data berlabel dan tidak berlabel, atau pembelajaran berbasis agen yang berinteraksi dengan lingkungan melalui *reward* dan *penalty*.

Dalam paradigma *Supervised Learning*, tugas utama dibagi menjadi dua jenis masalah:
- **Klasifikasi (Classification):** Memprediksi nilai diskrit atau kategori (misalnya: email spam vs non-spam, diagnosis penyakit positif vs negatif).
- **Regresi (Regression):** Memprediksi nilai kontinu atau numerik (misalnya: harga rumah, suhu udara, estimasi pendapatan).

Modul ini berfokus pada fondasi paling dasar namun sangat krusial dalam algoritma regresi, yaitu kelompok **Linear Models**.

---

## Linear Models

**Linear Models** (Model Linear) adalah keluarga algoritma pembelajar berbasis statistik dan Machine Learning yang mengasumsikan bahwa hubungan antara fitur masukan (input variables/features) dan variabel target (output/target variable) dapat dimodelkan melalui kombinasi linear (penjumlahan berdampak pembobotan).

Secara intuitif, jika kita memiliki variabel masukan $x$, model linear mencoba mencari bobot ($\beta$) yang tepat sehingga perkalian antara bobot dan fitur masukan dapat mendekati variabel target $y$.

### Karakteristik Utama Linear Models
- **Sederhana dan Interpretabel:** Setiap koefisien (bobot) dalam model linear memiliki makna fisik/statistik yang jelas, yaitu seberapa besar perubahan target $y$ jika suatu fitur bertambah 1 satuan.
- **Efisiensi Komputasi:** Proses pelatihan dan prediksi relatif sangat cepat dibanding model kompleks seperti Neural Networks atau Random Forest.
- **Dasar Evaluasi Baseline:** Linear Models hampir selalu dijadikan *baseline model* (model pembanding awal) sebelum mencoba algoritma yang lebih kompleks.

---

### Linear Regression

#### 1. Konsep dan Definisi
Linear Regression (sering disebut *Simple Linear Regression* atau Regresi Linear Sederhana) adalah algoritma supervised learning yang digunakan untuk memprediksi nilai target kontinu $y$ berdasarkan **satu** variabel prediktor/fitur $x$.

#### 2. Intuisi
Bayangkan Anda ingin memprediksi harga sewa rumah berdasarkan luas bangunannya. Secara intuitif, semakin luas rumah, semakin mahal harga sewanya. Jika Anda menggambar titik-titik data luas rumah (sumbu-X) dan harga sewa (sumbu-Y) pada grafik kartesian, Regresi Linear mencoba menarik **satu garis lurus terbaik** (line of best fit) yang paling dekat dengan titik-titik data tersebut.

#### 3. Bagaimana Algoritma Bekerja
Regresi Linear mencari nilai perpotongan garis pada sumbu-Y (*intercept*) dan kemiringan garis (*slope*) yang meminimalkan total jarak (kesalahan) antara titik data asli dengan garis prediksi.

#### 4. Dasar Matematika
Persamaan umum Simple Linear Regression adalah:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Di mana:
- $y$: Variabel target (dependent variable) yang ingin diprediksi.
- $x$: Variabel prediktor (independent variable/feature).
- $\beta_0$: Intercept (titik potong sumbu-Y saat $x = 0$).
- $\beta_1$: Slope (koefisien regresi/kemiringan garis), menggambarkan perubahan nilai $y$ untuk setiap kenaikan 1 unit $x$.
- $\epsilon$: Residual/error (galat), yaitu selisih antara nilai aktual dan nilai prediksi yang tidak dapat dijelaskan oleh model.

##### Fungsi Kerugian (Loss Function)
Untuk menemukan garis terbaik, algoritma harus mengukur seberapa buruk kinerja garis saat ini. Pengukur ini disebut **Residual Sum of Squares (RSS)** atau **Sum of Squared Errors (SSE)**:

$$RSS(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

Di mana:
- $n$: Jumlah sampel data.
- $y_i$: Nilai aktual ke-$i$.
- $\hat{y}_i$: Nilai prediksi ke-$i$.

##### Metode Optimasi: Ordinary Least Squares (OLS)
OLS menentukan nilai $\hat{\beta}_0$ dan $\hat{\beta}_1$ yang meminimalkan nilai $RSS$ menggunakan turunan kalkulus (menyetel turunan parsial ke nol). Solusi bentuk tertutup (*closed-form solution*) adalah:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

Di mana $\bar{x}$ dan $\bar{y}$ masing-masing adalah rata-rata dari variabel $x$ dan $y$.

#### 5. Proses Training dan Prediksi
- **Training:** Menghitung statistik deskriptif dari data latih ($\bar{x}, \bar{y}$, kovarians, dan varians) untuk mendapatkan nilai parameter $\hat{\beta}_0$ dan $\hat{\beta}_1$.
- **Prediksi:** Memasukkan nilai $x$ baru ke dalam persamaan $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$.

#### 6. Asumsi Utama (Standar OLS)
1. **Linearitas:** Hubungan antara $x$ dan $y$ adalah linear.
2. **Independensi Residual:** Residual ($\epsilon_i$) tidak saling berkorelasi satu sama lain (tidak ada autokorelasi).
3. **Homoskedastisitas:** Varians dari residual bernilai konstan untuk semua nilai $x$.
4. **Normalitas Residual:** Residual terdistribusi secara normal (penting untuk uji hipotesis dan pembuatan confidence interval).

#### 7. Contoh Sederhana (Kalkulasi Manual)
Diberikan data luas rumah ($x$ dalam m²) dan harga ($y$ dalam puluhan juta IDR):

| $i$ | Luas ($x$) | Harga ($y$) |
|---|---|---|
| 1 | 2 | 4 |
| 2 | 3 | 5 |
| 3 | 5 | 7 |

* Hitung rata-rata: $\bar{x} = \frac{2+3+5}{3} = 3.333$, $\bar{y} = \frac{4+5+7}{3} = 5.333$
* Deviation & Product calculation:
  - $(x_1 - \bar{x}) = -1.333$, $(y_1 - \bar{y}) = -1.333 \rightarrow$ Prod = $1.777$, $(x_1 - \bar{x})^2 = 1.777$
  - $(x_2 - \bar{x}) = -0.333$, $(y_2 - \bar{y}) = -0.333 \rightarrow$ Prod = $0.111$, $(x_2 - \bar{x})^2 = 0.111$
  - $(x_3 - \bar{x}) = 1.667$, $(y_3 - \bar{y}) = 1.667 \rightarrow$ Prod = $2.778$, $(x_3 - \bar{x})^2 = 2.778$
* Sum of products $= 1.777 + 0.111 + 2.778 = 4.666$
* Sum of squared $x$ dev $= 1.777 + 0.111 + 2.778 = 4.666$
* $\hat{\beta}_1 = \frac{4.666}{4.666} = 1.0$
* $\hat{\beta}_0 = 5.333 - (1.0 \times 3.333) = 2.0$

Persamaan prediksi: $\hat{y} = 2.0 + 1.0 x$.
Jika luas rumah $= 4$, maka prediksi harga $= 2.0 + 1.0(4) = 6.0$ (60 juta IDR).

#### 8. Implementasi Python
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Data latihan
X = np.array([[2], [3], [5]])  # Fitur (harus 2D array di scikit-learn)
y = np.array([4, 5, 7])        # Target

# Inisialisasi dan fitting model
model = LinearRegression()
model.fit(X, y)

# Menampilkan hasil parameter
print(f"Intercept (beta_0): {model.intercept_:.2f}")
print(f"Slope (beta_1): {model.coef_[0]:.2f}")

# Prediksi data baru
X_new = np.array([[4]])
y_pred = model.predict(X_new)
print(f"Prediksi untuk X=4: {y_pred[0]:.2f}")
```

#### 9. Kelebihan dan Kekurangan
- **Kelebihan:** Sangat mudah dipahami dan diinterpretasikan, efisien secara komputasi, tidak memerlukan hyperparameter tuning.
- **Kekurangan:** Terlalu sederhana untuk hubungan yang kompleks/non-linear, sangat sensitif terhadap pencilan (*outliers*).

#### 10. Kapan Digunakan / Tidak Digunakan
- **Cocok:** Ketika variabel target memiliki hubungan linear jelas dengan satu variabel input dan interpretabilitas tinggi dibutuhkan.
- **Tidak Cocok:** Ketika data memiliki pola non-linear atau melibatkan banyak fitur saling berinteraksi.

---

### Multiple Linear Regression

#### 1. Konsep dan Definisi
Multiple Linear Regression (Regresi Linear Berganda) adalah perluasan dari Simple Linear Regression yang memprediksi satu variabel target kontinu $y$ menggunakan **dua atau lebih** variabel prediktor/fitur ($x_1, x_2, \dots, x_p$).

#### 2. Intuisi
Jika Simple Linear Regression membentuk **garis 2D**, maka Multiple Linear Regression membentuk **bidang datar (plane 3D)** untuk dua fitur, atau **hiperbidang (hyperplane $p$-dimensi)** untuk $p$ fitur.

#### 3. Dasar Matematika
Persamaan matematis untuk $p$ fitur:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$

Dalam notasi matriks (vektor):

$$y = X\beta + \epsilon$$

Di mana:
- $y$ adalah vektor target berukuran $(n \times 1)$.
- $X$ adalah matriks desain (design matrix) berukuran $(n \times (p+1))$, di mana kolom pertama berisi angka 1 untuk mengkompensasi $\beta_0$.
- $\beta$ adalah vektor parameter/koefisien berukuran $((p+1) \times 1)$.
- $\epsilon$ adalah vektor error berukuran $(n \times 1)$.

##### Solusi Matrix OLS
Untuk meminimalkan $RSS(\beta) = (y - X\beta)^T (y - X\beta)$, turunan terhadap $\beta$ disetel ke nol, menghasilkan rumus **Normal Equation**:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

Syarat penting: Matriks $X^T X$ harus bersifat *invertible* (memiliki invers/penentu tidak nol).

#### 4. Asumsi Tambahan
Selain 4 asumsi Simple Linear Regression, Multiple Linear Regression menambahkan asumsi:
5. **Tidak Ada Multikolinearitas Sempurna (No Multicollinearity):** Variabel prediktor tidak boleh memiliki korelasi linear yang sangat tinggi satu sama lain. Jika fitur $x_1$ dan $x_2$ berkorelasi sempurna, matriks $X^T X$ menjadi non-invertible (singular).

#### 5. Perbandingan: Simple vs Multiple Linear Regression

| Aspek | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| **Jumlah Fitur ($p$)** | 1 | $\ge 2$ |
| **Geometri Prediksi** | Garis 2D | Bidang/Hiperbidang ($p+1$ dimensi) |
| **Kompleksitas Solusi** | Formula skalar | Invers Matriks $(X^T X)^{-1}$ |
| **Tantangan Utama** | Underfitting | Multikolinearitas & Overfitting |

#### 6. Implementasi Python
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset buatan
data = {
    'Luas_m2': [50, 70, 100, 120, 150],
    'Jml_Kamar': [1, 2, 3, 3, 4],
    'Harga_Juta': [300, 450, 600, 700, 850]
}
df = pd.DataFrame(data)

X = df[['Luas_m2', 'Jml_Kamar']]
y = df['Harga_Juta']

model = LinearRegression()
model.fit(X, y)

print("Intercept (beta_0):", model.intercept_)
print("Koefisien (beta_1, beta_2):", model.coef_)

# Prediksi rumah dengan luas 90m2 dan 2 kamar
prediksi = model.predict([[90, 2]])
print("Prediksi Harga:", prediksi[0])
```

#### 7. Kesalahan Umum (Misconception)
- **Misconception:** Menambahkan lebih banyak fitur pasti meningkatkan kinerja model.
- **Fakta:** Menambahkan fitur yang tidak relevan dapat meningkatkan $R^2$ secara semu pada data latih, namun meningkatkan varians model dan risiko *overfitting*. Gunakan *Adjusted $R^2$* untuk mengevaluasi model berganda.

---

### Polynomial Regression

#### 1. Konsep dan Definisi
Polynomial Regression (Regresi Polinomial) adalah bentuk analisis regresi di mana hubungan antara variabel independen $x$ dan variabel dependen $y$ dimodelkan sebagai polinomial derajat ke-$d$.

Meskipun memodelkan hubungan non-linear antara $x$ dan $y$, **Polynomial Regression tetap dikategorikan sebagai model linear**.

#### 2. Mengapa Disebut Model Linear?
Di dalam Machine Learning dan Statistik, sebuah model dinamakan **linear** jika persamaan tersebut **linear terhadap koefisien/parameternya ($\beta$)**, bukan terhadap fitur masukan ($x$).

#### 3. Dasar Matematika
Persamaan Regresi Polinomial derajat $d$ dengan satu fitur masukan:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \dots + \beta_d x^d + \epsilon$$

Jika kita mendefinisikan fitur baru: $z_1 = x, z_2 = x^2, \dots, z_d = x^d$, maka persamaan berubah menjadi:

$$y = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + \dots + \beta_d z_d + \epsilon$$

Persamaan di atas identik dengan **Multiple Linear Regression**. Solusinya tetap dapat dihitung menggunakan rumus OLS yang sama.

#### 4. Polynomial Features Expansion
Proses pelatihan Regresi Polinomial diawali dengan transformasi *feature expansion*:
- Input asli: $[x_1, x_2]$
- Transforma derajat 2 ($d=2$): $[1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]$ (Termasuk term interaksi $x_1 x_2$).

#### 5. Perbandingan: Linear vs Polynomial Regression

```
Underfitting (Degree 1)        Balanced (Degree 2)          Overfitting (Degree 15)
     y |   /                      y |  .--.                    y | /\/\/\
       |  /                         | /    \                     |/        \
       | /                          |/      '                    |          \
       +------ x                    +------ x                    +------ x
```

- **Degree 1 (Linear):** *High Bias*, *Low Variance*. Gagal menangkap pola kurva (*underfitting*).
- **Degree Optimal (misal 2 atau 3):** Keseimbangan baik antara *Bias* dan *Variance*.
- **Degree Sangat Tinggi (misal 15):** *Low Bias*, *High Variance*. Mempelajari *noise* dari data latih secara berlebihan (*overfitting*).

#### 6. Implementasi Python
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Membuat data non-linear bergelombang
np.random.seed(42)
X = 6 * np.random.rand(50, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(50, 1)

# Membuat pipeline Polynomial Regression derajat 2
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X, y)

# Prediksi nilai
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = poly_model.predict(X_test)

# Menampilkan hasil koefisien
lin_reg = poly_model.named_steps['linearregression']
print("Koefisien Polinomial:", lin_reg.coef_)
```

#### 7. Kelebihan dan Kekurangan
- **Kelebihan:** Mampu menangkap hubungan non-linear tanpa harus berpindah ke algoritma yang jauh lebih rumit.
- **Kekurangan:** Jumlah fitur bertambah secara eksponensial seiring meningkatnya derajat $d$ dan jumlah fitur asli, rentan terhadap masalah *multicollinearity* serta *overfitting* di ujung rentang data (ekstrapolasi buruk).

---

## Konsep Pendukung: Bias-Variance Tradeoff & Regularization

Sebelum mempelajari Ridge, Lasso, dan Elastic Net, kita harus memahami mengapa algoritma OLS biasa bisa mengalami kegagalan pada data yang kompleks atau berdimensi tinggi.

### 1. Bias-Variance Tradeoff
Total Error dari sebuah model prediksi dapat didekomposisi menjadi tiga komponen:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias:** Kesalahan akibat asumsi model yang terlalu sederhana. High Bias $\rightarrow$ **Underfitting**.
- **Variance:** Sensitivitas model terhadap fluktuasi kecil pada dataset latih. High Variance $\rightarrow$ **Overfitting**.
- **Irreducible Error ($\sigma^2$):** Noise alami pada data yang tidak dapat dihilangkan oleh model apa pun.

```
       Error ^
             |       Total Error
             |  \                   /
             |   \    /\           /
             |    \  /  \  Bias^2 /
             |     \/    \       /   Variance
             |     /\     \_____/____
             |    /  \_________/
             +-----------------------------> Kompleksitas Model
```

### 2. Regularization (Regulerisasi)
Regularization adalah teknik untuk mencegah *overfitting* dengan cara memberikan sanksi (*penalty*) terhadap besarnya nilai koefisien ($\beta$) dalam fungsi kerugian (*loss function*).

Dengan menambahkan penalti, model dipaksa untuk memilih koefisien yang lebih kecil/sederhana, menukar sedikit kenaikan *Bias* untuk penurunan *Variance* yang signifikan.

---

### Ridge Regression

#### 1. Konsep dan Definisi
Ridge Regression (dikenal juga sebagai **$L_2$ Regularization** atau Tikhonov Regularization) adalah variasi dari Linear Regression yang menambahkan sanksi berupa jumlah kuadrat dari koefisien pada fungsi kerugian OLS.

#### 2. Intuisi
Dalam OLS biasa, jika fitur-fitur saling berkorelasi erat (multikolinearitas), model dapat memberikan koefisien positif yang sangat besar pada satu fitur dan koefisien negatif yang sangat besar pada fitur lainnya untuk saling meniadakan. Ridge Regression mengekang ("memenjarakan") nilai koefisien agar tidak bisa tumbuh menjadi sangat besar.

#### 3. Dasar Matematika
Fungsi kerugian (Objective Function) Ridge Regression:

$$J_{Ridge}(\beta) = \sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{p} x_{ij}\beta_j \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

Atau dalam notasi vektor/matriks:

$$J_{Ridge}(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \|\beta\|_2^2$$

Di mana:
- $\|\beta\|_2^2 = \sum_{j=1}^{p} \beta_j^2$ adalah norma $L_2$ kuadrat dari vektor koefisien (intersept $\beta_0$ biasanya tidak dikenakan sanksi).
- $\lambda \ge 0$ (dalam Scikit-Learn disebut parameter `alpha`): Hyperparameter pembobot sanksi.
  - Jika $\lambda = 0$: Ridge menjadi identik dengan Ordinary Least Squares (OLS).
  - Jika $\lambda \rightarrow \infty$: Semua koefisien $\beta_j \rightarrow 0$ (model menjadi konstan sama dengan rata-rata target).

##### Solusi Closed-Form Matriks
Ridge Regression memiliki solusi matematis eksplisit:

$$\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y$$

Di mana $I$ adalah matriks identitas. Penambahan $\lambda I$ memastikan bahwa matriks $(X^T X + \lambda I)$ **selalu memiliki invers**, bahkan jika $X^T X$ mengalami multikolinearitas atau jumlah fitur $p$ lebih besar dari jumlah sampel $n$ ($p > n$).

#### 4. Catatan Penting: Feature Scaling
Karena sanksi dihitung berdasarkan magnitudo numerik $\beta_j^2$, **Ridge Regression sangat sensitif terhadap skala fitur**. Fitur dengan skala besar akan menghasilkan koefisien kecil secara alami, sedangkan fitur skala kecil membutuhkan koefisien besar. 
*Wajib melakukan Standardisasi (StandardScaler: mean=0, std=1) sebelum menjalankan Ridge Regression.*

#### 5. Implementasi Python
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Generate data acak dengan multikolinearitas
np.random.seed(42)
X = np.random.randn(100, 5)
# Buat kolom ke-2 sangat berkorelasi dengan kolom ke-0
X[:, 1] = X[:, 0] + 0.01 * np.random.randn(100)
y = 3 * X[:, 0] + 2 * X[:, 2] + np.random.randn(100)

# Build pipeline (Scaling + Ridge)
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
ridge_model.fit(X, y)

print("Koefisien Ridge:", ridge_model.named_steps['ridge'].coef_)
```

#### 6. Kelebihan dan Kekurangan
- **Kelebihan:** Efektif menangani multikolinearitas, stabil saat $p > n$, mencegah overfitting.
- **Kekurangan:** Tidak melakukan pembagian/pemilihan fitur (*feature selection*). Koefisien diperkecil mendekati nol tetapi **tidak pernah menyentuh nol secara tepat**.

---

### Lasso Regression

#### 1. Konsep dan Definisi
Lasso Regression (**Least Absolute Shrinkage and Selection Operator**, atau **$L_1$ Regularization**) adalah teknik regresi ter-regulerisasi yang menambahkan sanksi berupa jumlah nilai mutlak (absolut) dari koefisien model.

#### 2. Intuisi
Berbeda dengan Ridge yang menekan semua koefisien secara proporsional, Lasso mampu menekan koefisien dari fitur-fitur yang kurang penting **tepat menjadi nol**. Ini berarti Lasso bertindak otomatis sebagai metode **Feature Selection (Seleksi Fitur)**.

#### 3. Dasar Matematika
Fungsi kerugian Lasso Regression:

$$J_{Lasso}(\beta) = \sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{p} x_{ij}\beta_j \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

Atau dalam notasi norma $L_1$:

$$J_{Lasso}(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \|\beta\|_1$$

Di mana $\|\beta\|_1 = \sum_{j=1}^{p} |\beta_j|$.

##### Mengapa $L_1$ Menghasilkan Koefisien Nol (Sparsity)?
Secara geometris, optimasi terikat dapat diilustrasikan sebagai pencarian titik potong antara kontur fungsi kerugian OLS (berbentuk elips) dengan ruang kendala sanksi (*constraint region*):

```
       L1 (Lasso) Constraint               L2 (Ridge) Constraint
            \  beta_2  /                        beta_2
             \   |   /                            |  ***
              \  |  /                          *  |     *
               \ | /                         *    |       *
       ----------+---------- beta_1   ----------+---------- beta_1
               / | \                         *    |       *
              /  |  \                          *  |     *
             /   |   \                            |  ***
            /    |    \                           |
```

- Kontur kendala $L_1$ berbentuk **berlian/belah ketupat** (memiliki sudut tajam tepat di sepanjang sumbu koordinat $\beta_j = 0$). Seringkali titik singgung pertama kontur OLS terjadi pada sudut-sudut ini.
- Kontur kendala $L_2$ berbentuk **lingkaran/bola mulus** tanpa sudut, sehingga titik singgung hampir tidak pernah jatuh tepat di titik sumbu zero.

##### Solusi Optimasi
Fungsi nilai absolut $|\beta_j|$ tidak memiliki turunan pada titik $\beta_j = 0$. Oleh karena itu, Lasso **tidak memiliki solusi bentuk tertutup (closed-form solution)** seperti OLS atau Ridge. Lasso diselesaikan menggunakan algoritma optimasi numerik seperti **Coordinate Descent**.

#### 4. Perbandingan: Ridge vs Lasso Regression

| Aspek | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
|---|---|---|
| **Sanksi Penalti** | $\lambda \sum \beta_j^2$ | $\lambda \sum \|\beta_j\|$ |
| **Bentuk Kendala** | Lingkaran/Bola | Berlian/Piramida |
| **Solusi Analitis** | Ada: $(X^T X + \lambda I)^{-1} X^T y$ | Tidak ada (Gunakan Coordinate Descent) |
| **Efek Koefisien** | Mengecilkan $\beta \rightarrow 0$ | Mengecilkan $\beta$ hingga **tepat 0** |
| **Fitur Seleksi** | Tidak (Menyimpan semua fitur) | Ya (Menghasilkan *Sparse Model*) |
| **Kondisi Terbaik** | Banyak fitur relevan berkorelasi | Sedikit fitur penting di antara banyak fitur *noise* |

#### 5. Implementasi Python
```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Data dengan 5 fitur, namun hanya fitur 0 dan 2 yang benar-benar berpengaruh
np.random.seed(42)
X = np.random.randn(100, 5)
y = 5 * X[:, 0] - 3 * X[:, 2] + np.random.randn(100) * 0.1

# Pipeline Lasso
lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=0.2))
lasso_model.fit(X, y)

koefisien = lasso_model.named_steps['lasso'].coef_
print("Koefisien Lasso:", np.round(koefisien, 3))
# Hasil akan memperlihatkan koefisien fitur 1, 3, 4 bernilai 0.0
```

#### 6. Keterbatasan Lasso
- Jika terdapat sekelompok fitur yang saling berkorelasi tinggi (*high collinearity group*), Lasso cenderung **hanya memilih satu fitur acak** dari kelompok tersebut dan membuat sisanya menjadi nol.
- Ketika jumlah fitur lebih besar dari sampel ($p > n$), Lasso maksimal hanya dapat memilih $n$ fitur.

---

### Elastic Net

#### 1. Konsep dan Definisi
Elastic Net adalah algoritma regresi ter-regulerisasi yang mengombinasikan sanksi **$L_1$ (Lasso)** dan **$L_2$ (Ridge)** secara simultan dalam satu fungsi kerugian.

#### 2. Masalah yang Diselesaikan
Elastic Net diciptakan oleh Zou dan Hastie (2005) untuk mengatasi dua keterbatasan utama Lasso:
1. Ketika $p > n$, Lasso terbatasi memilih maksimal $n$ variabel.
2. Ketika ada kelompok fitur yang saling berkorelasi kuat, Lasso hanya memilih satu secara acak. Elastic Net mampu mempertahankan seluruh kelompok fitur tersebut (*grouping effect*).

#### 3. Dasar Matematika
Fungsi kerugian Elastic Net dapat dituliskan sebagai:

$$J_{ElasticNet}(\beta) = RSS(\beta) + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

Dalam konversi formulasi yang digunakan oleh **Scikit-Learn**, rumusan menggunakan dua hyperparameter utama: `alpha` ($\alpha$) dan `l1_ratio` ($r$):

$$J_{ElasticNet}(\beta) = \frac{1}{2n} RSS(\beta) + \alpha \cdot r \|\beta\|_1 + \frac{\alpha(1 - r)}{2} \|\beta\|_2^2$$

Di mana:
- $\alpha \ge 0$: Kekuatan total penalti regulerisasi.
- $r \in [0, 1]$ (`l1_ratio`): Rasio pencampuran antara penalti $L_1$ dan $L_2$.
  - Jika $r = 1$: Model menjadi sepenuhnya **Lasso Regression**.
  - Jika $r = 0$: Model menjadi sepenuhnya **Ridge Regression**.
  - Jika $0 < r < 1$: Model menggabungkan karakteristik Lasso dan Ridge.

#### 4. Perbandingan Lengkap Regulerisasi Linear

| Algoritma | Sanksi Penalti | Keunggulan Utama | Kasus Penggunaan Ideal |
|---|---|---|---|
| **Linear Regression (OLS)** | Tidak ada | Tanpa bias penalti, interpretasi mudah | Dataset kecil/sederhana, $n \gg p$, tidak ada multikolinearitas |
| **Ridge Regression** | $L_2$ ($\sum \beta^2$) | Mengatasi multikolinearitas, stabil | Banyak fitur yang hampir semuanya relevan |
| **Lasso Regression** | $L_1$ ($\sum \|\beta\|$) | Efektif eliminasi fitur tak penting (*sparsity*) | Memiliki sangat banyak fitur tetapi diperkirakan hanya sedikit yang berpengaruh |
| **Elastic Net** | $L_1 + L_2$ | Grouping effect, stabil pada $p > n$ | Dataset berdimensi tinggi ($p \gg n$) dengan korelatif antar fitur |

#### 5. Implementasi Python
```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Data sintetis
np.random.seed(42)
X = np.random.randn(100, 6)
# Kolom 0 dan 1 saling berkorelasi
X[:, 1] = X[:, 0] + 0.05 * np.random.randn(100)
y = 3 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(100)

# Pipeline ElasticNet (alpha=0.1, l1_ratio=0.5 -> 50% L1, 50% L2)
elastic_model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))
elastic_model.fit(X, y)

print("Koefisien Elastic Net:", np.round(elastic_model.named_steps['elasticnet'].coef_, 3))
```

#### 6. Inti yang Harus Dipahami dari Seluruh Linear Models
1. **Model Linear** memprediksi output berupa kombinasi linear bobot $\beta$ dan input $x$.
2. **OLS** meminimalkan kuadrat kesalahan (RSS), tetapi rentan *overfitting* dan masalah multikolinearitas.
3. **Polynomial Regression** mengubah fitur untuk memodelkan garis lengkung, namun tetap merupakan *linear model* terhadap parameter $\beta$.
4. **Ridge ($L_2$)** mengecilkan koefisien mendekati nol untuk mengatasi multikolinearitas (menurunkan varians).
5. **Lasso ($L_1$)** mengecilkan koefisien tepat hingga nol untuk seleksi fitur otomatis.
6. **Elastic Net ($L_1+L_2$)** mengombinasikan keunggulan Ridge dan Lasso untuk dataset kompleks berdimensi tinggi.

---

## References

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R* (2nd ed.). Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
3. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), 301-320.
4. Scikit-Learn Documentation - Linear Models: https://scikit-learn.org/stable/modules/linear_model.html