# Machine Learning: Linear Models

Materi ini membahas keluarga **Linear Models** untuk regresi. Fokusnya adalah membangun intuisi, memahami mekanisme training, dan mengetahui kapan tiap algoritma cocok digunakan.

Prasyarat singkat:
- Aljabar linear dasar (vektor, matriks, dot product).
- Kalkulus dasar (turunan, gradien).
- Statistik dasar (mean, varians, korelasi).

---

## Konsep Pendukung: Regresi & Loss Function

**Regresi** adalah tugas memprediksi nilai kontinu (misalnya harga rumah, suhu, gaji) berdasarkan fitur input.

**Loss function** mengukur seberapa "salah" prediksi model dibanding nilai sebenarnya. Untuk regresi, yang paling umum adalah **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- $y_i$: nilai aktual sampel ke-$i$.
- $\hat{y}_i$: nilai prediksi model.
- $n$: jumlah sampel.
- Kuadrat digunakan agar error positif dan negatif tidak saling menghapus, sekaligus memberi penalti besar pada error besar.

**Gradient Descent** adalah metode iteratif untuk meminimalkan loss dengan bergerak berlawanan arah gradien:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

- $\theta$: parameter model (bobot).
- $\eta$: learning rate.
- $\nabla L$: gradien loss terhadap parameter.

---

## 1. Linear Regression

### Konsep
Linear Regression (Simple Linear Regression) memodelkan hubungan antara **satu fitur** $x$ dan **target kontinu** $y$ sebagai garis lurus:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

- $\beta_0$: intercept (nilai $y$ saat $x=0$).
- $\beta_1$: slope (perubahan $y$ per satu unit perubahan $x$).

### Intuisi
Kita mencari garis lurus terbaik yang "melewati" titik-titik data sedekat mungkin. "Terbaik" didefinisikan sebagai garis yang meminimalkan jumlah kuadrat jarak vertikal antara titik data dan garis.

### Cara Kerja & Training: Ordinary Least Squares (OLS)
Kita minimalkan **Sum of Squared Errors (SSE)**:

$$
L(\beta_0, \beta_1) = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2
$$

Dengan menurunkan $L$ terhadap $\beta_0$ dan $\beta_1$ lalu menyamakan dengan nol, diperoleh **solusi tertutup (closed-form)**:

$$
\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}
$$

Intuisi $\beta_1$: kovarians $(x,y)$ dibagi varians $x$ — seberapa kuat $y$ ikut bergerak saat $x$ bergerak, dinormalisasi terhadap sebaran $x$.

### Asumsi (Gauss-Markov)
1. **Linearitas** hubungan $x$ dan $y$.
2. **Independensi** residual antar observasi.
3. **Homoskedastisitas**: varians residual konstan.
4. **Normalitas** residual (untuk inferensi statistik).
5. Tidak ada **multikolinearitas** (untuk versi multi-fitur).

### Contoh Manual
Data: $x = [1, 2, 3, 4]$, $y = [2, 4, 5, 4]$.

- $\bar{x} = 2.5$, $\bar{y} = 3.75$.
- Pembilang: $(1-2.5)(2-3.75) + (2-2.5)(4-3.75) + (3-2.5)(5-3.75) + (4-2.5)(4-3.75) = 2.625 - 0.125 + 0.625 + 0.375 = 3.5$.
- Penyebut: $(1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2 = 5$.
- $\beta_1 = 0.7$, $\beta_0 = 3.75 - 0.7 \times 2.5 = 2.0$.
- Model: $\hat{y} = 2.0 + 0.7x$.

### Implementasi Python
~~~python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 4])

model = LinearRegression().fit(X, y)
print("intercept:", model.intercept_)   # ~2.0
print("slope:", model.coef_)            # ~0.7
print("prediksi x=5:", model.predict([[5]]))
~~~

### Kelebihan
- Sederhana, cepat, mudah diinterpretasikan.
- Ada solusi tertutup (tidak perlu iterasi).
- Baseline yang kuat.

### Kekurangan
- Hanya menangkap hubungan linear.
- Sensitif terhadap outlier (kuadrat memperbesar pengaruh outlier).
- Asumsi ketat.

### Kapan Digunakan
- Hubungan antar variabel mendekati linear.
- Butuh interpretabilitas (koefisien punya arti).

### Kapan Tidak Cocok
- Hubungan sangat non-linear.
- Banyak outlier tanpa penanganan.

---

## 2. Multiple Linear Regression

### Konsep
Perluasan Linear Regression untuk **banyak fitur**:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p = \mathbf{x}^\top \boldsymbol{\beta}
$$

Dalam bentuk matriks:

$$
\hat{\mathbf{y}} = X \boldsymbol{\beta}
$$

- $X$: matriks desain berukuran $n \times (p+1)$ (kolom pertama = 1 untuk intercept).
- $\boldsymbol{\beta}$: vektor koefisien.

### Intuisi
Alih-alih garis, kita mencari **hyperplane** di ruang berdimensi tinggi yang paling dekat dengan titik-titik data.

### Training: Normal Equation
Meminimalkan $\|y - X\beta\|^2$ menghasilkan:

$$
\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top y
$$

- $(X^\top X)^{-1}$ ada jika kolom $X$ linearly independent (tidak ada multikolinearitas sempurna).
- Untuk dataset besar, gradient descent atau dekomposisi QR/SVD lebih efisien daripada inversi langsung.

### Interpretasi Koefisien
$\beta_j$ = perubahan rata-rata $y$ per unit kenaikan $x_j$, **dengan menjaga fitur lain konstan** (ceteris paribus).

### Contoh & Implementasi Python
Dataset toy: prediksi harga rumah dari luas dan jumlah kamar.

~~~python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [50, 1],
    [70, 2],
    [90, 3],
    [110, 3],
    [150, 4],
])
y = np.array([150, 200, 250, 280, 360])

model = LinearRegression().fit(X, y)
print("intercept:", model.intercept_)
print("coef:", model.coef_)  # [koef_luas, koef_kamar]
print("prediksi:", model.predict([[100, 3]]))
~~~

### Perbandingan dengan Simple Linear Regression

| Aspek | Simple LR | Multiple LR |
|---|---|---|
| Jumlah fitur | 1 | ≥ 2 |
| Bentuk model | Garis | Hyperplane |
| Solusi tertutup | Ya | Ya (Normal Equation) |
| Multikolinearitas | Tidak relevan | Bisa jadi masalah |

### Kelebihan
- Menangani banyak fitur sekaligus.
- Tetap interpretabel.

### Kekurangan
- **Multikolinearitas** membuat koefisien tidak stabil.
- Sensitif terhadap fitur yang tidak relevan.
- Tetap terbatas pada hubungan linear.

### Kapan Digunakan
- Banyak fitur numerik dengan hubungan linear terhadap target.
- Perlu memahami kontribusi tiap fitur.

---

## 3. Polynomial Regression

### Konsep
Perluasan linear regression untuk hubungan **non-linear**, dengan menambahkan pangkat fitur sebagai fitur baru:

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d
$$

**Penting**: model ini tetap **linear dalam parameter** $\beta$, sehingga bisa dilatih dengan OLS. Yang non-linear adalah hubungan antara $x$ dan $y$.

### Intuisi
Kalau data melengkung (parabola, kubik), garis lurus tidak cukup. Kita tambahkan $x^2, x^3, \dots$ sebagai fitur baru sehingga model bisa menyesuaikan kurva.

### Cara Kerja
1. Transformasi fitur: $x \rightarrow [x, x^2, \dots, x^d]$.
2. Jalankan multiple linear regression pada fitur hasil transformasi.

### Bias-Variance Tradeoff
- Derajat $d$ kecil → underfitting (bias tinggi).
- Derajat $d$ besar → overfitting (variance tinggi), kurva bergelombang liar.

### Implementasi Python
~~~python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([1, 4, 9, 16, 25, 36])  # y = x^2

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
print("prediksi x=7:", model.predict([[7]]))  # ~49
~~~

### Kelebihan
- Menangkap hubungan non-linear tanpa meninggalkan kerangka linear.
- Sederhana untuk diimplementasikan.

### Kekurangan
- Rentan overfitting pada derajat tinggi.
- Ekstrapolasi di luar rentang data sering meledak.
- Jumlah fitur meledak jika banyak variabel input (interaksi + pangkat).

### Kapan Digunakan
- Pola data jelas melengkung namun sederhana.
- Fitur input sedikit.

### Kapan Tidak Cocok
- Data multivariat dengan banyak fitur.
- Butuh stabilitas ekstrapolasi.

---

## Konsep Pendukung: Regularisasi

Saat model terlalu fleksibel (banyak fitur atau derajat tinggi), koefisien bisa membesar dan model overfit. **Regularisasi** menambahkan penalti pada besarnya koefisien di loss function untuk menekan kompleksitas.

Bentuk umum:

$$
L(\beta) = \underbrace{\|y - X\beta\|_2^2}_{\text{loss data}} + \lambda \cdot \underbrace{R(\beta)}_{\text{penalti}}
$$

- $\lambda \geq 0$: hyperparameter kekuatan regularisasi.
- $R(\beta)$: fungsi penalti (bentuk berbeda menghasilkan Ridge, Lasso, Elastic Net).

---

## 4. Ridge Regression (L2 Regularization)

### Konsep
Menambahkan penalti **kuadrat** koefisien:

$$
L(\beta) = \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

Solusi tertutup:

$$
\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

Penambahan $\lambda I$ membuat matriks selalu invertible → mengatasi multikolinearitas.

### Intuisi
Ridge "menyusutkan" (shrink) koefisien menuju nol, tetapi **tidak pernah tepat nol**. Semua fitur tetap dipertahankan, hanya pengaruhnya diperkecil.

### Efek $\lambda$
- $\lambda = 0$: sama dengan OLS.
- $\lambda \to \infty$: semua koefisien mendekati 0.

### Catatan Penting
Fitur harus **di-standarisasi** (mean 0, std 1) sebelum Ridge, karena penalti sensitif terhadap skala.

### Implementasi Python
~~~python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)
~~~

### Kelebihan
- Menangani multikolinearitas.
- Stabil, koefisien tidak meledak.

### Kekurangan
- Tidak melakukan seleksi fitur (semua koefisien tetap non-zero).

### Kapan Digunakan
- Banyak fitur berkorelasi.
- Ingin mempertahankan semua fitur namun dengan koefisien yang terkendali.

---

## 5. Lasso Regression (L1 Regularization)

### Konsep
Menambahkan penalti **nilai absolut** koefisien:

$$
L(\beta) = \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j|
$$

Tidak ada solusi tertutup; dioptimalkan dengan algoritma seperti **coordinate descent**.

### Intuisi
Bentuk penalti L1 (berbentuk "berlian" di ruang koefisien) memiliki sudut-sudut tajam pada sumbu. Solusi optimum sering "menempel" di sudut ini → beberapa koefisien menjadi **tepat nol**.

Efek: Lasso melakukan **seleksi fitur otomatis**.

### Efek $\lambda$
- $\lambda$ kecil: mirip OLS.
- $\lambda$ besar: semakin banyak fitur dieliminasi (koefisien = 0).

### Implementasi Python
~~~python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
model.fit(X_train, y_train)
print(model.named_steps['lasso'].coef_)  # sebagian bisa 0
~~~

### Kelebihan
- Seleksi fitur otomatis → model sparse & interpretabel.
- Efektif ketika hanya sedikit fitur yang benar-benar relevan.

### Kekurangan
- Jika ada grup fitur berkorelasi, Lasso cenderung memilih satu secara acak dan mengabaikan sisanya.
- Tidak stabil pada data dengan multikolinearitas kuat.

### Kapan Digunakan
- Butuh model sparse (misalnya di domain genomik, teks).
- Diduga hanya subset kecil fitur yang penting.

---

## 6. Elastic Net

### Konsep
Kombinasi penalti L1 dan L2:

$$
L(\beta) = \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \beta)^2 + \lambda_1 \sum_{j=1}^{p}|\beta_j| + \lambda_2 \sum_{j=1}^{p}\beta_j^2
$$

Di scikit-learn diparameterisasi dengan `alpha` (kekuatan total) dan `l1_ratio` $\rho$:
- $\rho = 1$ → Lasso murni.
- $\rho = 0$ → Ridge murni.
- $0 < \rho < 1$ → campuran.

### Intuisi
Menggabungkan kelebihan Ridge (stabilitas pada fitur berkorelasi) dan Lasso (sparsity). Ketika ada grup fitur berkorelasi, Elastic Net cenderung memilih grup tersebut bersamaan, bukan hanya satu.

### Implementasi Python
~~~python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5)
)
model.fit(X_train, y_train)
~~~

### Kelebihan
- Stabil pada multikolinearitas.
- Tetap melakukan seleksi fitur.
- Fleksibel via `l1_ratio`.

### Kekurangan
- Dua hyperparameter untuk di-tune (`alpha`, `l1_ratio`).
- Sedikit lebih kompleks untuk diinterpretasi.

### Kapan Digunakan
- Fitur banyak, sebagian berkorelasi tinggi.
- Ingin sparsity sekaligus stabilitas.

---

## Perbandingan Ridge vs Lasso vs Elastic Net

| Aspek | Ridge (L2) | Lasso (L1) | Elastic Net (L1+L2) |
|---|---|---|---|
| Penalti | $\sum \beta_j^2$ | $\sum \|\beta_j\|$ | Kombinasi |
| Koefisien = 0? | Tidak | Ya | Ya |
| Seleksi fitur | Tidak | Ya | Ya |
| Multikolinearitas | Tangguh | Lemah | Tangguh |
| Solusi tertutup | Ya | Tidak | Tidak |
| Hyperparameter | `alpha` | `alpha` | `alpha`, `l1_ratio` |

---

## Kesalahan Umum (Misconceptions)

1. **"Polynomial Regression bukan model linear"** — salah. Linear terhadap parameter, bukan terhadap $x$.
2. **"Regularisasi selalu memperbaiki model"** — hanya jika ada risiko overfitting. Jika model sudah underfit, regularisasi memperburuk.
3. **"Lupa standardisasi sebelum Ridge/Lasso/ElasticNet"** — penalti sensitif skala; hasil bias.
4. **"Koefisien besar = fitur penting"** — hanya benar jika fitur berada di skala yang sama.
5. **"R² tinggi berarti model bagus"** — R² bisa tinggi karena overfitting; validasi dengan test set atau cross-validation.

---

## Inti yang Harus Dipahami

- Semua model di sini adalah **kombinasi linear parameter** yang meminimalkan variasi dari MSE.
- Perbedaannya terletak pada **transformasi fitur** (Polynomial) atau **penalti** (Ridge, Lasso, Elastic Net).
- Regularisasi adalah alat utama melawan overfitting pada model linear.
- Pilihan algoritma tergantung: jumlah fitur, korelasi antar fitur, kebutuhan interpretabilitas, dan kebutuhan seleksi fitur.

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://hastie.su.domains/ElemStatLearn/
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. https://www.statlearning.com/
3. Scikit-learn Developers. *Linear Models — scikit-learn documentation*. https://scikit-learn.org/stable/modules/linear_model.html
4. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288.
5. Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society, Series B*, 67(2), 301–320.
6. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.