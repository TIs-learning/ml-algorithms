Business Requirement & Problem Definition
1. Background
[Nama Industri/Domain Properti/Bisnis/Subjek] menghadapi tantangan dalam [Proses/Kegiatan Utama]. Saat ini, proses tersebut dilakukan secara [Manual / Komputasi Tradisional / Intuisi], yang berisiko menimbulkan kendala berikut:

[Risiko A / Kerugian Sisi 1]: [Dampak negatif jika estimasi/klasifikasi terlalu tinggi atau salah].

[Risiko B / Kerugian Sisi 2]: [Dampak negatif jika estimasi/klasifikasi terlalu rendah atau salah].

[Kompleksitas Faktor]: Keputusan dipengaruhi oleh kombinasi kompleks dari [Sebutkan 3-5 kategori variabel utama].

Oleh karena itu, diperlukan pendekatan berbasis data (data-driven) menggunakan [Tipe Machine Learning] untuk [Tujuan Utama] secara otomatis, objektif, dan presisi.

2. Business / Project Objective
[Kata Kerja Aksi 1]: [Memprediksi / Mengklasifikasikan / Mengelompokkan] [Nama Target] secara akurat berdasarkan [Fitur Utama].

[Kata Kerja Aksi 2]: [Meningkatkan / Meminimalkan] [Efisiensi waktu / Margin eror / Biaya operasional].

[Kata Kerja Aksi 3]: Mengidentifikasi fitur yang paling dominan (Feature Importance) dalam mempengaruhi [Target].

3. Problem Definition
Business Question
Berapa/Apakah [Pertanyaan utama yang ingin dijawab oleh model] berdasarkan [Kelompok variabel utama]?

Machine Learning Framing
Problem Type: [Supervised Learning / Unsupervised Learning]

Task: [Regression / Binary Classification / Multi-class Classification / Clustering]

Target Variable: [Nama Kolom Target] ([Penjelasan rinci satuan/makna kolom target])

Evaluation Metric: [RMSLE / RMSE / MAE / F1-Score / ROC-AUC / Accuracy] — [Alasan singkat pemilihan metrik]

4. Input Data
Dataset terdiri dari data historis [Objek Data] berjumlah [Jumlah Baris/Kolom] yang dibagi menjadi:

Numerical Features: [Kolom 1], [Kolom 2], [Dst] (Ukuran kuantitatif seperti luas, jumlah, tahun, nilai numerik).

Categorical Features: [Kolom A], [Kolom B], [Dst] (Data teks, kelas nominal, atau tingkatan ordinal).

Identifier Features: [Kolom ID] (Pengenal unik data yang perlu diabaikan dari model).

Target Variable: [Nama Target]

5. Data Notes & Constraints
(Dapat diisi awal berdasarkan asumsi/deskripsi dataset, lalu diperbarui setelah tahap EDA)

Distribusi Target: [Apakah distribusi normal, mencong/skewed, atau imbalanced? sebutkan rencana penanganan].

Missing Values: [Apakah ada missing value yang bermakna implisit atau butuh imputasi khusus?].

Tantangan Fitur: [Sebutkan isu seperti High Dimensionality / Multikolinearitas / Outliers].

Regulasi / Batasan Etis: [Pemeriksaan terhadap isu privasi, bias data, atau pembatasan penggunaan fitur tertentu jika ada].

6. Expected Outcome
Model Pipeline: Model Machine Learning dengan performa metrik evaluasi yang optimal pada data uji (test set).

Wawasan Bisnis (Feature Importance): Pemahaman mendalam tentang variabel penentu utama yang memengaruhi [Nama Target].

Sistem Prediksi: Pipeline kode terstruktur yang siap dijalankan (data loading, preprocessing, modeling, hingga inference).

Tips Penggunaan di Masa Depan:

Untuk Kompetisi (Kaggle): Tekankan bagian Evaluation Metric dan Data Constraints.

Untuk Portofolio Kerja: Tekankan bagian Business Objective dan Expected Outcome untuk menunjukkan kemampuan berpikir dari sisi bisnis/solusi.