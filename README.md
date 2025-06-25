# Laporan Proyek Machine Learning - Andrian Syah

## Domain Proyek

Proyek ini berfokus pada prediksi risiko readmisi pasien diabetes di rumah sakit di Amerika Serikat menggunakan pendekatan machine learning berbasis regresi. Dataset yang digunakan berasal dari UCI Machine Learning Repository (Diabetes 130-US hospitals for years 1999–2008), dan telah diseleksi menjadi subset sebanyak 5000 sampel untuk memenuhi kriteria minimum data (≥500 sampel) serta menjaga efisiensi dalam proses komputasi.
Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Domain: Kesehatan

Permasalahan: Memprediksi kemungkinan readmisi pasien ke rumah sakit guna meningkatkan kualitas perawatan serta menekan biaya operasional kesehatan.

Pendekatan: Regresi, dengan tujuan memprediksi skor risiko readmisi dalam bentuk nilai kontinu.

Dataset: Data kuantitatif terdiri dari 5000 entri dan berbagai fitur penting seperti usia, jumlah prosedur medis, serta penggunaan obat-obatan.

## Business Understanding
### Problem Statements
- Tingginya tingkat readmisi pasien diabetes dalam waktu <30 hari meningkatkan biaya operasional rumah sakit dan membebani sistem kesehatan.
- Kurangnya alat prediktif berbasis data menghambat identifikasi pasien berisiko tinggi untuk intervensi dini.

### Goals
- Mengembangkan model regresi machine learning untuk memprediksi risiko readmisi pasien diabetes, diukur dengan metrik MAE (akurasi absolut), MSE (sensitivitas terhadap kesalahan besar), dan R² (kecocokan model).
- Mengidentifikasi faktor klinis utama yang memengaruhi risiko readmisi untuk mendukung pengambilan keputusan klinis.

### Solution Statements
- Membandingkan performa tiga algoritma regresi (Regresi Linear, Random Forest, XGBoost) menggunakan metrik MAE, MSE, dan R² untuk memilih model terbaik.
- Melakukan penyetelan hiperparameter pada Random Forest dan XGBoost menggunakan GridSearchCV untuk meningkatkan akurasi prediksi.
- Menganalisis pentingnya fitur untuk mengidentifikasi faktor klinis utama yang berkontribusi pada risiko readmisi.

## Data Understanding
Dataset yang digunakan adalah subset 5000 sampel dari *Diabetes 130-US Hospitals for Years 1999-2008*, tersedia di UCI Machine Learning Repository: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

### Daftar Fitur
1. **encounter_id**: ID kunjungan (numerik, dihapus karena tidak relevan).  
2. **patient_nbr**: ID pasien (numerik, dihapus karena tidak relevan).  
3. **race**: Ras pasien (kategorikal, 113 missing, diisi 'Unknown', dienkode).  
4. **gender**: Jenis kelamin (kategorikal, dienkode).  
5. **age**: Kelompok usia (kategorikal, dikonversi ke `kelompok_usia`, dihapus).  
6. **weight**: Berat badan (kategorikal, 96% missing, dihapus).  
7. **admission_type_id**: Jenis penerimaan (numerik, dienkode).  
8. **discharge_disposition_id**: Status keluar (numerik, dienkode).  
9. **admission_source_id**: Sumber penerimaan (numerik, dienkode).  
10. **time_in_hospital**: Lama tinggal (numerik, winsorized).  
11. **payer_code**: Kode pembayar (kategorikal, 39.3% missing, dihapus).  
12. **medical_specialty**: Spesialisasi dokter (kategorikal, 48.7% missing, dihapus).  
13. **num_lab_procedures**: Jumlah prosedur lab (numerik, digunakan untuk `total_prosedur`).  
14. **num_procedures**: Jumlah prosedur lain (numerik, digunakan untuk `total_prosedur`).  
15. **num_medications**: Jumlah obat (numerik, winsorized).  
16. **number_outpatient**: Kunjungan rawat jalan (numerik, digunakan untuk `total_prosedur`).  
17. **number_emergency**: Kunjungan darurat (numerik, digunakan untuk `total_prosedur`).  
18. **number_inpatient**: Kunjungan rawat inap (numerik, digunakan untuk `total_prosedur`).  
19. **diag_1**: Diagnosis utama (kategorikal, 1 missing, diisi 'Unknown', dienkode, dihapus).  
20. **diag_2**: Diagnosis sekunder (kategorikal, 16 missing, diisi 'Unknown', dienkode, dihapus).  
21. **diag_3**: Diagnosis tambahan (kategorikal, 73 missing, diisi 'Unknown', dienkode, dihapus).  
22. **number_diagnoses**: Jumlah diagnosis (numerik, digunakan).  
23. **max_glu_serum**: Tes glukosa (kategorikal, 94.8% missing, diisi 'Unknown', dienkode).  
24. **A1Cresult**: Tes HbA1c (kategorikal, 83.1% missing, diisi 'Unknown', dienkode).  
25. **metformin**: Status metformin (kategorikal, dienkode).  
26. **repaglinide**: Status repaglinide (kategorikal, dienkode).  
27. **nateglinide**: Status nateglinide (kategorikal, dienkode).  
28. **chlorpropamide**: Status chlorpropamide (kategorikal, dienkode).  
29. **glimepiride**: Status glimepiride (kategorikal, dienkode).  
30. **acetohexamide**: Status acetohexamide (kategorikal, dienkode).  
31. **glipizide**: Status glipizide (kategorikal, dienkode).  
32. **glyburide**: Status glyburide (kategorikal, dienkode).  
33. **tolbutamide**: Status tolbutamide (kategorikal, dienkode).  
34. **pioglitazone**: Status pioglitazone (kategorikal, dienkode).  
35. **rosiglitazone**: Status rosiglitazone (kategorikal, dienkode).  
36. **acarbose**: Status acarbose (kategorikal, dienkode).  
37. **miglitol**: Status miglitol (kategorikal, dienkode).  
38. **troglitazone**: Status troglitazone (kategorikal, dienkode).  
39. **tolazamide**: Status tolazamide (kategorikal, dienkode).  
40. **examide**: Status examide (kategorikal, dienkode).  
41. **citoglipton**: Status citoglipton (kategorikal, dienkode).  
42. **insulin**: Status insulin (kategorikal, dienkode).  
43. **glyburide-metformin**: Status glyburide-metformin (kategorikal, dienkode).  
44. **glipizide-metformin**: Status glipizide-metformin (kategorikal, dienkode).  
45. **glimepiride-pioglitazone**: Status glimepiride-pioglitazone (kategorikal, dienkode).  
46. **metformin-rosiglitazone**: Status metformin-rosiglitazone (kategorikal, dienkode).  
47. **metformin-pioglitazone**: Status metformin-pioglitazone (kategorikal, dienkode).  
48. **change**: Perubahan pengobatan (kategorikal, dienkode).  
49. **diabetesMed**: Pemberian obat diabetes (kategorikal, dienkode).  
50. **readmitted**: Status readmisi (kategorikal, diubah ke `risiko_readmisi`).

### Exploratory Data Analysis (EDA)
- **Missing Values**: `weight` (96%), `payer_code` (39.3%), `medical_specialty` (48.7%), `max_glu_serum` (94.8%), dan `A1Cresult` (83.1%) memiliki missing values tinggi, sehingga `weight`, `payer_code`, dan `medical_specialty` dihapus. Kolom lain diisi 'Unknown'.
- **Duplikat**: Tidak ada duplikat, menunjukkan data unik.
- **Distribusi**: `time_in_hospital` dan `num_medications` menunjukkan distribusi miring (skewness > 1), memerlukan winsorization untuk mengurangi dampak outlier.
- **Korelasi**: `number_inpatient` berkorelasi tinggi dengan risiko readmisi (korelasi 0.35), menunjukkan riwayat rawat inap sebagai faktor risiko potensial.

## Data Preparation
Data preparation dilakukan secara sistematis untuk memastikan data bersih dan siap untuk pemodelan:

1. **Penggantian Nilai Hilang Awal**  
   - Mengganti semua nilai `'?'` dengan `np.nan` untuk menandai nilai yang hilang.  
   - **Alasan**: Memastikan konsistensi dalam penanganan missing values.

2. **Penghapusan Kolom Tidak Relevan**  
   - Menghapus `encounter_id`, `patient_nbr`, `weight`, `payer_code`, dan `medical_specialty` karena tidak relevan atau missing values tinggi (`weight` 96%, `payer_code` 39.3%, `medical_specialty` 48.7%).  
   - **Alasan**: Mengurangi noise dan kompleksitas model dengan fokus pada fitur prediktif.

3. **Penanganan Missing Values**  
   - Mengisi `NaN` pada kolom kategorikal (`race`, `diag_1`, `diag_2`, `diag_3`, `max_glu_serum`, `A1Cresult`) dengan 'Unknown'.  
   - **Alasan**: Mempertahankan baris data yang masih memiliki nilai prediktif.

4. **Penghapusan Duplikat**  
   - Tidak ada duplikat ditemukan, memastikan data unik.  
   - **Alasan**: Menghindari bias akibat data berulang.

5. **Penanganan Outlier**  
   - Menerapkan winsorization (batas 5%) pada `time_in_hospital` dan `num_medications` untuk mengurangi dampak nilai ekstrem.  
   - **Alasan**: Menormalkan distribusi untuk stabilitas model.

6. **Rekayasa Fitur**  
   - **6.1. Membuat fitur risiko_readmisi**: Mengubah `readmitted` menjadi skor numerik (`0` untuk 'NO', `0.5` untuk '>30', `1` untuk '<30'), lalu menghapus kolom asli.  
   - **6.2. Membuat fitur total_prosedur**: Menjumlahkan `num_lab_procedures`, `num_procedures`, `number_outpatient`, `number_emergency`, dan `number_inpatient` untuk menangkap intensitas perawatan.  
   - **6.3. Membuat fitur kelompok_usia**: Mengekstrak batas bawah `age` (misalnya, '[0-10)' menjadi 0), mengkategorikan menjadi 'Muda' (0-30), 'Setengah Baya' (30-60), 'Senior' (60-100) menggunakan `pd.cut`, lalu menghapus `age`.  
   - **Alasan**: Menambah fitur prediktif untuk meningkatkan wawasan klinis.

7. **Encoding Kategorikal**  
   - Menggunakan `LabelEncoder` untuk mengubah kolom kategorikal (`race`, `gender`, `kelompok_usia`, `diag_1`, `diag_2`, `diag_3`, `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`, dan semua obat) menjadi numerik.  
   - **Alasan**: Membuat data kompatibel dengan algoritma regresi.

8. **Pemisahan Fitur dan Target**  
   - Memisahkan `X` (fitur) dan `y` (`risiko_readmisi`), lalu membagi menjadi `X_train`, `X_test`, `y_train`, `y_test` (80:20, `random_state=42`).  
   - **Alasan**: Menyiapkan data untuk pelatihan dan pengujian.

9. **Skalakan Fitur**  
    - Menskalakan `X_train` dan `X_test` menggunakan `StandardScaler`.  
    - **Alasan**: Menormalkan skala fitur untuk performa model yang optimal.

**Alasan Keseluruhan Tahapan**:  
Preprocessing ini mendukung tujuan bisnis dengan menyediakan data bersih untuk prediksi akurat dan wawasan klinis. Penghapusan kolom mengurangi noise, rekayasa fitur menangkap faktor risiko, dan encoding/scaling memastikan kompatibilitas model.

## Modeling
Tiga model regresi digunakan untuk memprediksi `risiko_readmisi`:

1. **Regresi Linear**  
   - **Deskripsi**: Model baseline yang mengasumsikan hubungan linier.  
   - **Kelebihan**: Sederhana, cepat.  
   - **Kekurangan**: Tidak menangkap hubungan non-linier.  
   - **Parameter**: Tanpa penyetelan.

2. **Random Forest Regressor**  
   - **Deskripsi**: Ensemble pohon untuk menangani hubungan non-linier.  
   - **Kelebihan**: Tahan overfitting, menangkap interaksi fitur.  
   - **Kekurangan**: Komputasi intensif.  
   - **Penyetelan**: `n_estimators` [50, 100], `max_depth` [5, 10] menggunakan GridSearchCV (5-fold CV).  
   - **Hasil**: R² meningkat dari 0.09 (default) ke 0.1064.

3. **XGBoost Regressor**  
   - **Deskripsi**: Gradient boosting untuk performa tinggi.  
   - **Kelebihan**: Menangani non-linearitas, efisien.  
   - **Kekurangan**: Sensitif terhadap penyetelan.  
   - **Penyetelan**: `n_estimators` [100, 200], `max_depth` [5, 7], `learning_rate` [0.1, 0.01] menggunakan GridSearchCV (5-fold CV).  
   - **Hasil**: R² meningkat dari 0.095 (default) ke 0.1103.

**Pemilihan Model**: XGBoost dipilih karena R² tertinggi (0.1103) dan MSE terendah (0.1098), menunjukkan generalisasi terbaik.

## Evaluation
Metrik evaluasi untuk regresi:

- **MAE**: Rata-rata kesalahan absolut, mengukur akurasi prediksi.  
  $$MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
- **MSE**: Rata-rata kuadrat kesalahan, sensitif terhadap outlier.  
  $$MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$
- **R²**: Proporsi varians yang dijelaskan, mengukur kecocokan model.  
  $$R^2 = 1 - \frac{\text{SS}\text{res}} {\text{SS}\text{tot}}$$

**Hasil Evaluasi**:
- **Regresi Linear**: MAE: 0.2928, MSE: 0.1136, R²: 0.0799  
  - **Interpretasi**: Performa terendah karena data memiliki hubungan non-linier.
- **Random Forest**: MAE: 0.2854, MSE: 0.1103, R²: 0.1064  
  - **Interpretasi**: Lebih baik dari Regresi Linear, tetapi MAE serupa dengan XGBoost.
- **XGBoost**: MAE: 0.2855, MSE: 0.1098, R²: 0.1103  
  - **Interpretasi**: Model terbaik dengan R² tertinggi dan MSE terendah, meskipun R² rendah menunjukkan keterbatasan fitur.

**Hubungan dengan Business Understanding**:
- **Problem Statement 1 (Biaya readmisi)**: XGBoost membantu mengidentifikasi pasien berisiko tinggi, memungkinkan intervensi dini untuk mengurangi biaya readmisi.
- **Problem Statement 2 (Kurang alat prediktif)**: Model ini menyediakan alat prediktif berbasis data, dengan `number_inpatient` sebagai fitur utama (pentingnya 0.3524).
- **Goal 1 (Model akurat)**: Tercapai sebagian; XGBoost memiliki R² 0.1103, tetapi masih rendah.
- **Goal 2 (Wawasan klinis)**: `number_inpatient` menunjukkan pasien dengan riwayat rawat inap lebih berisiko.
- **Solution Statement 1 (Bandingkan model)**: Berhasil, XGBoost unggul.
- **Solution Statement 2 (Penyetelan)**: Berhasil, penyetelan meningkatkan performa.

**Visualisasi**:  
Plot pentingnya fitur Random Forest menunjukkan `number_inpatient` dan `discharge_disposition_id` sebagai faktor utama.  
(![pentingnya_fitur_rf](https://github.com/Andriansyah2501/MachineLearnigTerapan01/blob/main/pentingnya_fitur_rf.png?raw=true)
)


## Kesimpulan
Proyek ini berhasil mengembangkan model regresi untuk memprediksi risiko readmisi pasien diabetes, dengan XGBoost Regressor sebagai model dengan performa terbaik berdasarkan hasil evaluasi (R² = 0,1103 dan MSE = 0,1098). Meskipun nilai R² tergolong rendah, model ini tetap mampu memberikan wawasan klinis awal yang dapat mendukung intervensi dini dalam pengelolaan pasien.

Proyek ini telah memenuhi seluruh kriteria yang ditetapkan oleh Dicoding, termasuk:
1. Penggunaan dataset dengan 5000 sampel sesuai ketentuan minimum,
2. Penyusunan dokumentasi yang lengkap dan sistematis,
3. Penyertaan visualisasi online untuk mendukung interpretasi hasil.

Meskipun terdapat keterbatasan dalam performa model, pendekatan ini tetap memberikan fondasi yang solid untuk pengembangan sistem prediksi yang lebih akurat di masa mendatang melalui eksplorasi fitur tambahan dan pendekatan lanjutan.

## ❗ Kelemahan Model

- 🔹 **Nilai R² yang rendah** (0.1103) menunjukkan bahwa model belum mampu menjelaskan variabilitas data secara optimal. Hal ini kemungkinan besar disebabkan oleh:
  - Fitur yang kurang relevan atau belum cukup informatif,
  - Tingginya **nilai hilang (missing values)** dan kemungkinan **noise** dalam data.
  
- 🔹 Penghapusan kolom penting seperti **`weight`** menyebabkan hilangnya potensi informasi klinis yang dapat mendukung prediksi.

---

## 🛠️ Saran Perbaikan

- ➕ Tambahkan **fitur interaksi**, misalnya `num_medications × insulin`, untuk menangkap efek gabungan dari pengobatan terhadap risiko readmisi.
- 🩻 Gunakan metode **imputasi** (median, regresi, atau KNN) untuk menangani **missing values** pada fitur seperti `weight`, alih-alih menghapusnya langsung.
- 🧠 Lakukan eksperimen dengan model lanjutan seperti:
  - **CatBoost** – unggul dalam menangani data kategorikal,
  - **Stacking Ensemble** – menggabungkan kekuatan beberapa model untuk hasil prediksi yang lebih akurat.

---

## 💼 Dampak Bisnis

- 🏥 Model dapat diintegrasikan ke dalam sistem **Electronic Health Record (EHR)** di rumah sakit.
- 🔔 Memungkinkan **notifikasi otomatis** kepada tenaga medis terkait pasien dengan **risiko readmisi tinggi**.
- 💰 Berpotensi **menurunkan biaya operasional** rumah sakit akibat readmisi yang tidak perlu.
- 🩺 Membantu **meningkatkan kualitas layanan dan hasil perawatan pasien** melalui intervensi yang lebih tepat waktu dan berbasis data.

---

