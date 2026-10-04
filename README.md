# Klasifikasi Status Meteorit (*Fell* vs *Found*) dengan TabTransformer dan FT-Transformer

Eksperimen klasifikasi biner pada dataset **Meteorite Landings** (The Meteoritical Society, melalui NASA Open Data Portal) untuk memprediksi apakah jatuhnya sebuah meteorit **teramati** (*Fell*) atau **ditemukan belakangan** (*Found*). Proyek ini membandingkan model *Transformer* untuk data tabular dengan baseline klasik pada lima pembagian data acak.

> **Status:** eksperimen untuk penulisan jurnal. Hasil di bawah adalah rata-rata dari 5 seed, bukan satu kali percobaan.

## Ringkasan

- **Tugas:** klasifikasi biner, kelas positif = *Fell* (sekitar 2,6% data, sangat tidak seimbang).
- **Model utama:** TabTransformer dan FT-Transformer (implementasi PyTorch sendiri, bukan dari library).
- **Baseline:** *Logistic Regression*, *Random Forest*, *HistGradientBoosting*.
- **Temuan singkat:** FT-Transformer lebih baik daripada TabTransformer, tetapi model berbasis pohon (*Random Forest*) tetap menjadi yang terkuat pada data ini.

## Dataset

| Item | Keterangan |
|---|---|
| Sumber | The Meteoritical Society, melalui [NASA Open Data Portal](https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh) |
| File | `meteorite-landings.csv` |
| Ukuran awal | 45.716 baris, 10 kolom |
| Setelah pembersihan | sekitar 43 ribu baris |
| Target | `fall` (*Fell* = 1, *Found* = 0) |
| Fitur dasar | `mass`, `year`, `reclat`, `reclong` (numerik); `recclass`, `nametype` (kategorikal) |

## Alur Pemrosesan

1. **Pembersihan:** hanya baris `Fell`/`Found`, konversi numerik, nilai kategorikal kosong menjadi `Unknown`, hapus duplikat.
2. **Rekayasa fitur:**
   - indikator data hilang: `mass_missing`, `year_missing`, `coord_missing`
   - koordinat (0, 0) dianggap placeholder dan diubah menjadi `NaN`
   - `mass` ditransformasi `log1p`
   - fitur lokasi turunan: `abs_lat`, `is_antarctica`
   - `recclass` yang jarang (< 20 sampel) digabung menjadi `Other`
3. **Pembagian data:** 80% latih / 20% uji (stratified), lalu 15% dari data latih menjadi data validasi.
4. **Praproses:** imputasi median + `StandardScaler` untuk numerik, encoding indeks untuk kategorikal (`0` = tidak dikenal). Semua di-*fit* hanya pada data latih.
5. **Pelatihan:** `BCEWithLogitsLoss` dengan `pos_weight = sqrt(neg/pos)`, Adam, `ReduceLROnPlateau`, *gradient clipping*, *early stopping* pada *validation loss*.
6. **Ambang klasifikasi:** dipilih dari **data validasi** dengan memaksimalkan nilai terkecil antara presisi dan *recall*. Data uji hanya dipakai sekali untuk pelaporan.
7. **Pengulangan:** seed `42, 7, 13, 21, 99`. Setiap seed mengulang pembagian data, pelatihan, pemilihan ambang, dan evaluasi.

## Hasil (kelas *Fell*, rata-rata ± simpangan baku, 5 seed)

| Model | Presisi | *Recall* | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,555 ± 0,019 | 0,547 ± 0,016 | 0,551 ± 0,015 | 0,979 ± 0,003 | 0,573 ± 0,024 |
| TabTransformer | 0,730 ± 0,047 | 0,719 ± 0,026 | 0,724 ± 0,034 | 0,987 ± 0,003 | 0,761 ± 0,048 |
| FT-Transformer | 0,746 ± 0,042 | 0,765 ± 0,034 | 0,755 ± 0,027 | 0,990 ± 0,002 | 0,779 ± 0,026 |
| HistGradientBoosting | 0,776 ± 0,027 | 0,767 ± 0,031 | 0,771 ± 0,022 | 0,993 ± 0,002 | 0,835 ± 0,033 |
| Random Forest | 0,800 ± 0,018 | 0,783 ± 0,031 | 0,791 ± 0,012 | 0,994 ± 0,002 | 0,847 ± 0,020 |

Catatan membaca hasil:

- **Akurasi tidak dilaporkan** karena menyesatkan pada data dengan sekitar 97% kelas *Found*. Fokus ke presisi, *recall*, F1, dan PR-AUC kelas *Fell*.
- ROC-AUC kurang membedakan model (semua ≥ 0,979). PR-AUC lebih informatif.
- Simpangan baku dihitung dari 5 pembagian acak yang saling tumpang tindih, jadi cenderung lebih optimistis daripada validasi silang berlapis.

## Struktur Repositori

```
.
├── NASA_Meteorite_TabTransformer.ipynb   # notebook utama (pipeline, model, multi-seed, FT-Transformer)
├── meteorite-landings.csv                # dataset
├── tabtransformer_nasa_meteorite.pth     # model TabTransformer contoh (satu run) + praproses + threshold
└── README.md
```

> Model yang tersimpan adalah **contoh satu run**, bukan "model final" dari angka rata-rata di atas.

## Cara Menjalankan

Python 3.10.

```bash
# Dependensi umum
pip install numpy pandas scikit-learn matplotlib seaborn ipykernel

# PyTorch (GPU NVIDIA RTX 50-series / Blackwell butuh build CUDA 12.8+; contoh CUDA 13.0)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Atau versi CPU
# pip install torch
```

Cek GPU:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Lalu buka `NASA_Meteorite_TabTransformer.ipynb` dan jalankan semua sel secara berurutan (**Restart → Run All**). Eksperimen multi-seed (5 seed × 4 model) memakan waktu belasan menit pada GPU, dan lebih lama pada CPU.

Memuat model tersimpan (berisi objek scikit-learn, sehingga butuh `weights_only=False`; **hanya untuk file buatan sendiri**):

```python
ckpt = torch.load("tabtransformer_nasa_meteorite.pth", weights_only=False)
best_thr = ckpt["best_thr"]
```

## Keterbatasan

- Hiperparameter bawaan atau ditetapkan satu kali, belum ada pencarian sistematis.
- Hanya ±221 sampel *Fell* pada data uji, sehingga selisih beberapa prediksi dapat menggeser metrik beberapa poin persentase.
- Label *Fell/Found* mencerminkan cara meteorit ditemukan. Fitur lokasi dan tahun dapat menangkap pola program pencarian (misalnya ekspedisi di Antartika), bukan hanya sifat fisik meteorit.
- Hanya ada dua fitur kategorikal, sehingga *attention* pada TabTransformer bekerja pada dua token saja.
- Kebaruan (belum ada penelitian terindeks yang memakai *tabular Transformer* pada dataset ini) baru dicek lewat pencarian web umum, belum lewat Scopus/Web of Science.

## Rencana

- [ ] Pencarian hiperparameter sistematis untuk semua model
- [ ] Validasi silang berlapis dan uji signifikansi antar model
- [ ] Ablasi terkontrol untuk fitur turunan
- [ ] Model *Transformer* tabular lain (misalnya SAINT)

## Referensi

- Gorishniy, Rubachev, Khrulkov, Babenko. *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer), NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Huang, Khetan, Cvitkovic, Karnin. *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*, 2020. [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
- The Meteoritical Society. *Meteorite Landings*, NASA Open Data Portal.

## Penulis

`[Nama]`, Informatika, Institut Teknologi Nasional (ITN) Malang.

## Lisensi

`[Pilih lisensi, mis. MIT, dan sesuaikan dengan lisensi dataset]`
