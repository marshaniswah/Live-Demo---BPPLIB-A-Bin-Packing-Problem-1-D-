# 📦 Live Interactive Demo: 1D Bin Packing Problem (1D-BPP)
**Course:** AI5103 Fundamental of AI & Data Science — Problem Solving Agent  
**Authors:** Marshaniswah Syamsul, Hasna Putri Priswati  
**Algorithms Compared:** First Fit Decreasing (FFD) vs. Genetic Algorithm (GA) vs. Particle Swarm Optimization (PSO)

---

## 🎬 Live Demo Preview
![Live Demo](./Live%20Demo%20.gif)

---

## 🌟 Fitur Baru Aplikasi Demo Minimalis & User-Friendly
1. **🎯 Step-by-Step Interactive Visualizer (Setiap Algoritma)**:
   - **First Fit Decreasing (FFD)**: Menampilkan penempatan item satu per satu, jejak evaluasi kelayakan tiap bin, hingga pembukaan bin baru lengkap dengan *highlight* item yang baru masuk.
   - **Genetic Algorithm (GA)**: Menampilkan evolusi generasi demi generasi, rekaman populasi, dan transformasi layout susunan bin dari acak hingga optimum.
   - **Particle Swarm Optimization (PSO)**: Menampilkan pergerakan partikel per iterasi, pembaruan posisi kontinu, decoding *argsort*, dan evolusi solusi $g_{best}$.
2. **⚙️ Kontrol Parameter Interaktif per Algoritma**:
   - **FFD**: Pilihan strategi pengurutan (*Decreasing / FFD*, *Increasing / FFI*, *Pure FF / Asli*, *Random*).
   - **GA**: Ukuran populasi ($N$), jumlah generasi, probabilitas crossover ($p_c$), mutasi ($p_m$), *tournament size* ($k$), dan opsi elitisme.
   - **PSO**: Ukuran *swarm* ($N$), jumlah iterasi, *inertia weight* ($w$), bobot kognitif ($c_1$), bobot sosial ($c_2$), dan batas kecepatan ($v_{max}$).
3. **🌟 Ragam Skenario & Dataset**:
   - **Mini Demo (15 Item - Cepat & Jelas)**: Skenario ringkas instan yang pas dalam satu layar tanpa perlu scroll.
   - **Medium Demo (30 Item)**: Skenario menengah representatif.
   - **Benchmark Falkenauer U120**: 5 instans standar BPPLIB ($n=120, C=150$).
   - **Custom / Generator Acak**: Masukkan bobot manual atau buat problem acak sendiri.
4. **⚖️ Komparasi Head-to-Head (FFD vs GA vs PSO)**:
   - Evaluasi komparatif adil di bawah *budget* fungsi objektif yang sama.
   - Kurva konvergensi Plotly interaktif bersama garis BKS Optimum dan Baseline FFD.
   - Grafik batang perbandingan jumlah bin dan efisiensi waktu komputasi.
5. **💡 Panduan & Integritas Matematika**:
   - Formulasi formal Integer Linear Programming (ILP).
   - Penjelasan matematis batas bawah kontinu ($L_1$) dan teorema aproksimasi FFD ($\le \frac{11}{9}\text{OPT} + 1$).
   - Audit kelayakan solusi otomatis ($P_1, P_2, P_3$).

---

## 🚀 How to Run Locally

```bash
# 1. Masuk ke direktori aplikasi
cd bpp_streamlit_demo/bpp_streamlit_demo

# 2. Install dependensi jika belum
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.
