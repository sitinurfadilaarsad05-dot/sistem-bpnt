import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Kelayakan BPNT",
    page_icon="🏛️",
    layout="wide"
)

# ============================================================
# CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .header-box {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white; padding: 30px 20px; border-radius: 16px;
        text-align: center; margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(21,101,192,0.3);
    }
    .header-box h1 { font-size: 24px; font-weight: 800; margin: 0 0 6px 0; }
    .header-box p { font-size: 13px; opacity: 0.85; margin: 2px 0; }

    .card {
        background: white; padding: 24px; border-radius: 16px;
        border: 1px solid #e0e8f0; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 15px; font-weight: 700; color: #1565C0;
        margin-bottom: 16px; padding-bottom: 8px;
        border-bottom: 2px solid #E3F2FD;
    }

    .metric-box {
        background: linear-gradient(135deg, #E3F2FD, #EFF8FF);
        border: 1px solid #90CAF9; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .metric-box h2 { font-size: 32px; font-weight: 800; color: #1565C0; margin: 0; }
    .metric-box p { font-size: 13px; color: #555; margin: 4px 0 0 0; }

    .metric-green {
        background: linear-gradient(135deg, #E8F5E9, #F1F8F1);
        border: 1px solid #A5D6A7;
    }
    .metric-green h2 { color: #2E7D32; }

    .metric-red {
        background: linear-gradient(135deg, #FFEBEE, #FFF5F5);
        border: 1px solid #EF9A9A;
    }
    .metric-red h2 { color: #C62828; }

    .akurasi-box {
        background: linear-gradient(135deg, #E3F2FD, #EFF8FF);
        border: 2px solid #1976D2; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .akurasi-best {
        background: linear-gradient(135deg, #E8F5E9, #F1F8F1);
        border: 2px solid #4CAF50; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .akurasi-k { font-size: 16px; font-weight: 700; color: #1565C0; }
    .akurasi-val { font-size: 34px; font-weight: 800; color: #1565C0; }
    .akurasi-best .akurasi-k { color: #2E7D32; }
    .akurasi-best .akurasi-val { color: #2E7D32; }

    .result-layak {
        background: linear-gradient(135deg, #E8F5E9, #F1F8F1);
        border: 2px solid #4CAF50; border-radius: 16px;
        padding: 28px; text-align: center; margin-top: 10px;
    }
    .result-tidak {
        background: linear-gradient(135deg, #FFEBEE, #FFF5F5);
        border: 2px solid #EF5350; border-radius: 16px;
        padding: 28px; text-align: center; margin-top: 10px;
    }
    .result-icon { font-size: 52px; margin-bottom: 8px; }
    .result-name { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
    .result-status-layak { font-size: 28px; font-weight: 800; color: #2E7D32; margin-bottom: 8px; }
    .result-status-tidak { font-size: 28px; font-weight: 800; color: #C62828; margin-bottom: 8px; }
    .result-desc { font-size: 14px; opacity: 0.75; }

    .info-box {
        background: #FFF8E1; border: 1px solid #FFD54F;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
        font-size: 13px; color: #5D4037;
    }

    .footer {
        text-align: center; font-size: 12px; color: #999;
        margin-top: 30px; padding-top: 16px;
        border-top: 1px solid #e0e0e0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white; font-weight: 700; font-size: 15px;
        padding: 12px 32px; border-radius: 10px; border: none;
        width: 100%; cursor: pointer; letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0D47A1, #1565C0);
        box-shadow: 0 4px 15px rgba(21,101,192,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA & PREPROCESSING
# ============================================================
@st.cache_resource
def load_data():
    try:
        df_train = pd.read_excel('DATA FIX SKRIPSI KNN.xlsx',
                                  sheet_name='data sampel knn', header=4)
        df_train = df_train.dropna(how='all').iloc[:, 1:]
        df_train.columns = ['No','Nama','Pendapatan','Tanggungan',
                            'Pekerjaan','Rumah','Aset','Status','Label']
        df_train = df_train[df_train['No'].notna()].reset_index(drop=True)

        df_test = pd.read_excel('DATA FIX SKRIPSI KNN.xlsx',
                                 sheet_name='data uji knn', header=4)
        df_test = df_test.dropna(how='all').iloc[:, 1:]
        df_test.columns = ['No','Nama','Pendapatan','Tanggungan',
                           'Pekerjaan','Rumah','Aset','Status','Label']
        df_test = df_test[df_test['No'].notna()].reset_index(drop=True)

        def bersihkan(nilai):
            nilai = str(nilai).strip().replace('Rp','').replace(',','').replace('.','')
            try:
                angka = float(nilai)
                if angka < 10000: angka *= 1000
                return angka
            except:
                return np.nan

        df_train['Pendapatan'] = df_train['Pendapatan'].apply(bersihkan)
        df_test['Pendapatan']  = df_test['Pendapatan'].apply(bersihkan)

        kolom_fitur    = ['Pendapatan','Tanggungan','Pekerjaan','Rumah','Aset','Status']
        kolom_kategori = ['Pekerjaan','Rumah','Aset','Status']

        X_train = df_train[kolom_fitur].copy()
        y_train = df_train['Label'].copy()
        X_test  = df_test[kolom_fitur].copy()
        y_test  = df_test['Label'].copy()

        X_all = pd.concat([X_train, X_test], ignore_index=True)
        le_dict = {}
        for k in kolom_kategori:
            le = LabelEncoder()
            X_all[k] = le.fit_transform(X_all[k].astype(str))
            le_dict[k] = le

        X_train_enc = X_all.iloc[:len(X_train)].copy().reset_index(drop=True)
        X_test_enc  = X_all.iloc[len(X_train):].copy().reset_index(drop=True)

        le_label = LabelEncoder()
        semua = pd.concat([y_train, y_test], ignore_index=True)
        le_label.fit(semua.astype(str))
        y_train_enc = le_label.transform(y_train.astype(str))
        y_test_enc  = le_label.transform(y_test.astype(str))

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_enc)
        X_test_scaled  = scaler.transform(X_test_enc)

        return (X_train_scaled, y_train_enc, X_test_scaled, y_test_enc,
                scaler, le_dict, le_label, kolom_fitur, kolom_kategori,
                df_train, df_test, True)

    except Exception as e:
        return (None,)*11 + (str(e),)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>🏛️ SISTEM KLASIFIKASI KELAYAKAN PENERIMA BPNT</h1>
    <p>Menggunakan Algoritma K-Nearest Neighbor (KNN)</p>
    <p>Desa Paris, Kecamatan Mootilango</p>
</div>
""", unsafe_allow_html=True)

result = load_data()
(X_train_scaled, y_train_enc, X_test_scaled, y_test_enc,
 scaler, le_dict, le_label, kolom_fitur, kolom_kategori,
 df_train, df_test, status) = result

if status != True:
    st.error(f"❌ Gagal memuat data: {status}\n\nPastikan file **'DATA FIX SKRIPSI KNN.xlsx'** ada di folder yang sama!")
    st.stop()


# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Beranda",
    "📋 Data Latih",
    "📋 Data Uji",
    "📊 Hasil Pengujian K=3,5,7",
    "🔍 Prediksi Data Baru"
])


# ============================================================
# TAB 1 - BERANDA
# ============================================================
with tab1:
    st.markdown("## 👋 Selamat Datang di Sistem Klasifikasi Kelayakan BPNT")
    st.markdown("""
    Sistem ini menggunakan algoritma **K-Nearest Neighbor (KNN)** untuk mengklasifikasikan 
    kelayakan masyarakat sebagai penerima bantuan **BPNT (Bantuan Pangan Non Tunai)** 
    di Desa Paris, Kecamatan Mootilango.
    """)

    st.markdown("---")

    # Statistik umum
    jumlah_train = len(df_train)
    jumlah_test  = len(df_test)
    layak_train  = len(df_train[df_train['Label'] == 'Layak'])
    tidak_train  = len(df_train[df_train['Label'] == 'Tidak Layak'])
    layak_test   = len(df_test[df_test['Label'] == 'Layak'])
    tidak_test   = len(df_test[df_test['Label'] == 'Tidak Layak'])

    st.markdown("### 📊 Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><h2>{jumlah_train}</h2><p>Total Data Latih</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h2>{jumlah_test}</h2><p>Total Data Uji</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box metric-green"><h2>{layak_train + layak_test}</h2><p>Total Layak</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box metric-red"><h2>{tidak_train + tidak_test}</h2><p>Total Tidak Layak</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Fitur yang Digunakan")
    fitur_data = {
        "No": [1, 2, 3, 4, 5, 6],
        "Fitur": ["Pendapatan", "Jumlah Tanggungan", "Pekerjaan", "Kondisi Rumah", "Kepemilikan Aset", "Status Kepemilikan"],
        "Tipe": ["Numerik", "Numerik", "Kategorik", "Kategorik", "Kategorik", "Kategorik"],
        "Keterangan": [
            "Pendapatan per bulan (Rp)",
            "Jumlah anggota keluarga yang ditanggung",
            "Buruh/Tani, Wiraswasta, PNS/Tetap",
            "Bambu/Tanah, Semi Perm., Tembok/Keramik",
            "Tidak Ada, Motor, Mobil",
            "Menumpang, Milik Keluarga, Milik Sendiri"
        ]
    }
    st.dataframe(pd.DataFrame(fitur_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔄 Alur Sistem")
    st.markdown("""
    1. **Input Data** → Data masyarakat dimasukkan ke sistem
    2. **Preprocessing** → Data dibersihkan, encoding kategori, normalisasi MinMax
    3. **Training KNN** → Model dilatih menggunakan data latih
    4. **Klasifikasi** → Data diuji menggunakan algoritma KNN
    5. **Output** → Hasil klasifikasi: **Layak** atau **Tidak Layak**
    """)


# ============================================================
# TAB 2 - DATA LATIH
# ============================================================
with tab2:
    st.markdown("### 📋 Data Latih (Training)")
    st.markdown(f"Total data latih: **{len(df_train)} data**")

    col1, col2 = st.columns(2)
    with col1:
        layak = len(df_train[df_train['Label'] == 'Layak'])
        st.markdown(f'<div class="metric-box metric-green"><h2>{layak}</h2><p>Layak</p></div>', unsafe_allow_html=True)
    with col2:
        tidak = len(df_train[df_train['Label'] == 'Tidak Layak'])
        st.markdown(f'<div class="metric-box metric-red"><h2>{tidak}</h2><p>Tidak Layak</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tampilkan tabel dengan warna label
    df_train_tampil = df_train[['No','Nama','Pendapatan','Tanggungan',
                                 'Pekerjaan','Rumah','Aset','Status','Label']].copy()
    df_train_tampil['Pendapatan'] = df_train_tampil['Pendapatan'].apply(
        lambda x: f"Rp {int(x):,}".replace(',', '.') if pd.notna(x) else '-')
    df_train_tampil['No'] = range(1, len(df_train_tampil)+1)

    def warna_label(val):
        if val == 'Layak':
            return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
        elif val == 'Tidak Layak':
            return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
        return ''

    styled = df_train_tampil.style.map(warna_label, subset=['Label'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    > 🟢 **Layak** = Berhak menerima bantuan BPNT &nbsp;&nbsp; 🔴 **Tidak Layak** = Tidak berhak menerima bantuan BPNT
    """)


# ============================================================
# TAB 3 - DATA UJI
# ============================================================
with tab3:
    st.markdown("### 📋 Data Uji (Testing)")
    st.markdown(f"Total data uji: **{len(df_test)} data**")

    col1, col2 = st.columns(2)
    with col1:
        layak = len(df_test[df_test['Label'] == 'Layak'])
        st.markdown(f'<div class="metric-box metric-green"><h2>{layak}</h2><p>Layak</p></div>', unsafe_allow_html=True)
    with col2:
        tidak = len(df_test[df_test['Label'] == 'Tidak Layak'])
        st.markdown(f'<div class="metric-box metric-red"><h2>{tidak}</h2><p>Tidak Layak</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    df_test_tampil = df_test[['No','Nama','Pendapatan','Tanggungan',
                               'Pekerjaan','Rumah','Aset','Status','Label']].copy()
    df_test_tampil['Pendapatan'] = df_test_tampil['Pendapatan'].apply(
        lambda x: f"Rp {int(x):,}".replace(',', '.') if pd.notna(x) else '-')
    df_test_tampil['No'] = range(1, len(df_test_tampil)+1)

    styled2 = df_test_tampil.style.map(warna_label, subset=['Label'])
    st.dataframe(styled2, use_container_width=True, hide_index=True)

    st.markdown("""
    > 🟢 **Layak** = Berhak menerima bantuan BPNT &nbsp;&nbsp; 🔴 **Tidak Layak** = Tidak berhak menerima bantuan BPNT
    """)


# ============================================================
# TAB 4 - HASIL PENGUJIAN K=3,5,7
# ============================================================
with tab4:
    st.markdown("### 📊 Perbandingan Hasil Pengujian K=3, K=5, K=7")
    st.markdown("""
    Pengujian dilakukan terhadap seluruh **data uji** menggunakan 3 nilai K yang berbeda 
    untuk mencari nilai K terbaik berdasarkan akurasi tertinggi.
    """)

    hasil_semua   = {}
    akurasi_semua = {}
    cm_semua      = {}

    for k in [3, 5, 7]:
        knn_k = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn_k.fit(X_train_scaled, y_train_enc)
        pred_enc   = knn_k.predict(X_test_scaled)
        pred_label = le_label.inverse_transform(pred_enc)
        akurasi    = accuracy_score(y_test_enc, pred_enc) * 100
        cm         = confusion_matrix(y_test_enc, pred_enc)
        hasil_semua[k]   = pred_label
        akurasi_semua[k] = akurasi
        cm_semua[k]      = cm

    best_k = max(akurasi_semua, key=akurasi_semua.get)

    # Kartu akurasi
    st.markdown("#### 🎯 Akurasi per Nilai K")
    col_k1, col_k2, col_k3 = st.columns(3)
    for col, k in zip([col_k1, col_k2, col_k3], [3, 5, 7]):
        with col:
            css   = "akurasi-best" if k == best_k else "akurasi-box"
            badge = " 🏆 Terbaik" if k == best_k else ""
            st.markdown(f"""
            <div class="{css}">
                <div class="akurasi-k">K = {k}{badge}</div>
                <div class="akurasi-val">{akurasi_semua[k]:.1f}%</div>
                <div style="font-size:12px;color:#666;margin-top:4px;">Akurasi</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabel perbandingan prediksi
    st.markdown("#### 📋 Tabel Perbandingan Hasil Prediksi")
    st.markdown("""
    <div class="info-box">
        ℹ️ <b>Keterangan:</b> Kolom <b>Label Asli</b> adalah label yang sudah ada di data uji. 
        Kolom <b>Prediksi K=3/5/7</b> adalah hasil klasifikasi oleh algoritma KNN. 
        🟢 Hijau = prediksi benar, 🔴 Merah = prediksi salah.
    </div>
    """, unsafe_allow_html=True)

    df_hasil = df_test[['Nama','Label']].copy()
    df_hasil.columns = ['Nama', 'Label Asli']
    df_hasil['Prediksi K=3'] = hasil_semua[3]
    df_hasil['Prediksi K=5'] = hasil_semua[5]
    df_hasil['Prediksi K=7'] = hasil_semua[7]
    df_hasil.insert(0, 'No', range(1, len(df_hasil)+1))

    def style_prediksi(row):
        asli   = row['Label Asli']
        styles = ['', '']  # No, Nama
        # Label Asli
        if asli == 'Layak':
            styles.append('background-color: #E8F5E9; color: #2E7D32; font-weight: bold')
        else:
            styles.append('background-color: #FFEBEE; color: #C62828; font-weight: bold')
        for col in ['Prediksi K=3', 'Prediksi K=5', 'Prediksi K=7']:
            if row[col] == asli:
                styles.append('background-color: #E8F5E9; color: #2E7D32; font-weight: bold')
            else:
                styles.append('background-color: #FFEBEE; color: #C62828; font-weight: bold')
        return styles

    styled3 = df_hasil.style.apply(style_prediksi, axis=1)
    st.dataframe(styled3, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion Matrix
    st.markdown("#### 🔢 Confusion Matrix per Nilai K")
    st.markdown("""
    <div class="info-box">
        ℹ️ <b>Confusion Matrix</b> menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.
        <br>• <b>TP (True Positive)</b>: Prediksi Layak, Aslinya Layak ✅
        <br>• <b>TN (True Negative)</b>: Prediksi Tidak Layak, Aslinya Tidak Layak ✅
        <br>• <b>FP (False Positive)</b>: Prediksi Layak, Aslinya Tidak Layak ❌
        <br>• <b>FN (False Negative)</b>: Prediksi Tidak Layak, Aslinya Layak ❌
    </div>
    """, unsafe_allow_html=True)

    classes = le_label.classes_
    for k in [3, 5, 7]:
        badge = " 🏆 (Terbaik)" if k == best_k else ""
        st.markdown(f"**K = {k}{badge} — Akurasi: {akurasi_semua[k]:.1f}%**")
        cm = cm_semua[k]
        df_cm = pd.DataFrame(cm,
                             index=[f"Asli: {c}" for c in classes],
                             columns=[f"Prediksi: {c}" for c in classes])
        st.dataframe(df_cm, use_container_width=True)
        st.markdown("")

    st.markdown("---")

    # Ringkasan
    st.markdown(f"### 🏆 Kesimpulan")
    st.success(f"""
    Berdasarkan hasil pengujian dengan nilai K=3, K=5, dan K=7, diperoleh bahwa:
    - **K=3** → Akurasi: **{akurasi_semua[3]:.1f}%**
    - **K=5** → Akurasi: **{akurasi_semua[5]:.1f}%**
    - **K=7** → Akurasi: **{akurasi_semua[7]:.1f}%**

    **Nilai K terbaik adalah K={best_k}** dengan akurasi tertinggi sebesar **{akurasi_semua[best_k]:.1f}%**.
    """)


# ============================================================
# TAB 5 - PREDIKSI DATA BARU
# ============================================================
with tab5:
    st.markdown("### 🔍 Prediksi Kelayakan Masyarakat Baru")
    st.markdown("""
    <div class="info-box">
        ℹ️ Masukkan data masyarakat yang ingin diklasifikasikan, pilih nilai K, 
        lalu klik <b>Prediksi Kelayakan</b>.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_hasil = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Data Masyarakat</div>', unsafe_allow_html=True)

        nama       = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")
        pendapatan = st.number_input("Pendapatan (Rp)", min_value=0, step=50000, help="Contoh: 800000")
        tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, step=1, help="Contoh: 4")
        pekerjaan  = st.selectbox("Pekerjaan", ["Buruh/Tani", "Wiraswasta", "PNS/Tetap"])
        rumah      = st.selectbox("Kondisi Rumah", ["Bambu/Tanah", "Semi Perm.", "Tembok/Keramik"])
        aset       = st.selectbox("Kepemilikan Aset", ["Tidak Ada", "Motor", "Mobil"])
        status_kep = st.selectbox("Status Kepemilikan", ["Menumpang", "Milik Keluarga", "Milik Sendiri"])

        st.markdown("---")
        k_val = st.radio("Nilai K (Tetangga Terdekat):", [3, 5, 7], horizontal=True, index=0)

        col_b1, col_b2 = st.columns([2,1])
        with col_b1:
            prediksi_btn = st.button("🔍 PREDIKSI KELAYAKAN", use_container_width=True)
        with col_b2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_hasil:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Hasil Klasifikasi</div>', unsafe_allow_html=True)

        if reset_btn:
            st.rerun()

        if prediksi_btn:
            if not nama.strip():
                st.warning("⚠️ Nama tidak boleh kosong!")
            elif pendapatan <= 0:
                st.warning("⚠️ Pendapatan harus lebih dari 0!")
            else:
                data_input = pd.DataFrame([{
                    'Pendapatan': float(pendapatan),
                    'Tanggungan': int(tanggungan),
                    'Pekerjaan' : pekerjaan,
                    'Rumah'     : rumah,
                    'Aset'      : aset,
                    'Status'    : status_kep
                }])

                for k in kolom_kategori:
                    data_input[k] = le_dict[k].transform(data_input[k].astype(str))

                data_scaled = scaler.transform(data_input)
                knn = KNeighborsClassifier(n_neighbors=k_val, metric='euclidean')
                knn.fit(X_train_scaled, y_train_enc)
                hasil_enc   = knn.predict(data_scaled)
                hasil_label = le_label.inverse_transform(hasil_enc)[0]

                st.markdown(f"**Menggunakan K = {k_val}**")

                if hasil_label == 'Layak':
                    st.markdown(f"""
                    <div class="result-layak">
                        <div class="result-icon">✅</div>
                        <div class="result-name">{nama}</div>
                        <div class="result-status-layak">LAYAK</div>
                        <div class="result-desc">Masyarakat ini <strong>LAYAK</strong><br>menerima bantuan BPNT</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-tidak">
                        <div class="result-icon">❌</div>
                        <div class="result-name">{nama}</div>
                        <div class="result-status-tidak">TIDAK LAYAK</div>
                        <div class="result-desc">Masyarakat ini <strong>TIDAK LAYAK</strong><br>menerima bantuan BPNT</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Detail input
                st.markdown("---")
                st.markdown("**📌 Detail Data yang Diinput:**")
                detail = {
                    "Keterangan": ["Pendapatan", "Tanggungan", "Pekerjaan", "Kondisi Rumah", "Kepemilikan Aset", "Status Kepemilikan"],
                    "Nilai": [
                        f"Rp {int(pendapatan):,}".replace(',', '.'),
                        f"{tanggungan} orang",
                        pekerjaan, rumah, aset, status_kep
                    ]
                }
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        else:
            st.info("⬅️ Silakan isi data masyarakat di sebelah kiri, lalu klik **Prediksi Kelayakan**.")

        st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown("""
<div class="footer">
    Skripsi Teknik Informatika | Sistem Klasifikasi Kelayakan Penerima BPNT | Desa Paris, Kecamatan Mootilango
</div>
""", unsafe_allow_html=True)
