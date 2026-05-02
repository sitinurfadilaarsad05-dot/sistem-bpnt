import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Kelayakan BPNT",
    page_icon="🏛️",
    layout="centered"
)

# ============================================================
# CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .header-box {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white;
        padding: 30px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(21,101,192,0.3);
    }

    .header-box h1 {
        font-size: 22px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: 0.5px;
    }

    .header-box p {
        font-size: 13px;
        opacity: 0.85;
        margin: 2px 0;
    }

    .form-box {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e0e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E3F2FD;
    }

    .result-layak {
        background: linear-gradient(135deg, #E8F5E9, #F1F8F1);
        border: 2px solid #4CAF50;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
    }

    .result-tidak {
        background: linear-gradient(135deg, #FFEBEE, #FFF5F5);
        border: 2px solid #EF5350;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
    }

    .result-icon { font-size: 52px; margin-bottom: 8px; }
    .result-name { font-size: 20px; font-weight: 700; margin-bottom: 4px; }

    .result-status-layak {
        font-size: 28px; font-weight: 800;
        color: #2E7D32; margin-bottom: 8px;
    }

    .result-status-tidak {
        font-size: 28px; font-weight: 800;
        color: #C62828; margin-bottom: 8px;
    }

    .result-desc { font-size: 14px; opacity: 0.75; }

    .akurasi-box {
        background: linear-gradient(135deg, #E3F2FD, #EFF8FF);
        border: 2px solid #1976D2;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        text-align: center;
    }

    .akurasi-best {
        background: linear-gradient(135deg, #E8F5E9, #F1F8F1);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        text-align: center;
    }
            
    .akurasi-k { font-size: 16px; font-weight: 700; color: #1565C0; }
    .akurasi-val { font-size: 28px; font-weight: 800; color: #1565C0; }
    .akurasi-best .akurasi-k { color: #2E7D32; }
    .akurasi-best .akurasi-val { color: #2E7D32; }
    

    .footer {
        text-align: center;
        font-size: 12px;
        color: #999;
        margin-top: 30px;
        padding-top: 16px;
        border-top: 1px solid #e0e0e0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white;
        font-weight: 700;
        font-size: 15px;
        padding: 12px 32px;
        border-radius: 10px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0D47A1, #1565C0);
        box-shadow: 0 4px 15px rgba(21,101,192,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA & PREPROCESSING (cached)
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
                df_test, True)

    except Exception as e:
        return (None,)*9 + (None, str(e))


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>🏛️ SISTEM KLASIFIKASI KELAYAKAN BPNT</h1>
    <p>Menggunakan Algoritma K-Nearest Neighbor (KNN)</p>
    <p>Desa Paris, Kecamatan Mootilango</p>
</div>
""", unsafe_allow_html=True)

# Load data
result = load_data()
(X_train_scaled, y_train_enc, X_test_scaled, y_test_enc,
 scaler, le_dict, le_label, kolom_fitur, kolom_kategori, df_test, status) = result

if status != True:
    st.error(f"❌ Gagal memuat data: {status}\n\nPastikan file **'DATA FIX SKRIPSI KNN.xlsx'** ada di folder yang sama!")
    st.stop()


# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2 = st.tabs(["🔍 Prediksi Data Baru", "📊 Uji Data dengan K=3,5,7"])


# ============================================================
# TAB 1 - PREDIKSI DATA BARU
# ============================================================
with tab1:
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Data Masyarakat</div>', unsafe_allow_html=True)

    nama = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")

    col1, col2 = st.columns(2)
    with col1:
        pendapatan = st.number_input("Pendapatan (Rp)", min_value=0, step=50000, help="Contoh: 800000")
    with col2:
        tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, step=1, help="Contoh: 4")

    col3, col4 = st.columns(2)
    with col3:
        pekerjaan = st.selectbox("Pekerjaan", ["Buruh/Tani", "Wiraswasta", "PNS/Tetap"])
    with col4:
        rumah = st.selectbox("Kondisi Rumah", ["Bambu/Tanah", "Semi Perm.", "Tembok/Keramik"])

    col5, col6 = st.columns(2)
    with col5:
        aset = st.selectbox("Kepemilikan Aset", ["Tidak Ada", "Motor", "Mobil"])
    with col6:
        status_kep = st.selectbox("Status Kepemilikan", ["Menumpang", "Milik Keluarga", "Milik Sendiri"])

    st.markdown("---")
    k_val = st.radio("Pilih Nilai K (Jumlah Tetangga Terdekat):",
                     [3, 5, 7], horizontal=True, index=0)

    st.markdown('</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        prediksi_btn = st.button("🔍 PREDIKSI KELAYAKAN", use_container_width=True, key="btn_prediksi")
    with col_btn2:
        reset_btn = st.button("🔄 Reset", use_container_width=True, key="btn_reset")

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

            st.markdown(f"**Hasil prediksi menggunakan K={k_val}:**")

            if hasil_label == 'Layak':
                st.markdown(f"""
                <div class="result-layak">
                    <div class="result-icon">✅</div>
                    <div class="result-name">{nama}</div>
                    <div class="result-status-layak">LAYAK</div>
                    <div class="result-desc">Masyarakat ini <strong>LAYAK</strong> menerima bantuan BPNT</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-tidak">
                    <div class="result-icon">❌</div>
                    <div class="result-name">{nama}</div>
                    <div class="result-status-tidak">TIDAK LAYAK</div>
                    <div class="result-desc">Masyarakat ini <strong>TIDAK LAYAK</strong> menerima bantuan BPNT</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# TAB 2 - UJI DATA DENGAN K=3,5,7
# ============================================================
with tab2:
    st.markdown("### 📊 Perbandingan Akurasi K=3, K=5, K=7")
    st.markdown("Menguji seluruh **data uji** dengan 3 nilai K berbeda dan membandingkan hasilnya.")

    hasil_semua = {}
    akurasi_semua = {}

    for k in [3, 5, 7]:
        knn_k = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn_k.fit(X_train_scaled, y_train_enc)
        pred_enc = knn_k.predict(X_test_scaled)
        pred_label = le_label.inverse_transform(pred_enc)
        akurasi = accuracy_score(y_test_enc, pred_enc) * 100
        hasil_semua[k] = pred_label
        akurasi_semua[k] = akurasi

    # Tampilkan akurasi per K
    best_k = max(akurasi_semua, key=akurasi_semua.get)
    col_k1, col_k2, col_k3 = st.columns(3)

    for col, k in zip([col_k1, col_k2, col_k3], [3, 5, 7]):
        with col:
            css_class = "akurasi-best" if k == best_k else "akurasi-box"
            badge = " 🏆 Terbaik" if k == best_k else ""
            st.markdown(f"""
            <div class="{css_class}">
                <div class="akurasi-k">K = {k}{badge}</div>
                <div class="akurasi-val">{akurasi_semua[k]:.1f}%</div>
                <div style="font-size:12px;color:#666;">Akurasi</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabel perbandingan hasil prediksi
    st.markdown("### 📋 Tabel Hasil Prediksi Data Uji")

    df_hasil = df_test[['Nama','Label']].copy()
    df_hasil.columns = ['Nama', 'Label Asli']
    df_hasil['Prediksi K=3'] = hasil_semua[3]
    df_hasil['Prediksi K=5'] = hasil_semua[5]
    df_hasil['Prediksi K=7'] = hasil_semua[7]

    def style_df(row):
        asli = row['Label Asli']
        styles = ['', '']
        for col in ['Prediksi K=3', 'Prediksi K=5', 'Prediksi K=7']:
            if row[col] == asli:
                styles.append('background-color: #E8F5E9; color: #2E7D32; font-weight: bold')
            else:
                styles.append('background-color: #FFEBEE; color: #C62828; font-weight: bold')
        return styles

    styled = df_hasil.style.apply(style_df, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    > 🟢 **Hijau** = prediksi benar &nbsp;&nbsp; 🔴 **Merah** = prediksi salah
    """)

    st.markdown("---")
    st.markdown(f"### 🏆 Kesimpulan: K terbaik adalah **K={best_k}** dengan akurasi **{akurasi_semua[best_k]:.1f}%**")


# Footer
st.markdown("""
<div class="footer">
    Skripsi Teknik Informatika | Desa Paris, Kecamatan Mootilango
</div>
""", unsafe_allow_html=True)
