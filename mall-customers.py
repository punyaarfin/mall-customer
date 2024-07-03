import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import base64

# Load data
df = pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace=True)

X = df.drop({'CustomerID', 'Gender'}, axis=1)

# Create a sidebar menu
menu = st.sidebar.selectbox("Menu", ["Beranda", "Clustering", "Petunjuk"])

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

image_base64 = load_image("LOGO-UNIVERSITAS-MUHAMMADIYAH-SORONG-UMS.png")  # Adjust the path to your image

if menu == "Beranda":
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
            <img src="data:image/png;base64,{image_base64}" width="150">
            <h1 style="text-align: center;">MENERAPKAN SEGMENTASI PELANGGAN MENGGUNAKAN METODE K-MEANS CLUSTERING</h1>
            <h3 style="text-align: center;">Oleh: Kelompok 2</h3>
            <h4 style="text-align: center;">Program Studi Teknik Informatika</h4>
            <h4 style="text-align: center;">Fakultas Teknik</h4>
            <h4 style="text-align: center;">Universitas Muhammadiyah Sorong</h4>
            <h4 style="text-align: center;">Tahun 2024</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "Clustering":
    st.header("Isi Dataset")
    st.write(X)
    
    # Menampilkan panah elbow
    cluster = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(X)
        cluster.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 11)), y=cluster, ax=ax)
    ax.set_title('Mencari Elbow')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')

    # Panah elbow
    ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
    ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

    st.sidebar.subheader("Nilai Jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2, 10, 3, 1)

    def k_means(n_clust):
        kmean = KMeans(n_clusters=n_clust).fit(X)
        X['Labels'] = kmean.labels_

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Income', y='Score', hue='Labels', data=X, palette=sns.color_palette('hls', n_clust))

        for label in X['Labels'].unique():
            plt.annotate(label,
                         (X[X['Labels'] == label]['Income'].mean(),
                          X[X['Labels'] == label]['Score'].mean()),
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=20, weight='bold',
                         color='black')
        st.header('Cluster Plot')
        st.pyplot()
        st.write(X)

    k_means(clust)

elif menu == "Petunjuk":
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Petunjuk Penggunaan Aplikasi Klasifikasi K-Means")
    
    st.markdown("""
    Aplikasi ini memungkinkan Anda untuk melakukan analisis clustering pada dataset pelanggan mall menggunakan metode K-Means. Berikut adalah langkah-langkah untuk menggunakan aplikasi ini:

    ### 1. Persiapan
    - Pastikan Anda memiliki file `Mall_Customers.csv` yang berisi data pelanggan mall.
    - Instal Streamlit jika belum terpasang dengan perintah berikut:
      ```sh
      pip install streamlit
      ```

    ### 2. Menjalankan Aplikasi
    - Buka terminal atau command prompt.
    - Navigasikan ke direktori tempat file `mall-customers.py` berada.
    - Jalankan perintah berikut untuk memulai aplikasi:
      ```sh
      streamlit run mall-customers.py
      ```

    ### 3. Navigasi Menu
    Aplikasi ini memiliki tiga menu utama yang dapat diakses melalui sidebar:

    #### A. Beranda
    - **Tampilan Beranda**: pada bagian beranda ini berisi tentang judul aplikasi ini dan pembuat aplikasi dimana tertera dibuat oleh Kelompok 2, serta program studi dan nama universitas.
    
    #### B. Clustering
    - **Isi Dataset**: Menampilkan dataset yang digunakan untuk analisis clustering.
    - **Mencari Elbow**: Menampilkan grafik Elbow untuk menentukan jumlah cluster yang optimal. Grafik ini membantu Anda menemukan titik "elbow" yang menunjukkan jumlah cluster yang paling sesuai untuk data.
    - **Plot Cluster**: Setelah Anda memilih jumlah cluster, aplikasi akan menampilkan scatter plot yang mengilustrasikan hasil clustering menggunakan metode K-Means.
    - **Jumlah K**: Anda dapat menentukan jumlah cluster K yang akan digunakan dalam algoritma K-Means dengan menggunakan slider yang tersedia di sidebar.

    #### C. Petunjuk
    - **Petunjuk Penggunaan**: Menampilkan panduan penggunaan aplikasi termasuk cara memulai klasifikasi dan informasi penting lainnya.

    ### 4. Langkah-langkah Penggunaan
    1. **Beranda**:
       - Klik menu "Beranda" untuk melihat isi dataset yang digunakan untuk analisis clustering.
       - Grafik Elbow akan ditampilkan untuk membantu menentukan jumlah cluster yang optimal.
       - Gunakan slider di sidebar untuk memilih jumlah cluster K yang diinginkan.
       - Hasil clustering akan ditampilkan dalam bentuk scatter plot, yang menunjukkan distribusi data pelanggan ke dalam cluster-cluster yang berbeda.
    
    2. **Petunjuk**:
       - Klik menu "Petunjuk" untuk membaca panduan penggunaan aplikasi dan informasi penting lainnya.
    """)
