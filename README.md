# Recommender System - Book Recommender

Dataset : [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset]  

Secara umum, sistem rekomendasi adalah algoritma yang ditujukan untuk menyarankan item yang relevan kepada pengguna (item seperti film untuk ditonton, buku untuk dibaca, produk untuk dibeli, atau apa pun tergantung pada industri). 
 
Proyek Machine Learning ini merupakan proyek sistem rekomendasi buku dengan tujuan akhir mampu merekomendasikan buku kepada pengguna melalui platform online.  
 
Alasan saya mengambil domain ini adalah karena permasalahan ini sangat sering ditemukan di dunia nyata, dimana semakin berkembangnya teknologi maka semakin mudah juga kita mengakses sesuatu, salah satunya mengakses situs penjualan buku atau perpustakaan elektronik. Namun, banyaknya jumlah buku terkadang membuat kita bingung untuk mencari buku baru sesuai yang kita mau. Sama halnya ketika mengunjungi toko buku, kita biasanya bertanya kepada pegawai disana buku mana yang direkomendasikan sesuai kebutuhan kita. Begitu juga penerapannya di dunia berbasis digital, kita ingin mencari rekomendasi buku tetapi tidak ada pegawai yang bisa kita tanyai, maka jalan satu-satu nya adalah menerima rekomendasi dari website tersebut. 
Maka dari itulah dibentuknya proyek sistem rekomendasi buku ini. 
 
Penelitian terkait topik ini adalah karya Moh.Irfan, et al. dengan judul "Sistem Rekomendasi : Buku Online dengan Metode Collaborative Filtering". 
[Jurnal Teknologi Technoscientia](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612)
 
 
## Business Understanding  
 
### Problem Statements :   
1. Dibutuhkan model machine learning yang bisa memberikan rekomendasi kepada pengguna  
2. Dari banyaknya buku yang ada, manakah yang direkomendasikan kepada pengguna ?  
 
### Goals : 
1. Membuat model machine learning rekomendasi buku  
2. Dapat memberikan rekomendasi buku kepada pengguna tertentu (misal user A, B, atau C).  
 
### Solutions :  
- Menganalisa data yang ada dan menangani permasalahan pada dataset  
- Menggunakan metode Collaborative Filtering dengan algoritma K-Nearest Neighbours  
- Menggunakan 2 pendekatan yaitu User-Based dan Item-Based dimana masing-masing pendekatan menggunakan 2 metrik, yaitu Cosine dan Correlation untuk membuat model yang bisa merekomendasikan buku kepada pengguna.  
 
    - **Collaborative Filtering** : Metode CF adalah metode dalam membuat model sistem rekomendasi yang dapat menyaring item yang mungkin disukai pengguna berdasarkan reaksi oleh pengguna atau item serupa. Metode CF menyarankan item berdasarkan bagaimana item tersebut dinilai juga oleh orang lain.  
    - **K-Nearest Neighbours** : KNN adalah model yang mengklasifikasikan titik-titik data berdasarkan titik-titik yang paling mirip dengannya.
    KNN sering digunakan dalam sistem rekomendasi sederhana, teknologi pengenalan citra, dan model pengambilan keputusan. Ini adalah algoritma yang digunakan perusahaan seperti Netflix atau Amazon untuk merekomendasikan film yang berbeda untuk ditonton atau buku untuk dibeli. 
 
 
 
## Data Understanding 
 
Dalam proyek ini, item yang disarankan adalah buku untuk dibaca dengan rincian dataset sebagai berikut :
 
 
1. Users
Berisi 278858 dengan 3 kolom data pengguna.
 
    - ID pengguna (**User-ID**) telah dianonimkan dan dipetakan ke bilangan bulat.
    - Data demografis (**Lokasi**, **Usia**) jika tersedia. Jika tidak, bidang ini berisi nilai NULL.
    ![Screenshot (258)](https://user-images.githubusercontent.com/89563587/140680310-996c607e-2eda-47a8-9247-42164f3bfbeb.png)
     
 
2. Books
Berisi 271360 dengan 8 kolom data buku.
 
    - **ISBN**. ISBN yang tidak valid telah dihapus dari dataset.
    - **Judul Buku**, **Penulis Buku**, **Tahun Penerbitan**, **Penerbit**, diperoleh dari Amazon Web Services. Dalam kasus beberapa penulis, hanya nama yang pertama disediakan.
    - Gambar. URL yang tertaut ke gambar sampul juga diberikan (**Image-URL-S**, **Image-URL-M**, **Image-URL-L**), yaitu kecil, sedang, besar. URL ini mengarah ke situs web Amazon.
    ![Screenshot (256)](https://user-images.githubusercontent.com/89563587/140680217-f46fc161-6bbf-428e-a5dd-8383d5501d7e.png)
 
3. Ratings
Berisi 1149780 dengan 3 kolom data rating buku.
 
    - Rating (**Book-Rating**) bersifat eksplisit, dinyatakan dalam skala 1-10 (nilai yang lebih tinggi menunjukkan apresiasi yang lebih tinggi) atau implisit, yang dinyatakan dengan 0.
    ![Screenshot (257)](https://user-images.githubusercontent.com/89563587/140680291-50d8d9f5-9e8c-43c6-967d-866322b733b5.png)
    
Sumber data : [Kaggle](https://www.kaggle.com/arashnic/book-recommendation-dataset)
 
 
Pada tahap Data Understanding saya melakukan eksplorasi data terlebih dahulu, yaitu *Preliminary Exploration*, *Exploratory Data Analysis (EDA)*, dan visualisasi data.
 
*Preliminary Exploration* adalah tahapan paling awal dalam eksplorasi data yaitu saya melakukan pemeriksaan dataset terlebih dahulu untuk mendapatkan informasi mengenai dataset (Buku, *Rating*, Pengguna). Lalu mencoba menampilkan isi dari dataset.
 
*Exploratory Data Analysis (EDA)* adalah tahap menganalisis eksplorasi data untuk memahami data. Yang saya lakukan adalah mengganti nama variabel dengan lebih sederhana agar memudahkan memanggil nama variabel, menghapus kolom yang tidak dibutuhkan, dan mengecek baris data yang terduplikat. Lalu, untuk memudahkan dalam memahami data saya juga memvisualisasikan data seperti contohnya visualisasi *missing value* di data Books dan data penerbit yang paling banyak menerbitkan buku.
 
*Missing Value* pada dataset Books
 
![Screenshot (3465)](https://user-images.githubusercontent.com/89563587/139971278-65ff54dd-2e1a-4e97-9ac2-1d6cacaef9b0.png) 
 
Dari visualisasi diatas, terlihat tidak ada missing value pada dataset Books sehingga tidak ada value yang perlu kita atasi dari dataset ini. 
 
*Penerbit yang paling banyak menerbitkan buku*
 
 
![Screenshot (3466)](https://user-images.githubusercontent.com/89563587/139984169-32366bfe-1b22-4d45-b333-364159aa9ccc.png)
 
Dari grafik diatas terlihat bahwa penerbit Harlequin adalah penerbit yang paling banyak menerbitkan buku yaitu 7535 buku. Lalu disusul oleh penerbit Silhoutte dan Pocket di posisi 2 dan 3 dengan total buku 4220 dan 3905. Bisa kita asumsikan bahwa kemungkinan besar buku yang direkomendasikan kepada pengguna adalah salah satu dari penerbit yang paling banyak menerbitkan buku ini. 
 
 
## Data Preparation
Pada tahap ini, saya melakukan 
- Menggabungkan dataframe 
Yang pertama adalah saya menggabungkan kolom ISBN yang ada pada dataset Books dan Ratings karena sama-sama memiliki variabel ISBN. Tujuannya adalah agar memudahkan proses mengolah data dan tidak ada data ISBN terduplikat.
 
- Mencari dan mengatasi *wrong value* pada tahun terbit
Selanjutnya adalah mencari *wrong value* di bagian tahun terbit dan mengatasi *wrong value* tersebut yaitu dengan memastikan semua *value* nya bernilai angka dan mengatasi tahun yang bernilai 0 atau diatas 2021 dengan mencari nilai rata-rata dari keseluruhan data sehingga menjadi lebih masuk akal daripada diatas 2021. 
 
- Memisahkan nilai variabel
Dikarenakan informasi di awal dikatakan bahwa nilai rating memiliki 2 nilai yaitu eksplisit dan implisit, maka saya memisahkan nilai variabel tersebut karena nanti hanya akan menggunakan nilai yang eksplisit saja yaitu rating yang diberi nilai 1-10, sehingga tidak ada nilai rating 0.
 
    Berikut adalah grafik dari *rating* buku
    
    ![Screenshot (3467)](https://user-images.githubusercontent.com/89563587/139971114-2d4096ad-c55a-4e7e-9b8b-9740521fcc37.png) 
    
    Dari grafik diatas terlihat bahwa *rating* 8 adalah *rating* yang paling banyak diberikan oleh pengguna dengan mencapai total lebih dari 80000 buku yang diberi *rating* 8. 
 
 
 
## Modelling
 
Pada tahap ini, saya menggunakan algoritma KNN dengan pendekatan Item-Based dan User-based dengan menggunakan metrik correlation dan cosine sebagai pilihannya *(select feature)*. Tujuannya adalah untuk melihat bagaimana model memberikan rekomendasi melalui pendekatan Item based dan User based.
 
**K-Nearest Neighbours** 
Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN mencari kesamaan dari tetangga terdekatnya dan itu sangat cocok dengan sistem rekomendasi yang menggunakan metode collaborative filtering. 
 
Kelebihan algoritma ini adalah tidak adanya periode training sehingga mampu bekerja dengan baik pada prediksi langsung. Ini menyimpan set data pelatihan dan belajar darinya hanya pada saat membuat *real-time prediction*. Ini membuat algoritma KNN jauh lebih cepat daripada algoritma lain yang membutuhkan periode training. Selain itu, data baru dapat ditambahkan dengan mulus namun tidak akan memengaruhi keakuratan algoritma. Hal ini sesuai dengan sistem rekomendasi yang memberikan prediksi *real-time* dan dataset yang berubah-ubah. 
 
Namun, kekurangannya adalah algoritma KNN ensitif terhadap *noise*, *missing values* dan *outliers*. Kita perlu secara manual memasukkan nilai yang hilang dan menghapus outlier. Itulah sebabnya pada saat data preparation banyak *missing value* dan *wrong value* yang harus diatasi.
 
**User-Based and Item-Based** 
User based adalah pendekatan yang menghasilkan rekomendasi dari pengguna lain yang memiliki selera yang sama. Sedangkan Item based bekerja dengan cara menghitung kesamaan antara masing-masing item. Penerapannya dilakukan dengan memanfaatkan matriks rating yang sudah dibuat. 
 
Pada dasarnya ini hanyalah satu model yaitu KNN namun menggunakan 2 pendekatan yang berbeda tergantung kebutuhan. Jika ingin memberikan rekomendasi berdasarkan apa buku yang juga dibaca oleh orang lain, maka menggunakan User based. Namun jika ingin mencari buku yang serupa maka menggunakan Item based.
 
![Screenshot (3513)](https://user-images.githubusercontent.com/89563587/140598051-cd9eca88-f931-473b-9b8f-18adbd162914.png)
 
![Screenshot (3514)](https://user-images.githubusercontent.com/89563587/140598056-aca2b59f-bec7-4d68-b0db-c202d7f101d2.png)
 
 
Untuk metrik yang digunakan juga menggunakan 2 metrik yaitu cosine dan correlation untuk melihat dimana perbedannya. Cosine adalah metrik yang digunakan untuk menghitung kesamaan dalam 2 sampel sedangkan correlation menghitung korelasi antar 2 variabel acak.
 
Setelah dilakukan prediksi, rekomendasi buku yang diberikan dari pendekatan User-based dan Item-based menghasilkan rekomendasi yang berbeda. Sedangkan untuk pendekatan yang sama namun berbeda tetap memberikan rekomendasi yang sama. 
 
* Item-Based dan User Based (cosine)
![Screenshot (3515)](https://user-images.githubusercontent.com/89563587/140598105-3933b6ce-0a5a-46e5-ad87-0eefea257a9b.png)
 
* Cosine dan Correlation (User-based)
![Screenshot (3511)](https://user-images.githubusercontent.com/89563587/140598118-91080312-96d7-4b7b-86c9-bae478af8f28.png)
 
 
Kesimpulannya adalah rekomendasi sistem berpengaruh sangat besar terhadap jenis pendakatan yang akan kita lakukan, apakah berdasarkan pengguna atau item.  
 
## Evaluation
 
Pada tahap ini saya menggunakan metrik evaluasi RMSE dari model KNN.  
 
**Root Mean Square Error** 
Root mean squared error (RMSE) adalah akar kuadrat dari mean kuadrat dari semua error. Penggunaan RMSE sangat umum dan dianggap sebagai metrik error yang sangat baik untuk prediksi numerik dengan mengukur tingkat hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. 
 
Rumus RMSE
![image](https://user-images.githubusercontent.com/89563587/139566767-9c5a33f9-9b8b-4bcc-9474-7a1295a5af45.png)
 
 
Keterangan:
 
At = Nilai data Aktual
Ft = Nilai hasil peramalan
N= banyaknya data
∑ = Summation (Jumlah keseluruhan  nilai)  
 
 
RMSE pada KNN
 
![Screenshot (3518)](https://user-images.githubusercontent.com/89563587/140598330-9e452aa4-0b0d-432d-9035-b74c040a8e0c.png)
 
 
Dari hasil evaluasi diatas terlihat bahwa tingkat error(kesalahan) dari model termasuk rendah karena masih dibawah 1. Hal ini menunjukkan bahwa algoritma KNN bekerja cukup stabil meskipun *real-time prediction* dan dataset yang cukup besar. 
 
Pada dasarnya hanyalah terdapat satu model saja yaitu KNN meskipun menggunakan pendekatan yang berbeda. Moodel hanya bisa dievaluasi dari salah satu pendekatan. Saya sudah mencoba untuk mengevaluasi Item Based dan User Based namun model mengeluarkan hasil evaluasi hanya pada satu pendekatan saja. Sebaliknya jika saya memilih pendekatan lain, model tetap menghasilkan output evaluasi yang sama seperti pada pendekatan yang dpilih sebelumnya.
 
 
*Sekian dan terimakasih*

