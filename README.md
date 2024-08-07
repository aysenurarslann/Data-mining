# Hitters Dataset Analysis and Modeling

## Proje Açıklaması
Bu proje, Hitters veri setini kullanarak çeşitli makine öğrenme ve veri görselleştirme tekniklerini uygulamaktadır. Projede eksik verilerin işlenmesi, verilerin standardize edilmesi, PCA kullanarak boyut indirgeme, karar ağacı ve rastgele orman regresyon modelleri ile tahminler yapılması ve K-means kümeleme analizi bulunmaktadır.

## Kullanılan Kütüphaneler
- pandas
- sklearn
- matplotlib
- seaborn
- numpy

## Adımlar

1. **Veri Setinin Yüklenmesi ve Hazırlanması**
   - `hitters.csv` dosyasının yüklenmesi
   - Eksik verilerin işlenmesi ve kategorik değişkenlerin One-Hot Encoding ile dönüştürülmesi

2. **Veri Standardizasyonu**
   - Verilerin `StandardScaler` kullanılarak standardize edilmesi

3. **Randomized PCA ile Boyut İndirgeme**
   - PCA kullanarak verilerin iki bileşene indirgenmesi ve görselleştirilmesi

4. **Veri Görselleştirme**
   - Kutu grafikleri ve scatter plotlar ile veri dağılımının ve PCA sonuçlarının görselleştirilmesi

5. **Regresyon Modelleri**
   - Karar Ağacı ve Rastgele Orman regresyon modelleri oluşturulması
   - Modellerin eğitim ve test setlerinde değerlendirilmesi ve görselleştirilmesi

6. **K-means Kümeleme Analizi**
   - K-means kümeleme modeli oluşturulması ve sonuçların görselleştirilmesi

7. **Başarı Oranlarının Hesaplanması**
   - Eğitim, test ve çapraz doğrulama setlerinde RMSE (Root Mean Squared Error) hesaplanması

## Kullanım
Aşağıdaki adımları izleyerek projeyi çalıştırabilirsiniz:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas sklearn matplotlib seaborn numpy
