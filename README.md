# Mexico GP 2026 - Yarış Sonucu Tahmin Modeli

Bu projede FastF1 verisi ile önümüzdeki haftasonu (30 Ekim - 1 Kasım) gerçekleşecek olan Mexico GP için bir prediction modeli oluşturmaya çalıştım.
Projenin öğrendiklerime katkı sunacağını ve pratik yapmam açısından faydalı olduğunu düşünüyorum.
Geliştirme süresince veri çekme, modeli oluşturma ve eğitme, prediction oluşturma süreçlerinde mimariyi kendim kurdum;
kodu yapay zeka yardımıyla (tamamıyla değil yardımıyla) yazdım. Görselleştirme sürecini ise tamamen yapay zekaya bıraktım.

Sonuçları yarış sonrasında güncelleyeceğim

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma (sırasıyla)

```bash
python src/data_fethcer.py   # artımlı: sadece eksik yarışları çeker
python src/features.py       # özellikler -> data/processed/featured_races.csv
python src/models.py         # eğitim + 2026 testi + kalibrasyon -> models/race_ranker.pkl
python src/predict_race.py   # tahmin -> outputs/*.csv ve outputs/mexico_gp_2026_tahmin.html
```

## Grid kaynağı

`predict_race.py` gridi şu sırayla arar:

1. `data/grid_mexico_2026.csv` (elle: `driver_id,team,grid_position`; grid cezaları için)
2. FastF1'den Mexico 2026 sıralama turu sonuçları
3. Hiçbiri yoksa **geçici** grid: son 3 yarışın ortalama grid pozisyonu

Sıralama turları bittikten sonra script'i tekrar çalıştırın.

## Yöntem

- **Özellikler** (hepsi yarıştan önce bilinen): grid, son 3/8 yarış formu (DNF hariç),
  son 3 yarış grid ortalaması, takım formu (yarış bazında toplanıp kaydırılmış, sızıntı yok),
  son 10 yarış DNF oranı, pilotun bu pistteki geçmiş ortalaması.
- **Bölme:** eğitim 2022-24, doğrulama 2025 (ayar seçimi), test 2026 (yeni kurallar).
- **Skor:** LightGBM LambdaRank skoru, doğrulamada seçilen ağırlıkla `-log(grid)` ile harmanlanır.
- **Olasılık:** Plackett-Luce, pozisyona bağlı sıcaklık (ön sıralar daha belirleyici),
  2025-26 örneklem dışı tahminlerle maksimum olabilirlikle kalibre edilir; 20.000 simülasyon.

## Sınırlamalar

Modelin 2026 testindeki performansı "grid sırası = sonuç" referansına çok yakın:
kazananı bilme aynı, sıralama korelasyonu biraz daha iyi, olasılık log-kaybı biraz daha kötü.
Sıralama turundaki zaman farkları, antrenman temposu, hava durumu ve sprint verileri kullanılmıyor.
