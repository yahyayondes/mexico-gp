import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "historical_races.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "featured_races.csv")

# Modele girecek features (hepsi yarıştan önce bilinen bilgiler)
FEATURES = [
    "grid_position",
    "driver_form_3",
    "driver_form_8",
    "driver_grid_form_3",
    "driver_dnf_rate_10",
    "team_form_3",
    "team_grid_form_3",
    "driver_track_avg",
]

# İsim değiştiren takımları tek bir seri olarak takip ediyoruz
TEAM_LINEAGE = {
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Audi",
    "Kick Sauber": "Audi",
    "Sauber": "Audi",
}

# Bu statüler klasmana giren (yarışı bitiren) pilotlar
CLASSIFIED_STATUS = r"^(Finished|Lapped|\+\d+ Laps?)$"


def _rolling_prev(series_by_group, window):
    # shift(1): o anki yarışın sonucu kendi özelliğine karışmasın
    return series_by_group.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    #burası baya önemli çünkü feature'ların karışmaması data leakage'ı önler


def compute_features(df):
    """
    Ham yarış satırlarından özellik üretir. Henüz koşulmamış bir yarışın satırları
    (finish_position = NaN) da verilebilir; özellikleri sadece geçmişten hesaplanır.
    """
    df = df.copy()
    df["team"] = df["team"].replace(TEAM_LINEAGE)
    df["n_cars"] = df.groupby(["year", "round"])["driver_id"].transform("size")

    # Grid 0 = pit çıkışı -> pole değil, en son sıra
    df["grid_position"] = df["grid_position"].where(df["grid_position"] > 0, df["n_cars"])

    # Zaman sırasına göre baya kritik!
    df = df.sort_values(by=["year", "round", "grid_position"]).reset_index(drop=True)

    finished = df["finish_position"].notna()
    df["dnf"] = np.where(finished, ~df["status"].fillna("").str.match(CLASSIFIED_STATUS), np.nan)
    # Form hesabında DNF'leri sonuç olarak saymıyoruz (ayrıca DNF oranı var)
    df["classified_pos"] = df["finish_position"].where(df["dnf"] == 0)

    # 1. Sürücü formu: son 3 ve son 8 yarış, ortalama sıralama (grid) temposu, DNF oranı
    by_driver = df.groupby("driver_id")
    df["driver_form_3"] = _rolling_prev(by_driver["classified_pos"], 3)
    df["driver_form_8"] = _rolling_prev(by_driver["classified_pos"], 8)
    df["driver_grid_form_3"] = _rolling_prev(by_driver["grid_position"], 3)
    df["driver_dnf_rate_10"] = _rolling_prev(by_driver["dnf"], 10)

    # 2. Takım formu: önce yarış bazında topla, sonra kaydır.
    # Böylece takım arkadaşının AYNI yarıştaki sonucu özelliğe sızmıyor.
    team_race = (
        df.groupby(["team", "year", "round"], as_index=False)
        .agg(team_pos=("classified_pos", "mean"), team_grid=("grid_position", "mean"))
        .sort_values(["year", "round"])
    )
    by_team = team_race.groupby("team")
    team_race["team_form_3"] = _rolling_prev(by_team["team_pos"], 3)
    team_race["team_grid_form_3"] = _rolling_prev(by_team["team_grid"], 3)
    df = df.merge(team_race[["team", "year", "round", "team_form_3", "team_grid_form_3"]],
                  on=["team", "year", "round"], how="left")

    # 3. Piste özel geçmiş: sürücünün bu pistte önceki yıllardaki ortalama bitişi
    df = df.sort_values(["year", "round", "grid_position"]).reset_index(drop=True)
    df["driver_track_avg"] = df.groupby(["driver_id", "event_name"])["finish_position"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Eksik değerleri doldurmuyoruz: LightGBM NaN'ı kendisi "bilgi yok" olarak işliyor

    # Benzersiz yarış kimliği (race_id)
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)

    # Hedef: yüksek skor = iyi sonuç. Araç sayısına göre 0..n-1 (22 araçta negatif olmaz)
    df["target_score"] = (df["n_cars"] - df["finish_position"]).clip(lower=0)

    return df


def build_features(input_path=RAW_PATH, output_path=PROCESSED_PATH):
    df = compute_features(pd.read_csv(input_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n==================================================")
    print(f"Toplam Satır / Sütun: {df.shape}")
    print(f"Oluşan Dosya: {output_path}")
    print("==================================================")

    return df


if __name__ == "__main__":
    build_features()
