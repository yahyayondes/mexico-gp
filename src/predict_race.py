import os
import pickle
import warnings
import logging
import pandas as pd
import numpy as np
import fastf1

from features import compute_features, RAW_PATH, ROOT
from models import MODEL_PATH, blend_scores, stage_temperatures

logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.simplefilter("ignore", FutureWarning)
fastf1.Cache.enable_cache(os.path.join(ROOT, "data", "cache"))

YEAR, ROUND, EVENT_NAME = 2026, 19, "Mexico City Grand Prix"
# Ceza vb. yüzünden gerçek grid sıralamadan farklıysa bu dosyayı elle doldur:
# driver_id,team,grid_position
MANUAL_GRID_PATH = os.path.join(ROOT, "data", "grid_mexico_2026.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
N_SIMULATIONS = 20000


def load_qualifying_grid():
    """Sıralama turları bittiyse FastF1'den gridi çeker, yoksa None."""
    try:
        quali = fastf1.get_session(YEAR, ROUND, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        res = quali.results
        if res is None or res.empty or res["Position"].isna().all():
            return None
        grid = res[["Abbreviation", "TeamName", "Position"]].dropna()
        grid.columns = ["driver_id", "team", "grid_position"]
        return grid
    except Exception as e:
        print(f"Sıralama sonuçları alınamadı ({e})")
        return None


def provisional_grid(history):
    """
    Sıralama henüz yapılmadıysa: son yarıştaki kadro, son 3 yarışın ortalama
    grid pozisyonuna (sıralama temposu) göre dizilir.
    """
    last = history[(history["year"] == history["year"].max())]
    last = last[last["round"] == last["round"].max()]
    roster = last[["driver_id", "team"]].copy()
    roster["grid_position"] = np.arange(1, len(roster) + 1)  # geçici yer tutucu

    feats = upcoming_features(history, roster)
    order = feats.sort_values("driver_grid_form_3")["driver_id"].tolist()
    roster["grid_position"] = roster["driver_id"].map({d: i + 1 for i, d in enumerate(order)})
    return roster


def upcoming_features(history, grid):
    # Yeni yarışı geçmişe ekleyip aynı özellik fonksiyonundan geçiriyoruz (eğitimle birebir tutarlı)
    upcoming = grid.assign(year=YEAR, round=ROUND, event_name=EVENT_NAME,
                           finish_position=np.nan, status=np.nan, points=np.nan)
    feats = compute_features(pd.concat([history, upcoming], ignore_index=True))
    return feats[(feats["year"] == YEAR) & (feats["round"] == ROUND)].reset_index(drop=True)


def simulate_positions(scores, temperature_params, n_sims=N_SIMULATIONS, seed=42):
    """
    Plackett-Luce simülasyonu: 1. sıradan başlayarak her pozisyon için kalan pilotlar
    arasından skor/T_k olasılıklarıyla seçim yapılır (Gumbel-max hilesi ile).
    Dönen: [pilot, pozisyon] olasılık matrisi ve [simülasyon, pilot] pozisyonları.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    temps = stage_temperatures(n, temperature_params)
    alive = np.ones((n_sims, n), dtype=bool)
    positions = np.zeros((n_sims, n), dtype=int)
    rows = np.arange(n_sims)
    for k in range(n):
        logits = np.where(alive, scores[None, :] / temps[k], -np.inf) + rng.gumbel(size=(n_sims, n))
        pick = logits.argmax(axis=1)
        positions[rows, pick] = k + 1
        alive[rows, pick] = False
    prob = np.stack([(positions == p + 1).mean(axis=0) for p in range(n)], axis=1)
    return prob, positions


def predict_upcoming_race(grid_inputs=None, model_path=MODEL_PATH):
    """
    grid_inputs: [{'driver_id': 'VER', 'team': 'Red Bull Racing', 'grid_position': 1}, ...]
    Verilmezse: elle girilen CSV -> FastF1 sıralama sonuçları -> geçici grid (bu sırayla).
    """
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        return

    # Eğitilmiş modeli yükle
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    history = pd.read_csv(RAW_PATH)

    if grid_inputs is not None:
        grid, grid_source = pd.DataFrame(grid_inputs), "elle verilen liste"
    elif os.path.exists(MANUAL_GRID_PATH):
        grid, grid_source = pd.read_csv(MANUAL_GRID_PATH), MANUAL_GRID_PATH
    else:
        grid = load_qualifying_grid()
        grid_source = "FastF1 sıralama sonuçları"
        if grid is None:
            grid = provisional_grid(history)
            grid_source = "GEÇİCİ (sıralama yapılmadı, son 3 yarışın grid ortalamasından)"

    df_race = upcoming_features(history, grid)

    # Tahmin
    model_pred = bundle["model"].predict(df_race[bundle["features"]])
    df_race["pred_score"] = blend_scores(model_pred, df_race["grid_position"],
                                         bundle["scales"], bundle["grid_weight"])

    prob, positions = simulate_positions(df_race["pred_score"].to_numpy(), bundle["temperature"])
    n = len(df_race)
    pos_idx = np.arange(1, n + 1)
    df_race["expected_position"] = prob @ pos_idx
    df_race["p10_position"] = np.percentile(positions, 10, axis=0)
    df_race["p90_position"] = np.percentile(positions, 90, axis=0)
    df_race["p_win"] = prob[:, 0]
    df_race["p_podium"] = prob[:, :3].sum(axis=1)
    df_race["p_points"] = prob[:, :10].sum(axis=1)
    for p in range(n):
        df_race[f"p_pos_{p + 1}"] = prob[:, p]

    df_race = df_race.sort_values("expected_position").reset_index(drop=True)
    df_race["predicted_finish"] = df_race.index + 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_cols = (["predicted_finish", "driver_id", "team", "grid_position", "expected_position",
                 "p10_position", "p90_position", "p_win", "p_podium", "p_points"]
                + [f"p_pos_{p + 1}" for p in range(n)])
    out_path = os.path.join(OUTPUT_DIR, "mexico_gp_2026_probabilities.csv")
    df_race[out_cols].to_csv(out_path, index=False)

    print("\n==================================================")
    print(f"MEXICO CITY GP 2026 TAHMİNİ  |  Grid kaynağı: {grid_source}")
    print("==================================================")
    show = df_race[["predicted_finish", "driver_id", "team", "grid_position", "expected_position",
                    "p_win", "p_podium", "p_points"]].copy()
    for c in ["p_win", "p_podium", "p_points"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    show["expected_position"] = show["expected_position"].round(1)
    print(show.to_string(index=False))
    print(f"\nOlasılıklar kaydedildi: {out_path}")
    print("==================================================\n")

    return df_race, grid_source


if __name__ == "__main__":
    # Ceza vb. yüzünden gerçek grid farklıysa: MANUAL_GRID_PATH dosyasını oluştur
    # ya da predict_upcoming_race([...]) ile listeyi doğrudan ver.
    from visualize import build_html

    result = predict_upcoming_race()
    if result is not None:
        build_html(*result)
