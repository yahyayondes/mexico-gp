import os
import warnings
import logging
import fastf1
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.simplefilter("ignore", FutureWarning)

# Script nereden çalıştırılırsa çalıştırılsın proje köküne göre yollar
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")
RAW_PATH = os.path.join(ROOT, "data", "raw", "historical_races.csv")

# FastF1 caching
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

COLUMNS = ["year", "round", "event_name", "driver_id", "team", "grid_position",
           "finish_position", "status", "points"]


def fetch_historical_data(start_year=2022, end_year=2026, output_path=RAW_PATH):
    # Daha önce çekilen yarışları tekrar çekmiyoruz (artımlı güncelleme)
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
    else:
        existing = pd.DataFrame(columns=COLUMNS)
    done = set(zip(existing["year"], existing["round"]))

    new_rows = []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    try:
        for year in range(start_year, end_year + 1):
            print(f"\n------------------{year} sezonu çekiliyor--------------")
            try:
                schedule = fastf1.get_event_schedule(year)
                # sadece official ve yarışı bitmiş haftasonları
                official_events = schedule[schedule["RoundNumber"] > 0]
                official_events = official_events[official_events["Session5DateUtc"] < now]
            except Exception as e:
                print(f"{year} takvimi alınırken sorun oluştu: {e}")
                continue

            for _, event in official_events.iterrows():
                round_num = int(event["RoundNumber"])
                event_name = event["EventName"]
                if (year, round_num) in done:
                    continue
                print(f"Çekiliyor: {year} Round {round_num} - {event_name}")

                try:
                    # yarış seansı (laps=False -> çok daha az API isteği)
                    race_session = fastf1.get_session(year, round_num, "R")
                    race_session.load(laps=False, telemetry=False, weather=False, messages=False)
                    results = race_session.results
                except fastf1.req.RateLimitExceededError:
                    # Limit aşıldıysa devam etmenin anlamı yok, eldekini kaydedip çıkıyoruz
                    raise
                except Exception as e:
                    print(f"Hata ({year} R{round_num}): {e}")
                    continue

                if results is None or results.empty or results["Position"].isna().all():
                    print(f"UYARI: {year} R{round_num} verisi boş!!!")
                    continue

                saved_count = 0
                for _, driver in results.iterrows():
                    finish_pos = driver.get("Position", None)
                    # Eğer finiş pozisyonu yoksa geç
                    if pd.isna(finish_pos):
                        continue

                    new_rows.append(
                        {
                            "year": year,
                            "round": round_num,
                            "event_name": event_name,
                            "driver_id": driver.get("Abbreviation", None) or driver.get("BroadcastName", "UNK"),
                            "team": driver.get("TeamName", "Unknown"),
                            "grid_position": driver.get("GridPosition", 0),
                            "finish_position": finish_pos,
                            "status": driver.get("Status", "Finished"),
                            "points": driver.get("Points", 0.0),
                        }
                    )
                    saved_count += 1

                print(f"Başarılı {saved_count} sürücü kaydedildi")

    except fastf1.req.RateLimitExceededError as e:
        print(f"\nAPI limiti aşıldı ({e}). Eldeki veri kaydediliyor, bir saat sonra tekrar çalıştırın.")

    finally:
        # Hata olsa bile mevcut veriyi kaybetmeden yeni satırları ekle
        df = pd.concat([existing, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
        df = df.drop_duplicates(subset=["year", "round", "driver_id"], keep="last")
        df = df.sort_values(["year", "round", "finish_position"]).reset_index(drop=True)
        df.to_csv(output_path, index=False)

    print("\n==================================================")
    print(f"Yeni eklenen satır: {len(new_rows)}")
    print(f"Toplam satır sayısı: {len(df)}")
    print(f"Yıl başına yarış sayısı: {df.groupby('year')['round'].nunique().to_dict()}")
    print("==================================================")

    return df


if __name__ == "__main__":
    fetch_historical_data(2022, 2026)
