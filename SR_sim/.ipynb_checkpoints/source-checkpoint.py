# pip install -U astroquery pandas
from astroquery.gaia import Gaia
import pandas as pd
from pathlib import Path

# --- настройки ---
N_TOTAL      = 2_000_000          # нужно строк
CHUNK        = 500_000            # качаем кусками, чтобы избежать таймаутов
OUT_DIR      = Path("gaia_dl")    # сюда сложим файлы
COLUMNS      = ["source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "random_index"]
GAIA_TABLE   = "gaiadr3.gaia_source"   # таблица DR3

OUT_DIR.mkdir(parents=True, exist_ok=True)

# по желанию: логин в архив (для повышения квот/лимитов аккаунта)
# Gaia.login(user="your_email", password="your_password")

def adql_for_range(lo, hi):
    """
    Берём 'случайный' поднабор через непрерывный диапазон random_index.
    random_index — это случайная перестановка 0..N-1, так что диапазон даёт
    равномерную выборку. См. офиц. рекомендации. 
    """
    cols = ", ".join(COLUMNS)
    # выбираем ИМЕННО нужные колонки, без ORDER BY (так быстрее на архивах)
    return f"""
    SELECT {cols}
    FROM {GAIA_TABLE}
    WHERE random_index BETWEEN {lo} AND {hi - 1}
    """

def fetch_chunk(lo, hi, out_csv_gz):
    q = adql_for_range(lo, hi)
    job = Gaia.launch_job_async(
        q,
        dump_to_file=True,
        output_format="csv",       # сервер вернёт .csv.gz
        output_file=str(out_csv_gz)
    )
    print("JOB:", job.jobid, "PHASE:", job.phase, "->", out_csv_gz)

# ---- качаем куски ----
ranges = []
start = 0
while start < N_TOTAL:
    end = min(start + CHUNK, N_TOTAL)
    ranges.append((start, end))
    start = end

part_files = []
for i, (lo, hi) in enumerate(ranges, 1):
    out_csv_gz = OUT_DIR / f"gaia_dr3_sample_part{i:02d}.csv.gz"
    fetch_chunk(lo, hi, out_csv_gz)
    part_files.append(out_csv_gz)

# ---- объединяем в один CSV (по желанию) ----
# При 2 млн строк получится файл порядка сотен мегабайт.
final_csv = OUT_DIR / "gaia_dr3_sample_2M.csv.gz"
frames = (pd.read_csv(p) for p in part_files)
pd.concat(frames, ignore_index=True).to_csv(final_csv, index=False, compression="gzip")
print("DONE ->", final_csv)
