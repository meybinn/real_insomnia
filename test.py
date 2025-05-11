import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
import glob
import os

# EDF 파일들이 있는 폴더 경로
edf_folder = r"\\Msdl_nas\msdl\sleep_data\shhs\polysomnography\edfs\shhs1"

# 결과를 담을 리스트
results = []

# EDF 파일 목록 가져오기
edf_files = glob.glob(os.path.join(edf_folder, "*.edf"))

print(f"총 {len(edf_files)}개의 EDF 파일을 찾았습니다.\n")

for file_path in edf_files:
    try:
        print(f"▶ 처리 중: {os.path.basename(file_path)}")

        # EDF 파일 읽기
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')

        # SaO2 채널 데이터 추출
        spo2_data, times = raw.get_data(picks='SaO2', return_times=True)
        spo2_data = spo2_data.flatten()

        # 샘플링 주파수 확인
        sfreq = raw.info['sfreq']

        # ✅ 음수 및 0 값 제거
        spo2_data = np.where(spo2_data <= 0, np.nan, spo2_data)
        spo2_data_clean = spo2_data[~np.isnan(spo2_data)]

        # ✅ 다운샘플링 (125Hz → 1Hz)
        factor = int(sfreq // 1)
        if len(spo2_data_clean) < factor:
            print("⚠️ 데이터가 너무 짧아서 스킵됨.")
            continue

        spo2_downsampled = spo2_data_clean[:len(spo2_data_clean) // factor * factor].reshape(-1, factor).mean(axis=1)

        # 1️⃣ 평균 SpO2
        mean_spo2 = np.mean(spo2_downsampled)

        # 2️⃣ 표준편차 (SD)
        std_spo2 = np.std(spo2_downsampled)

        # 3️⃣ 최소값 (Min)
        min_spo2 = np.min(spo2_downsampled)

        # 4️⃣ 최대 감소폭 (Max Drop)
        max_drop = np.max(spo2_downsampled) - np.min(spo2_downsampled)

        # 5️⃣ ODI 4%
        odi_4 = 0
        for i in range(1, len(spo2_downsampled)):
            if (spo2_downsampled[i - 1] - spo2_downsampled[i]) >= 4:
                odi_4 += 1
        odi_4_per_hr = odi_4 / (len(spo2_downsampled) / 3600)

        # 6️⃣ ODI 3%
        odi_3 = 0
        for i in range(1, len(spo2_downsampled)):
            if (spo2_downsampled[i - 1] - spo2_downsampled[i]) >= 3:
                odi_3 += 1
        odi_3_per_hr = odi_3 / (len(spo2_downsampled) / 3600)

        # 7️⃣ 저산소 지속시간
        hypoxic_duration_sec = np.sum(spo2_downsampled < 90)

        # 8️⃣ 저주파 파워
        f, Pxx = welch(spo2_downsampled, fs=1.0)
        low_freq_power = np.sum(Pxx[(f >= 0) & (f <= 0.04)])

        # 9️⃣ 순환 주기 길이
        dominant_freq = f[np.argmax(Pxx)]
        if dominant_freq != 0:
            cycle_length_sec = 1 / dominant_freq
        else:
            cycle_length_sec = np.nan

        # 🔟 급강하 회복 속도
        diffs = np.diff(spo2_downsampled)
        recovery_speeds = diffs[diffs > 0]
        if len(recovery_speeds) > 0:
            avg_recovery_speed = np.mean(recovery_speeds)
        else:
            avg_recovery_speed = 0

        # 결과 딕셔너리 추가
        results.append({
            'filename': os.path.basename(file_path),
            'mean_spo2': mean_spo2,
            'std_spo2': std_spo2,
            'min_spo2': min_spo2,
            'max_drop': max_drop,
            'odi_4_per_hr': odi_4_per_hr,
            'odi_3_per_hr': odi_3_per_hr,
            'hypoxic_duration_min': hypoxic_duration_sec / 60,
            'low_freq_power': low_freq_power,
            'cycle_length_sec': cycle_length_sec,
            'avg_recovery_speed': avg_recovery_speed
        })

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        continue

# 결과를 데이터프레임으로 변환
df_results = pd.DataFrame(results)

# CSV로 저장
output_path = os.path.join(edf_folder, "spo2_features_summary.csv")
df_results.to_csv(output_path, index=False)

print(f"\n✅ 모든 작업 완료! 결과가 다음에 저장되었습니다:\n{output_path}")
