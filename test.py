import mne
import numpy as np
from scipy.signal import welch

# 파일 경로
file_path = r"C:\Users\henry\Desktop\shhs1-200001.edf"

# EDF 파일 읽기
raw = mne.io.read_raw_edf(file_path, preload=True)

# SaO2 채널 데이터 추출
spo2_data, times = raw.get_data(picks='SaO2', return_times=True)
spo2_data = spo2_data.flatten()

# 샘플링 주파수 확인
sfreq = raw.info['sfreq']
print(f"샘플링 주파수: {sfreq} Hz")

# ✅ 음수 값 제거 (0 미만은 NaN)
spo2_data = np.where(spo2_data < 0, np.nan, spo2_data)
spo2_data_clean = spo2_data[~np.isnan(spo2_data)]

# ✅ 다운샘플링 (125Hz → 1Hz 평균)
factor = int(sfreq // 1)  # 1Hz로 변환
spo2_downsampled = spo2_data_clean[:len(spo2_data_clean) // factor * factor].reshape(-1, factor).mean(axis=1)

# 1️⃣ 평균 SpO2
mean_spo2 = np.mean(spo2_downsampled)

# 2️⃣ 표준편차 (SD)
std_spo2 = np.std(spo2_downsampled)

# 3️⃣ 최소값 (Min)
min_spo2 = np.min(spo2_downsampled)

# 4️⃣ 최대 감소폭 (Max Drop)
max_drop = np.max(spo2_downsampled) - np.min(spo2_downsampled)

# 5️⃣ ODI 4%: 전 구간에서 4% 이상 떨어진 이벤트 탐지
odi_4 = 0
for i in range(1, len(spo2_downsampled)):
    if (spo2_downsampled[i - 1] - spo2_downsampled[i]) >= 4:
        odi_4 += 1
odi_4_per_hr = odi_4 / (len(spo2_downsampled) / 3600)  # 시간당 이벤트 수

# 6️⃣ ODI 3%
odi_3 = 0
for i in range(1, len(spo2_downsampled)):
    if (spo2_downsampled[i - 1] - spo2_downsampled[i]) >= 3:
        odi_3 += 1
odi_3_per_hr = odi_3 / (len(spo2_downsampled) / 3600)

# 7️⃣ 저산소 지속시간 (SpO2 < 90%)
hypoxic_duration_sec = np.sum(spo2_downsampled < 90)  # 이미 1Hz이므로 초 단위

# 8️⃣ 저주파 파워 (0~0.04 Hz)
f, Pxx = welch(spo2_downsampled, fs=1.0)  # 1Hz로 다운샘플링 했으므로 fs=1
low_freq_power = np.sum(Pxx[(f >= 0) & (f <= 0.04)])

# 9️⃣ 순환 주기 길이 (가장 강한 주기)
dominant_freq = f[np.argmax(Pxx)]
if dominant_freq != 0:
    cycle_length_sec = 1 / dominant_freq
else:
    cycle_length_sec = np.nan  # 주기가 없음

# 🔟 급강하 회복 속도
diffs = np.diff(spo2_downsampled)
recovery_speeds = diffs[diffs > 0]
if len(recovery_speeds) > 0:
    avg_recovery_speed = np.mean(recovery_speeds)
else:
    avg_recovery_speed = 0

# ✅ 결과 출력
print("\n📊 특징 추출 결과 (다운샘플링 적용)")
print(f"1️⃣ 평균 SpO2: {mean_spo2:.2f}")
print(f"2️⃣ 표준편차: {std_spo2:.2f}")
print(f"3️⃣ 최소값: {min_spo2:.2f}")
print(f"4️⃣ 최대 감소폭: {max_drop:.2f}")
print(f"5️⃣ ODI 4%: {odi_4_per_hr:.2f} /hr")
print(f"6️⃣ ODI 3%: {odi_3_per_hr:.2f} /hr")
print(f"7️⃣ 저산소 지속시간: {hypoxic_duration_sec/60:.2f} 분")
print(f"8️⃣ 저주파 파워: {low_freq_power:.4f}")
print(f"9️⃣ 순환 주기 길이: {cycle_length_sec:.2f} 초")
print(f"🔟 급강하 회복 속도: {avg_recovery_speed:.2f} (SpO2/초)")
