import os
import joblib
import numpy as np
import pandas as pd
from scipy import signal as sps
import librosa


# =========================================================
# 1. POS rPPG
# =========================================================

def pos_algorithm(rgb_signal, fps):
    """
    Plane-Orthogonal-to-Skin (POS) rPPG extraction.

    rgb_signal:
        Nx3 array containing mean [R, G, B] values from the
        forehead region.

    fps:
        Actual number of processed frames per second.
    """

    rgb_signal = np.asarray(rgb_signal, dtype=np.float64)

    if rgb_signal.ndim != 2 or rgb_signal.shape[1] != 3:
        return np.zeros(len(rgb_signal), dtype=np.float64)

    n = len(rgb_signal)

    if fps <= 0:
        return np.zeros(n, dtype=np.float64)

    window_len = max(int(round(fps * 1.6)), 8)

    if n < window_len:
        return np.zeros(n, dtype=np.float64)

    pulse = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)

    for start in range(0, n - window_len + 1):

        window = rgb_signal[start:start + window_len].copy()

        mean_rgb = np.mean(window, axis=0)
        mean_rgb[mean_rgb <= 1e-8] = 1e-8

        normalized = window / mean_rgb

        # POS projection
        Xs = (
            3.0 * normalized[:, 0]
            - 2.0 * normalized[:, 1]
        )

        Ys = (
            1.5 * normalized[:, 0]
            + normalized[:, 1]
            - 1.5 * normalized[:, 2]
        )

        std_x = np.std(Xs)
        std_y = np.std(Ys)

        if std_y <= 1e-8:
            continue

        alpha = std_x / std_y

        S = Xs - alpha * Ys

        S -= np.mean(S)

        pulse[start:start + window_len] += S
        weights[start:start + window_len] += 1.0

    pulse = np.divide(
        pulse,
        weights,
        out=np.zeros_like(pulse),
        where=weights > 0
    )

    return pulse


# =========================================================
# 2. Bandpass Filter
# =========================================================

def bandpass_filter(
    sig,
    fps,
    low_hz=0.75,
    high_hz=3.0,
    order=3
):
    """
    Bandpass for approximately 45–180 BPM.
    """

    sig = np.asarray(sig, dtype=np.float64)

    n = len(sig)

    if n < 15 or fps <= 0:
        if n > 0:
            return sig - np.mean(sig)
        return sig

    nyq = fps / 2.0

    if nyq <= low_hz:
        return sig - np.mean(sig)

    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)

    if low >= high:
        return sig - np.mean(sig)

    b, a = sps.butter(
        order,
        [low, high],
        btype="band"
    )

    padlen = min(
        n - 1,
        3 * max(len(a), len(b))
    )

    try:
        return sps.filtfilt(
            b,
            a,
            sig,
            padlen=padlen
        )
    except Exception:
        return sig - np.mean(sig)


# =========================================================
# 3. rPPG Signal Quality + HR + HRV
# =========================================================

def _calculate_rppg_quality(
    pulse_signal,
    fps,
    peaks=None
):
    """
    Estimate rPPG quality using:

    1. Spectral peak concentration
    2. Valid IBI ratio

    Returns a value between 0 and 1.
    """

    pulse_signal = np.asarray(
        pulse_signal,
        dtype=np.float64
    )

    n = len(pulse_signal)

    if (
        n < 20
        or fps <= 0
        or not np.all(np.isfinite(pulse_signal))
    ):
        return 0.0

    std = np.std(pulse_signal)

    if std < 1e-8:
        return 0.0

    try:
        nperseg = min(
            n,
            max(int(round(fps * 6)), 32)
        )

        freqs, psd = sps.welch(
            pulse_signal - np.mean(pulse_signal),
            fs=fps,
            nperseg=nperseg
        )

        valid = (
            (freqs >= 0.75)
            & (freqs <= 3.0)
        )

        if not np.any(valid):
            return 0.0

        valid_freqs = freqs[valid]
        valid_psd = psd[valid]

        total_power = np.sum(valid_psd)

        if total_power <= 1e-12:
            spectral_quality = 0.0
        else:
            peak_idx = np.argmax(valid_psd)

            peak_power = valid_psd[peak_idx]

            # Power concentration around dominant peak
            peak_freq = valid_freqs[peak_idx]

            neighborhood = (
                np.abs(valid_freqs - peak_freq)
                <= 0.15
            )

            neighborhood_power = np.sum(
                valid_psd[neighborhood]
            )

            spectral_quality = (
                neighborhood_power
                / total_power
            )

            spectral_quality = float(
                np.clip(
                    spectral_quality,
                    0.0,
                    1.0
                )
            )

    except Exception:
        spectral_quality = 0.0

    # -----------------------------------------------------
    # IBI quality
    # -----------------------------------------------------

    ibi_quality = 0.0

    if peaks is not None and len(peaks) >= 3:

        ibi_ms = (
            np.diff(peaks)
            / fps
            * 1000.0
        )

        valid_ibi = (
            (ibi_ms >= 333.0)
            & (ibi_ms <= 1333.0)
            & np.isfinite(ibi_ms)
        )

        if len(ibi_ms) > 0:
            ibi_quality = float(
                np.mean(valid_ibi)
            )

    # Spectral quality is more stable for HR.
    # IBI quality is used as supporting evidence.
    quality = (
        0.65 * spectral_quality
        + 0.35 * ibi_quality
    )

    return float(
        np.clip(
            quality,
            0.0,
            1.0
        )
    )


def estimate_hr_and_hrv(
    pulse_signal,
    fps,
    quality_threshold=0.35
):
    """
    Estimate HR and RMSSD from rPPG.

    Important:
    We do NOT return fake values such as 72 BPM / 45 ms
    when the signal is unusable.

    Low-quality measurements return None.
    """

    pulse_signal = np.asarray(
        pulse_signal,
        dtype=np.float64
    )

    n = len(pulse_signal)

    result = {
        "hr_bpm": None,
        "rmssd_ms": None,
        "rppg_signal_quality": 0.0,
    }

    if (
        fps <= 0
        or n < int(fps * 5)
        or np.std(pulse_signal) < 1e-8
    ):
        return result

    # -----------------------------------------------------
    # Frequency-domain HR
    # -----------------------------------------------------

    try:

        nperseg = min(
            n,
            max(int(round(fps * 8)), 32)
        )

        freqs, psd = sps.welch(
            pulse_signal - np.mean(pulse_signal),
            fs=fps,
            nperseg=nperseg
        )

        valid = (
            (freqs >= 0.75)
            & (freqs <= 3.0)
        )

        if not np.any(valid):
            return result

        valid_freqs = freqs[valid]
        valid_psd = psd[valid]

        peak_idx = np.argmax(valid_psd)

        peak_freq = float(
            valid_freqs[peak_idx]
        )

        hr_bpm = peak_freq * 60.0

    except Exception:
        return result

    # -----------------------------------------------------
    # Peak detection
    # -----------------------------------------------------

    pulse_std = np.std(pulse_signal)

    expected_distance = max(
        int(fps / max(peak_freq, 0.01) * 0.75),
        1
    )

    try:

        peaks, properties = sps.find_peaks(
            pulse_signal,
            distance=expected_distance,
            prominence=max(
                0.20 * pulse_std,
                1e-8
            )
        )

    except Exception:
        peaks = np.array([])

    quality = _calculate_rppg_quality(
        pulse_signal,
        fps,
        peaks
    )

    result["rppg_signal_quality"] = round(
        float(quality),
        3
    )

    # -----------------------------------------------------
    # Reject unreliable signal
    # -----------------------------------------------------

    if quality < quality_threshold:

        print(
            "[RPPG] Low signal quality:",
            round(quality, 3),
            "-> HR/HRV rejected"
        )

        return result

    # -----------------------------------------------------
    # Validate HR
    # -----------------------------------------------------

    if not (
        45.0
        <= hr_bpm
        <= 180.0
    ):
        return result

    result["hr_bpm"] = round(
        float(hr_bpm),
        1
    )

    # -----------------------------------------------------
    # HRV / RMSSD
    # -----------------------------------------------------

    if len(peaks) < 4:
        return result

    ibi_ms = (
        np.diff(peaks)
        / fps
        * 1000.0
    )

    if len(ibi_ms) < 3:
        return result

    # Physiologically reasonable IBI range
    valid_mask = (
        (ibi_ms >= 333.0)
        & (ibi_ms <= 1333.0)
        & np.isfinite(ibi_ms)
    )

    ibi_ms = ibi_ms[valid_mask]

    if len(ibi_ms) < 3:
        return result

    # Robust outlier rejection
    median_ibi = np.median(ibi_ms)

    if median_ibi <= 0:
        return result

    clean_ibi = ibi_ms[
        (ibi_ms >= 0.80 * median_ibi)
        & (ibi_ms <= 1.20 * median_ibi)
    ]

    if len(clean_ibi) < 3:
        return result

    successive_differences = np.diff(
        clean_ibi
    )

    rmssd_ms = np.sqrt(
        np.mean(
            successive_differences ** 2
        )
    )

    if not np.isfinite(rmssd_ms):
        return result

    result["rmssd_ms"] = round(
        float(
            np.clip(
                rmssd_ms,
                0.0,
                150.0
            )
        ),
        1
    )

    return result


# =========================================================
# 4. Eye Aspect Ratio
# =========================================================

def eye_aspect_ratio(eye_pts):

    eye_pts = np.asarray(
        eye_pts,
        dtype=np.float64
    )

    if eye_pts.shape[0] != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_pts

    horizontal = np.linalg.norm(
        p1 - p4
    )

    if horizontal <= 1e-8:
        return 0.0

    vertical_1 = np.linalg.norm(
        p2 - p6
    )

    vertical_2 = np.linalg.norm(
        p3 - p5
    )

    return float(
        (
            vertical_1
            + vertical_2
        )
        / (
            2.0 * horizontal
        )
    )


# =========================================================
# 5. Blink Counter
# =========================================================

class BlinkCounter:

    def __init__(
        self,
        threshold=0.21,
        consec_frames=2
    ):

        self.threshold = threshold
        self.consec_frames = consec_frames

        self._below_count = 0
        self.blink_count = 0

    def update(self, ear_value):

        if not np.isfinite(ear_value):
            return

        if ear_value < self.threshold:

            self._below_count += 1

        else:

            if (
                self._below_count
                >= self.consec_frames
            ):
                self.blink_count += 1

            self._below_count = 0

    def finalize(self):

        if (
            self._below_count
            >= self.consec_frames
        ):
            self.blink_count += 1

        self._below_count = 0

        return self.blink_count


# =========================================================
# 6. Voice Stress
# =========================================================

def extract_voice_stress_features(
    audio_data,
    sr=22050
):
    """
    Robust pYIN voice extraction.

    The old implementation accepted every non-NaN F0 value.
    That allowed octave/outlier errors to produce values such as
    pitch_std = 181 Hz.

    This version:
      - restricts F0 to 65–400 Hz
      - uses voiced probability
      - removes NaNs
      - removes extreme 5th/95th percentile outliers
      - uses median pitch
      - uses standard deviation of cleaned F0
    """

    neutral_result = {
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "spectral_centroid_hz": 0.0,
        "vocal_stress_subscore": 50.0,
    }

    if audio_data is None:
        return neutral_result

    audio_data = np.asarray(
        audio_data,
        dtype=np.float64
    ).flatten()

    if len(audio_data) < sr * 1:
        return neutral_result

    if not np.any(np.isfinite(audio_data)):
        return neutral_result

    audio_data = np.nan_to_num(
        audio_data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    max_amp = np.max(
        np.abs(audio_data)
    )

    if max_amp <= 1e-8:
        return neutral_result

    audio_data = (
        audio_data
        / max_amp
    )

    # -----------------------------------------------------
    # pYIN
    # -----------------------------------------------------

    try:

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio_data,
            fmin=65.0,
            fmax=400.0,
            sr=sr,
            frame_length=2048,
            hop_length=256
        )

    except Exception as exc:

        print(
            "[VOICE] pYIN error:",
            exc
        )

        return neutral_result

    if f0 is None:
        return neutral_result

    f0 = np.asarray(
        f0,
        dtype=np.float64
    )

    voiced_flag = np.asarray(
        voiced_flag,
        dtype=bool
    )

    voiced_prob = np.asarray(
        voiced_prob,
        dtype=np.float64
    )

    mask = (
        voiced_flag
        & np.isfinite(f0)
        & np.isfinite(voiced_prob)
        & (voiced_prob >= 0.10)
        & (f0 >= 65.0)
        & (f0 <= 400.0)
    )

    voiced_f0 = f0[mask]

    if len(voiced_f0) < 6:
        return neutral_result

    # -----------------------------------------------------
    # Remove octave / extreme outliers
    # -----------------------------------------------------

    low = np.percentile(
        voiced_f0,
        5
    )

    high = np.percentile(
        voiced_f0,
        95
    )

    voiced_f0 = voiced_f0[
        (voiced_f0 >= low)
        & (voiced_f0 <= high)
    ]

    if len(voiced_f0) < 5:
        return neutral_result

    pitch_mean = float(
        np.median(voiced_f0)
    )

    pitch_std = float(
        np.std(voiced_f0)
    )

    # Prevent one remaining numerical outlier
    pitch_std = float(
        np.clip(
            pitch_std,
            0.0,
            100.0
        )
    )

    # -----------------------------------------------------
    # Spectral centroid
    # -----------------------------------------------------

    try:

        spec_cent = librosa.feature.spectral_centroid(
            y=audio_data,
            sr=sr
        )

        mean_spec_cent = float(
            np.mean(spec_cent)
        )

    except Exception:

        mean_spec_cent = 0.0

    # -----------------------------------------------------
    # Voice stress score
    #
    # pitch variation:
    # 15–45 Hz -> 0–100
    #
    # spectral brightness:
    # 1200–2800 Hz -> 0–100
    # -----------------------------------------------------

    s_pitch_var = np.clip(
        (
            (pitch_std - 15.0)
            / (45.0 - 15.0)
        )
        * 100.0,
        0.0,
        100.0
    )

    s_spectral = np.clip(
        (
            (mean_spec_cent - 1200.0)
            / (2800.0 - 1200.0)
        )
        * 100.0,
        0.0,
        100.0
    )

    vocal_subscore = (
        0.60 * s_pitch_var
        + 0.40 * s_spectral
    )

    return {
        "pitch_mean_hz": round(
            pitch_mean,
            1
        ),
        "pitch_std_hz": round(
            pitch_std,
            1
        ),
        "spectral_centroid_hz": round(
            mean_spec_cent,
            1
        ),
        "vocal_stress_subscore": round(
            float(vocal_subscore),
            1
        ),
    }


# =========================================================
# 7. Lifestyle Model
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "lifestyle_stress_model.joblib"
)

_lifestyle_model = None


def _load_lifestyle_model():

    global _lifestyle_model

    if _lifestyle_model is None:

        if not os.path.exists(MODEL_PATH):
            print(
                "[LIFESTYLE] Model not found:",
                MODEL_PATH
            )
            return None

        try:

            _lifestyle_model = joblib.load(
                MODEL_PATH
            )

            print(
                "[LIFESTYLE] Model loaded:",
                MODEL_PATH
            )

        except Exception as exc:

            print(
                "[LIFESTYLE] Model loading error:",
                exc
            )

            _lifestyle_model = None

    return _lifestyle_model


# =========================================================
# 8. Time Parsing
# =========================================================

def parse_time_to_hours(time_str):

    if time_str is None or time_str == "":
        return 7.0

    if isinstance(
        time_str,
        (int, float)
    ):
        return float(time_str)

    time_str = str(
        time_str
    ).strip()

    dt = pd.to_datetime(
        time_str,
        format="%I:%M %p",
        errors="coerce"
    )

    if pd.isna(dt):

        dt = pd.to_datetime(
            time_str,
            format="%H:%M",
            errors="coerce"
        )

    if pd.isna(dt):
        return 7.0

    return (
        dt.hour
        + dt.minute / 60.0
    )


# =========================================================
# 9. Stress Score
# =========================================================

def score_stress(
    video_features: dict,
    survey_data: dict
) -> dict:

    field_mapping = {

        "age": "Age",
        "gender": "Gender",

        "sleep_duration": "Sleep_Duration",
        "sleep_hours_per_night": "Sleep_Duration",

        "sleep_quality": "Sleep_Quality",

        "wake_up_time": "Wake_Up_Time",
        "bed_time": "Bed_Time",

        "physical_activity_hours_daily":
            "Physical_Activity",

        "daily_screen_time_hours":
            "Screen_Time",

        "caffeinated_drinks_per_day":
            "Caffeine_Intake",

        "alcoholic_drinks_per_day":
            "Alcohol_Intake",

        "smokes":
            "Smoking_Habit",

        "avg_work_hours_per_day":
            "Work_Hours",

        "daily_commute_hours":
            "Travel_Time",

        "social_activity_hours_per_day":
            "Social_Interactions",

        "meditates_regularly":
            "Meditation_Practice",

        "preferred_exercise_type":
            "Exercise_Type",

        # Older/internal names

        "physical_activity":
            "Physical_Activity",

        "screen_time":
            "Screen_Time",

        "caffeine_intake":
            "Caffeine_Intake",

        "alcohol_intake":
            "Alcohol_Intake",

        "smoking_habit":
            "Smoking_Habit",

        "work_hours":
            "Work_Hours",

        "travel_time":
            "Travel_Time",

        "social_interactions":
            "Social_Interactions",

        "meditation_practice":
            "Meditation_Practice",

        "exercise_type":
            "Exercise_Type",
    }

    mapped_survey = {}

    for key, value in survey_data.items():

        if key == "session_id":
            continue

        mapped_key = field_mapping.get(
            key.lower(),
            key
        )

        if mapped_key in (
            "Wake_Up_Time",
            "Bed_Time"
        ):
            value = parse_time_to_hours(
                value
            )

        mapped_survey[
            mapped_key
        ] = value

    # =====================================================
    # BIOMETRIC SCORE
    # =====================================================

    rmssd = video_features.get(
        "rmssd_ms"
    )

    if rmssd is None:
        rmssd = 45.0

    rmssd = float(rmssd)

    # Improved HRV range:
    # 20–150 ms

    s_hrv = (
        1.0
        -
        (
            np.clip(
                rmssd,
                20.0,
                150.0
            )
            - 20.0
        )
        / 130.0
    ) * 100.0

    # -----------------------------------------------------
    # Voice
    # -----------------------------------------------------

    vocal_subscore = video_features.get(
        "vocal_stress_subscore"
    )

    if vocal_subscore is None:

        pitch_std = float(
            video_features.get(
                "pitch_std_hz",
                5.0
            )
        )

        spectral_centroid = float(
            video_features.get(
                "spectral_centroid_hz",
                0.0
            )
        )

        s_pitch_var = np.clip(
            (
                (pitch_std - 15.0)
                / 30.0
            )
            * 100.0,
            0.0,
            100.0
        )

        s_spectral = np.clip(
            (
                (spectral_centroid - 1200.0)
                / 1600.0
            )
            * 100.0,
            0.0,
            100.0
        )

        vocal_subscore = (
            0.60 * s_pitch_var
            + 0.40 * s_spectral
        )

    s_voice = float(
        np.clip(
            vocal_subscore,
            0.0,
            100.0
        )
    )

    # -----------------------------------------------------
    # Behaviour
    # -----------------------------------------------------

    blink_rate = float(
        video_features.get(
            "blink_rate_bpm",
            18.0
        )
    )

    brow_ratio = float(
        video_features.get(
            "brow_ratio",
            0.22
        )
    )

    # Blink stress
    s_blink = np.clip(
        (
            (blink_rate - 14.0)
            / (32.0 - 14.0)
        )
        * 100.0,
        0.0,
        100.0
    )

    # Brow tension
    s_brow = np.clip(
        (
            (0.22 - brow_ratio)
            / (0.22 - 0.14)
        )
        * 100.0,
        0.0,
        100.0
    )

    s_behavior = (
        0.60 * s_blink
        + 0.40 * s_brow
    )

    # -----------------------------------------------------
    # Biometric fusion
    #
    # HRV     = 40%
    # Voice   = 30%
    # Behavior= 30%
    # -----------------------------------------------------

    biometric_score = (
        0.40 * s_hrv
        + 0.30 * s_voice
        + 0.30 * s_behavior
    )

    biometric_score = float(
        np.clip(
            biometric_score,
            0.0,
            100.0
        )
    )

    # =====================================================
    # LIFESTYLE SCORE
    # =====================================================

    model = _load_lifestyle_model()

    if model is not None:

        try:

            df_in = pd.DataFrame(
                [mapped_survey]
            )

            lifestyle_score = float(
                model.predict(df_in)[0]
            )

            lifestyle_score = float(
                np.clip(
                    lifestyle_score,
                    0.0,
                    100.0
                )
            )

        except Exception as exc:

            print(
                "[LIFESTYLE] Prediction error:",
                exc
            )

            lifestyle_score = 50.0

    else:

        lifestyle_score = 50.0

    # =====================================================
    # FINAL FUSION
    #
    # Biometric = 55%
    # Lifestyle  = 45%
    # =====================================================

    final_score = round(
        (
            0.55 * biometric_score
            + 0.45 * lifestyle_score
        ),
        1
    )

    stress_prob = round(
        final_score / 100.0,
        2
    )

    is_critical = (
        final_score >= 65.0
    )

    # =====================================================
    # Classification
    # =====================================================

    classification = (
        "Critical Fatigue"
        if is_critical
        else "Cleared"
    )

    readiness_status = (
        "Mandatory Rest Required"
        if is_critical
        else "Fit for Duty"
    )

    # =====================================================
    # Insights
    # =====================================================

    key_insights = []
    recommendations = []

    if rmssd < 30.0:

        key_insights.append(
            f"Low HRV measurement "
            f"(RMSSD: {rmssd:.1f} ms)."
        )

        recommendations.append(
            "Consider a short recovery/breathing period."
        )

    if blink_rate > 25.0:

        key_insights.append(
            f"Elevated blink rate "
            f"({blink_rate:.1f}/min)."
        )

    sleep_hours = mapped_survey.get(
        "Sleep_Duration",
        8.0
    )

    try:
        sleep_hours = float(
            sleep_hours
        )
    except (
        TypeError,
        ValueError
    ):
        sleep_hours = 8.0

    if sleep_hours < 6.5:

        key_insights.append(
            f"Reported sleep duration "
            f"is {sleep_hours:.1f} hours."
        )

        recommendations.append(
            "Prioritize adequate sleep and recovery."
        )

    if not key_insights:

        key_insights.append(
            "No major deviations were detected "
            "in the measured features."
        )

        recommendations.append(
            "Maintain the existing recovery routine."
        )

    # =====================================================
    # Attribution
    # =====================================================

    shap_attribution = [

        {
            "feature": "biometric_score",
            "description":
                f"Multimodal biometric score: "
                f"{biometric_score:.1f}/100",
        },

        {
            "feature": "lifestyle_score",
            "description":
                f"Lifestyle score: "
                f"{lifestyle_score:.1f}/100",
        },

    ]

    # =====================================================
    # Result
    # =====================================================

    return {

        "stress_probability":
            stress_prob,

        "classification":
            classification,

        "readiness_status":
            readiness_status,

        "biometric_score":
            round(
                biometric_score,
                1
            ),

        "lifestyle_score":
            round(
                lifestyle_score,
                1
            ),

        "final_score":
            final_score,

        "subscore_breakdown": {

            "hrv_subscore":
                round(
                    float(s_hrv),
                    2
                ),

            "voice_subscore":
                round(
                    float(s_voice),
                    2
                ),

            "behavior_subscore":
                round(
                    float(s_behavior),
                    2
                ),

        },

        "shap_attribution":
            shap_attribution,

        "key_insights":
            key_insights,

        "recommendations":
            recommendations,
    }