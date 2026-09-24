"""
Chitta-Shanti video/audio processing.

This module:
    - reads uploaded video files
    - processes every available frame
    - uses actual source FPS
    - runs MediaPipe FaceMesh
    - extracts forehead rPPG
    - calculates HR / HRV
    - calculates blink rate
    - calculates brow ratio
    - calculates normalized head jitter
    - extracts audio through FFmpeg
    - calculates robust pYIN voice features

This file does NOT define API routes.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import librosa
import subprocess

from pipelines.pipeline_utils import (
    eye_aspect_ratio,
    pos_algorithm,
    bandpass_filter,
    estimate_hr_and_hrv,
    BlinkCounter,
    extract_voice_stress_features,
)


# =========================================================
# MediaPipe landmarks
# =========================================================

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,
]

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380,
]

LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336

NOSE_TIP = 1


# =========================================================
# Helpers
# =========================================================

def get_pts(
    landmarks,
    indices,
    w,
    h
):

    return np.array(
        [
            [
                landmarks[i].x * w,
                landmarks[i].y * h
            ]
            for i in indices
        ],
        dtype=np.float64
    )


def _get_forehead_roi(
    rgb_frame,
    landmarks,
    w,
    h
):
    """
    Face-relative forehead ROI.

    This is preferable to using a fixed screen region because
    the subject can move slightly inside the frame.
    """

    xs = np.array(
        [
            lm.x * w
            for lm in landmarks
        ]
    )

    ys = np.array(
        [
            lm.y * h
            for lm in landmarks
        ]
    )

    face_min_x = float(
        np.min(xs)
    )

    face_max_x = float(
        np.max(xs)
    )

    face_min_y = float(
        np.min(ys)
    )

    face_max_y = float(
        np.max(ys)
    )

    face_width = (
        face_max_x
        - face_min_x
    )

    face_height = (
        face_max_y
        - face_min_y
    )

    if (
        face_width <= 1
        or face_height <= 1
    ):
        return None

    # Central upper-face region
    x1 = int(
        face_min_x
        + 0.30 * face_width
    )

    x2 = int(
        face_min_x
        + 0.70 * face_width
    )

    y1 = int(
        face_min_y
        + 0.08 * face_height
    )

    y2 = int(
        face_min_y
        + 0.28 * face_height
    )

    x1 = max(
        0,
        min(x1, w - 1)
    )

    x2 = max(
        0,
        min(x2, w)
    )

    y1 = max(
        0,
        min(y1, h - 1)
    )

    y2 = max(
        0,
        min(y2, h)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    roi = rgb_frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    return roi


# =========================================================
# Main Video Processing
# =========================================================

def run_opencv_processing(
    temp_path: str,
    challenge_sequence=None
):
    """
    Process the uploaded video.

    Returns:

        hr_bpm
        rmssd_ms
        rppg_signal_quality

        blink_rate
        blink_count

        brow_ratio
        head_jitter

        ear_mean
        ear_min
        ear_threshold

        face_detection_rate
        fps
        duration_sec
        frame_count
    """

    print(
        "[CHITTA] RUNNING video_processing FROM:",
        __file__
    )

    cap = cv2.VideoCapture(
        temp_path
    )

    if not cap.isOpened():

        raise ValueError(
            "Could not read uploaded video file."
        )

    # -----------------------------------------------------
    # Get ACTUAL source properties
    # -----------------------------------------------------

    source_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames_metadata = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    video_duration_metadata = (
        total_frames_metadata / source_fps
        if source_fps > 0
        else 0.0
    )

    # Some codecs report invalid FPS.
    if (
        not np.isfinite(source_fps)
        or source_fps <= 1.0
        or source_fps > 240.0
    ):
        source_fps = 30.0

    # -----------------------------------------------------
    # Buffers
    # -----------------------------------------------------

    rgb_means = []

    brow_ratios = []

    head_positions = []

    ear_values = []

    timestamps = []

    blink_counter = BlinkCounter(
        threshold=0.21,
        consec_frames=2
    )

    processed_frames = 0
    detected_face_frames = 0

    frame_idx = 0

    # -----------------------------------------------------
    # MediaPipe compatibility
    # -----------------------------------------------------

    if not hasattr(
        mp,
        "solutions"
    ):

        cap.release()

        raise RuntimeError(
            "MediaPipe FaceMesh is unavailable. "
            "Install MediaPipe 0.10.21 with Python 3.10/3.11 "
            "for this Chitta-Shanti pipeline."
        )

    if not hasattr(
        mp.solutions,
        "face_mesh"
    ):

        cap.release()

        raise RuntimeError(
            "mediapipe.solutions.face_mesh is unavailable."
        )

    mp_face_mesh = (
        mp.solutions.face_mesh
    )

    # -----------------------------------------------------
    # FaceMesh
    # -----------------------------------------------------

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:
                break

            processed_frames += 1

            # -------------------------------------------------
            # Resize only spatially.
            #
            # IMPORTANT:
            # We DO NOT skip frames.
            # -------------------------------------------------

            h, w = frame.shape[:2]

            if w > 640:

                scale = (
                    640.0 / w
                )

                new_w = 640

                new_h = max(
                    1,
                    int(
                        h * scale
                    )
                )

                frame = cv2.resize(
                    frame,
                    (
                        new_w,
                        new_h
                    )
                )

                h, w = frame.shape[:2]

            # -------------------------------------------------
            # Timestamp based on actual video FPS
            # -------------------------------------------------

            current_time = (
                frame_idx / source_fps
            )

            frame_idx += 1

            # -------------------------------------------------
            # RGB
            # -------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(
                rgb_frame
            )

            if (
                results is None
                or not results.multi_face_landmarks
            ):
                continue

            detected_face_frames += 1

            landmarks = (
                results
                .multi_face_landmarks[0]
                .landmark
            )

            # -------------------------------------------------
            # Forehead rPPG
            # -------------------------------------------------

            forehead = _get_forehead_roi(
                rgb_frame,
                landmarks,
                w,
                h
            )

            if forehead is not None:

                rgb_mean = (
                    forehead
                    .reshape(-1, 3)
                    .mean(axis=0)
                )

                if np.all(
                    np.isfinite(
                        rgb_mean
                    )
                ):

                    rgb_means.append(
                        rgb_mean
                    )

                    timestamps.append(
                        current_time
                    )

            # -------------------------------------------------
            # EAR / Blink
            # -------------------------------------------------

            right_pts = get_pts(
                landmarks,
                RIGHT_EYE,
                w,
                h
            )

            left_pts = get_pts(
                landmarks,
                LEFT_EYE,
                w,
                h
            )

            right_ear = (
                eye_aspect_ratio(
                    right_pts
                )
            )

            left_ear = (
                eye_aspect_ratio(
                    left_pts
                )
            )

            ear = (
                right_ear
                + left_ear
            ) / 2.0

            if np.isfinite(ear):

                ear_values.append(
                    ear
                )

                blink_counter.update(
                    ear
                )

            # -------------------------------------------------
            # Brow ratio
            #
            # Distance between inner eyebrows
            # divided by face width.
            # -------------------------------------------------

            xs = np.array(
                [
                    lm.x
                    for lm in landmarks
                ]
            )

            face_width_normalized = (
                np.max(xs)
                - np.min(xs)
            )

            if (
                face_width_normalized
                > 1e-8
            ):

                left_brow = np.array(
                    [
                        landmarks[
                            LEFT_BROW_INNER
                        ].x,
                        landmarks[
                            LEFT_BROW_INNER
                        ].y,
                    ]
                )

                right_brow = np.array(
                    [
                        landmarks[
                            RIGHT_BROW_INNER
                        ].x,
                        landmarks[
                            RIGHT_BROW_INNER
                        ].y,
                    ]
                )

                brow_distance = np.linalg.norm(
                    left_brow
                    - right_brow
                )

                brow_ratio = (
                    brow_distance
                    / face_width_normalized
                )

                if np.isfinite(
                    brow_ratio
                ):

                    brow_ratios.append(
                        brow_ratio
                    )

            # -------------------------------------------------
            # Head jitter
            #
            # IMPORTANT:
            # Use MediaPipe normalized coordinates.
            #
            # This avoids the previous problem where image
            # resolution changed the jitter magnitude.
            # -------------------------------------------------

            nose = landmarks[
                NOSE_TIP
            ]

            head_positions.append(
                [
                    float(nose.x),
                    float(nose.y)
                ]
            )

    finally:

        face_mesh.close()

        cap.release()

    # =========================================================
    # Basic recording statistics
    # =========================================================

    duration_sec = (
        video_duration_metadata
        if video_duration_metadata > 0
        else (
            frame_idx / source_fps
            if source_fps > 0
            else 0.0
        )
    )

    if duration_sec <= 0:

        duration_sec = (
            timestamps[-1]
            if timestamps
            else 0.0
        )

    face_detection_rate = (
        detected_face_frames
        / max(processed_frames, 1)
    )

    # =========================================================
    # rPPG
    # =========================================================

    hr_bpm = None
    rmssd_ms = None
    rppg_signal_quality = 0.0

    if len(rgb_means) >= max(
        30,
        int(source_fps * 5)
    ):

        try:

            rgb_array = np.asarray(
                rgb_means,
                dtype=np.float64
            )

            # -------------------------------------------------
            # IMPORTANT:
            # Use actual source FPS.
            # -------------------------------------------------

            pulse = pos_algorithm(
                rgb_array,
                source_fps
            )

            filtered = bandpass_filter(
                pulse,
                source_fps
            )

            hrv_result = (
                estimate_hr_and_hrv(
                    filtered,
                    source_fps,
                    quality_threshold=0.35
                )
            )

            hr_bpm = hrv_result.get(
                "hr_bpm"
            )

            rmssd_ms = hrv_result.get(
                "rmssd_ms"
            )

            rppg_signal_quality = float(
                hrv_result.get(
                    "rppg_signal_quality",
                    0.0
                )
            )

        except Exception as exc:

            print(
                "[RPPG ERROR]",
                repr(exc)
            )

    else:

        print(
            "[RPPG] Not enough face frames:",
            len(rgb_means)
        )

    # =========================================================
    # Blink
    # =========================================================

    total_blinks = (
        blink_counter.finalize()
    )

    if duration_sec > 0:

        blink_rate = (
            total_blinks
            / duration_sec
            * 60.0
        )

    else:

        blink_rate = 0.0

    # =========================================================
    # Brow
    # =========================================================

    if brow_ratios:

        brow_ratio = float(
            np.median(
                brow_ratios
            )
        )

    else:

        brow_ratio = 0.22

    # =========================================================
    # Head jitter
    # =========================================================

    if len(head_positions) >= 2:

        head_array = np.asarray(
            head_positions,
            dtype=np.float64
        )

        # Standard deviation of normalized x/y coordinates.
        # This gives values in a resolution-independent scale.

        head_jitter = float(
            np.std(
                head_array
            )
        )

    else:

        head_jitter = 0.0

    # =========================================================
    # EAR statistics
    # =========================================================

    if ear_values:

        ear_mean = float(
            np.mean(
                ear_values
            )
        )

        ear_min = float(
            np.min(
                ear_values
            )
        )

    else:

        ear_mean = 0.0
        ear_min = 0.0

    # =========================================================
    # Quality
    # =========================================================

    quality_ok = (
        processed_frames > 0
        and face_detection_rate >= 0.70
    )

    # =========================================================
    # Debug output
    # =========================================================

    print(
        "[VIDEO DEBUG]",
        {
            "processed_frames":
                processed_frames,

            "detected_face_frames":
                detected_face_frames,

            "face_detection_rate":
                round(
                    face_detection_rate,
                    3
                ),

            "source_fps":
                round(
                    source_fps,
                    3
                ),

            "duration_sec":
                round(
                    duration_sec,
                    3
                ),

            "hr_bpm":
                hr_bpm,

            "rmssd_ms":
                rmssd_ms,

            "rppg_signal_quality":
                round(
                    rppg_signal_quality,
                    3
                ),

            "blink_count":
                total_blinks,

            "blink_rate":
                round(
                    blink_rate,
                    2
                ),

            "ear_mean":
                round(
                    ear_mean,
                    4
                ),

            "ear_min":
                round(
                    ear_min,
                    4
                ),

            "brow_ratio":
                round(
                    brow_ratio,
                    4
                ),

            "head_jitter":
                round(
                    head_jitter,
                    5
                ),
        }
    )

    # =========================================================
    # Final result
    # =========================================================

    return {

        "frame_count":
            processed_frames,

        "hr_bpm":
            (
                round(
                    float(hr_bpm),
                    2
                )
                if hr_bpm is not None
                else None
            ),

        "rmssd_ms":
            (
                round(
                    float(rmssd_ms),
                    2
                )
                if rmssd_ms is not None
                else None
            ),

        "rppg_signal_quality":
            round(
                rppg_signal_quality,
                3
            ),

        "blink_rate":
            round(
                float(blink_rate),
                2
            ),

        "blink_count":
            int(total_blinks),

        "brow_ratio":
            round(
                float(brow_ratio),
                4
            ),

        "head_jitter":
            round(
                float(head_jitter),
                5
            ),

        "ear_mean":
            round(
                float(ear_mean),
                4
            ),

        "ear_min":
            round(
                float(ear_min),
                4
            ),

        "ear_threshold":
            0.21,

        "face_detection_rate":
            round(
                float(face_detection_rate),
                3
            ),

        "fps":
            round(
                float(source_fps),
                3
            ),

        "duration_sec":
            round(
                float(duration_sec),
                3
            ),

        "quality_ok":
            bool(quality_ok),
    }


# =========================================================
# Audio Extraction
# =========================================================

def extract_audio_track(
    video_path: str,
    target_sr: int = 22050
):

    wav_path = (
        f"{video_path}.wav"
    )

    command = [

        "ffmpeg",

        "-y",

        "-i",
        video_path,

        "-vn",

        "-acodec",
        "pcm_s16le",

        "-ar",
        str(target_sr),

        "-ac",
        "1",

        wav_path,
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:

            print(
                "[AUDIO EXTRACTION ERROR]",
                result.stderr
            )

            return None

    except FileNotFoundError:

        print(
            "[AUDIO EXTRACTION ERROR] "
            "FFmpeg is not installed or not in PATH."
        )

        return None

    except subprocess.TimeoutExpired:

        print(
            "[AUDIO EXTRACTION ERROR] "
            "FFmpeg timed out."
        )

        return None

    if not os.path.exists(
        wav_path
    ):

        return None

    return wav_path


# =========================================================
# Audio Processing
# =========================================================

def run_audio_processing_from_video(
    video_path: str
):

    """
    Extract audio and run robust pYIN processing.
    """

    wav_path = extract_audio_track(
        video_path,
        target_sr=22050
    )

    if wav_path is None:

        return {

            "pitch_mean_hz": 0.0,

            "pitch_std_hz": 0.0,

            "spectral_centroid_hz": 0.0,

            "vocal_stress_subscore": 50.0,
        }

    try:

        audio_data, sr = librosa.load(
            wav_path,
            sr=22050,
            mono=True
        )

        voice_metrics = (
            extract_voice_stress_features(
                audio_data,
                sr=sr
            )
        )

        print(
            "[VOICE RESULT]",
            voice_metrics
        )

        return voice_metrics

    except Exception as exc:

        print(
            "[VOICE PROCESSING ERROR]",
            repr(exc)
        )

        return {

            "pitch_mean_hz": 0.0,

            "pitch_std_hz": 0.0,

            "spectral_centroid_hz": 0.0,

            "vocal_stress_subscore": 50.0,
        }

    finally:

        try:

            if os.path.exists(
                wav_path
            ):
                os.remove(
                    wav_path
                )

        except FileNotFoundError:

            pass