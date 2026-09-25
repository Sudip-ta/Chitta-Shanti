from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from datetime import (
    datetime,
    timezone,
)

from starlette.concurrency import (
    run_in_threadpool,
)

from pydantic import BaseModel

import shutil
import os
import tempfile

from typing import Optional


from database import get_db
from models_db import gen_id

from api.auth_api import (
    require_role,
    get_current_user,
    hash_identifier,
)

from pipelines.video_processing import (
    run_opencv_processing,
    run_audio_processing_from_video,
)

from pipelines.pipeline_utils import (
    score_stress,
)

from security_utils import (
    encrypt_blob,
    decrypt_blob,
)

from notifications import (
    send_critical_alert,
)


router = APIRouter()

CRITICAL_LABEL = "Critical Fatigue"


# =========================================================
# Questionnaire
# =========================================================

class QuestionnairePayload(BaseModel):

    session_id: str

    age: Optional[int] = None

    gender: Optional[str] = None

    sleep_hours_per_night: Optional[float] = None

    sleep_quality: Optional[int] = None

    wake_up_time: Optional[str] = None

    bed_time: Optional[str] = None

    physical_activity_hours_daily: Optional[float] = None

    daily_screen_time_hours: Optional[float] = None

    caffeinated_drinks_per_day: Optional[int] = None

    alcoholic_drinks_per_day: Optional[int] = None

    smokes: Optional[str] = None

    avg_work_hours_per_day: Optional[float] = None

    daily_commute_hours: Optional[float] = None

    social_activity_hours_per_day: Optional[float] = None

    meditates_regularly: Optional[str] = None

    preferred_exercise_type: Optional[str] = None


# =========================================================
# Welfare Intervention
# =========================================================

class InterventionPayload(BaseModel):

    personnel_id: str

    session_id: str

    action_type: str

    status: str = "initiated"

    notes: str | None = None

    scheduled_time: datetime | None = None


# =========================================================
# Workload Trend
# =========================================================

def _get_workload_trend(
    db,
    personnel_id: str
):

    recent = list(

        db.assessment_sessions.find(
            {
                "personnel_id":
                    personnel_id,

                "status":
                    "completed",
            }
        )

        .sort(
            "created_at",
            -1
        )

        .limit(5)
    )

    critical_count = sum(

        1
        for r in recent

        if r.get(
            "classification"
        ) == CRITICAL_LABEL
    )

    rest_deprived_count = sum(

        1
        for r in recent

        if (
            r.get(
                "relax_hours_preceding"
            )
            or 24
        ) < 4.0
    )

    duty_hours_values = [

        r.get(
            "duty_hours_streak"
        )

        for r in recent

        if r.get(
            "duty_hours_streak"
        ) is not None
    ]

    avg_duty_hours = (

        round(
            sum(
                duty_hours_values
            )
            / len(
                duty_hours_values
            ),
            1
        )

        if duty_hours_values

        else None
    )

    return {

        "sessions_considered":
            len(recent),

        "critical_count_recent":
            critical_count,

        "pattern":
            (
                "Recurring risk"
                if critical_count >= 2
                else "Isolated incident"
            ),

        "rest_deprived_sessions_recent":
            rest_deprived_count,

        "avg_duty_hours_recent":
            avg_duty_hours,
    }


# =========================================================
# STEP 1
# Upload Video
# =========================================================

@router.post(
    "/upload-video"
)
async def upload_video(

    video: UploadFile = File(...),

    challenge_sequence: str = Form(
        None
    ),

    current_user: dict = Depends(
        get_current_user
    ),

    db=Depends(get_db),
):

    personnel_id = (
        current_user["Username"]
    )

    challenge_list = (

        [
            c.strip()
            for c
            in challenge_sequence.split(",")
        ]

        if challenge_sequence

        else None
    )

    # -----------------------------------------------------
    # Safer temporary filename
    # -----------------------------------------------------

    suffix = os.path.splitext(
        video.filename or ".mp4"
    )[1]

    fd, video_temp = tempfile.mkstemp(
        suffix=suffix
    )

    os.close(fd)

    try:

        with open(
            video_temp,
            "wb"
        ) as f:

            shutil.copyfileobj(
                video.file,
                f
            )

        # -------------------------------------------------
        # VIDEO
        # -------------------------------------------------

        video_metrics = (
            await run_in_threadpool(
                run_opencv_processing,
                video_temp,
                challenge_list
            )
        )

        print(
            "[ASSESSMENT] Video metrics:",
            video_metrics
        )

        # -------------------------------------------------
        # Face quality
        # -------------------------------------------------

        if not video_metrics.get(
            "quality_ok",
            False
        ):

            raise HTTPException(

                status_code=422,

                detail=(
                    "Quality Insufficient – Re-record. "
                    "Face detected in only "
                    f"{video_metrics.get('face_detection_rate', 0) * 100:.0f}% "
                    "of frames. "
                    "Ensure good lighting and keep your face centered."
                ),
            )

        # -------------------------------------------------
        # AUDIO
        # -------------------------------------------------

        voice_metrics = (
            await run_in_threadpool(
                run_audio_processing_from_video,
                video_temp
            )
        )

        print(
            "[ASSESSMENT] Voice metrics:",
            voice_metrics
        )

    finally:

        if os.path.exists(
            video_temp
        ):

            os.remove(
                video_temp
            )

    # =====================================================
    # Store session
    # =====================================================

    session_id = gen_id()

    provisional_doc = {

        "_id":
            session_id,

        "personnel_id":
            personnel_id,

        "video_metrics": {

            "hr_bpm":
                video_metrics.get(
                    "hr_bpm"
                ),

            "rmssd_ms":
                video_metrics.get(
                    "rmssd_ms"
                ),

            "blink_rate_bpm":
                video_metrics.get(
                    "blink_rate"
                ),

            "blink_count":
                video_metrics.get(
                    "blink_count"
                ),

            "brow_ratio":
                video_metrics.get(
                    "brow_ratio"
                ),

            "head_jitter":
                video_metrics.get(
                    "head_jitter"
                ),

            "ear_mean":
                video_metrics.get(
                    "ear_mean"
                ),

            "ear_min":
                video_metrics.get(
                    "ear_min"
                ),

            "ear_threshold":
                video_metrics.get(
                    "ear_threshold"
                ),

            "rppg_signal_quality":
                video_metrics.get(
                    "rppg_signal_quality"
                ),

            "face_detection_rate":
                video_metrics.get(
                    "face_detection_rate"
                ),

            "fps":
                video_metrics.get(
                    "fps"
                ),

            "duration_sec":
                video_metrics.get(
                    "duration_sec"
                ),
        },

        "voice_metrics": {

            "pitch_mean_hz":
                voice_metrics.get(
                    "pitch_mean_hz"
                ),

            "pitch_std_hz":
                voice_metrics.get(
                    "pitch_std_hz"
                ),

            "spectral_centroid_hz":
                voice_metrics.get(
                    "spectral_centroid_hz"
                ),

            "vocal_stress_subscore":
                voice_metrics.get(
                    "vocal_stress_subscore"
                ),
        },

        "status":
            "pending_questionnaire",

        "created_at":
            datetime.now(
                timezone.utc
            ),
    }

    db.assessment_sessions.insert_one(
        provisional_doc
    )

    print(
        "[ASSESSMENT] Stored session:",
        session_id
    )

    return {

        "status":
            "success",

        "session_id":
            session_id,

        "message":
            "Video processed successfully. "
            "Proceed to questionnaire.",
    }


# =========================================================
# STEP 2
# Submit Questionnaire
# =========================================================

@router.post(
    "/submit-questionnaire"
)
async def submit_questionnaire(

    payload: QuestionnairePayload,

    current_user: dict = Depends(
        get_current_user
    ),

    db=Depends(get_db),
):

    personnel_id = (
        current_user["Username"]
    )

    # -----------------------------------------------------
    # Find session
    # -----------------------------------------------------

    session = (
        db.assessment_sessions.find_one(
            {
                "_id":
                    payload.session_id,

                "personnel_id":
                    personnel_id,
            }
        )
    )

    if not session:

        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found "
                "or unauthorized."
            ),
        )

    if session.get(
        "status"
    ) != "pending_questionnaire":

        raise HTTPException(
            status_code=409,
            detail=(
                "This assessment session "
                "has already been completed."
            ),
        )

    # -----------------------------------------------------
    # Video metrics
    # -----------------------------------------------------

    v_met = (
        session.get(
            "video_metrics",
            {}
        )
    )

    vo_met = (
        session.get(
            "voice_metrics",
            {}
        )
    )

    # -----------------------------------------------------
    # Combine video + audio
    # -----------------------------------------------------

    video_features = {

        "hr_bpm":
            v_met.get(
                "hr_bpm"
            ),

        "rmssd_ms":
            v_met.get(
                "rmssd_ms"
            ),

        "rppg_signal_quality":
            v_met.get(
                "rppg_signal_quality"
            ),

        "blink_rate_bpm":
            v_met.get(
                "blink_rate_bpm"
            ),

        "blink_count":
            v_met.get(
                "blink_count"
            ),

        "ear_mean":
            v_met.get(
                "ear_mean"
            ),

        "ear_min":
            v_met.get(
                "ear_min"
            ),

        "ear_threshold":
            v_met.get(
                "ear_threshold"
            ),

        "head_jitter":
            v_met.get(
                "head_jitter"
            ),

        "brow_ratio":
            v_met.get(
                "brow_ratio"
            ),

        "pitch_mean_hz":
            vo_met.get(
                "pitch_mean_hz"
            ),

        "pitch_std_hz":
            vo_met.get(
                "pitch_std_hz"
            ),

        "spectral_centroid_hz":
            vo_met.get(
                "spectral_centroid_hz"
            ),

        "vocal_stress_subscore":
            vo_met.get(
                "vocal_stress_subscore"
            ),
    }

    # -----------------------------------------------------
    # Questionnaire
    # -----------------------------------------------------

    survey_data = payload.model_dump()

    # -----------------------------------------------------
    # Final scoring
    # -----------------------------------------------------

    result = score_stress(
        video_features,
        survey_data
    )

    print(
        "[STRESS RESULT]",
        result
    )

    # =====================================================
    # Store complete clinical/assessment payload
    # =====================================================

    clinical_payload = {

        **video_features,

        "face_detection_rate":
            v_met.get(
                "face_detection_rate"
            ),

        "video_fps":
            v_met.get(
                "fps"
            ),

        "video_duration_sec":
            v_met.get(
                "duration_sec"
            ),

        "biometric_score":
            result.get(
                "biometric_score"
            ),

        "lifestyle_score":
            result.get(
                "lifestyle_score"
            ),

        "final_score":
            result.get(
                "final_score"
            ),

        "subscore_breakdown":
            result.get(
                "subscore_breakdown"
            ),

        "shap_attribution":
            result.get(
                "shap_attribution"
            ),

        "questionnaire_responses":
            survey_data,
    }

    # -----------------------------------------------------
    # Update MongoDB
    # -----------------------------------------------------

    db.assessment_sessions.update_one(

        {
            "_id":
                payload.session_id
        },

        {
            "$set": {

                "stress_probability":
                    result[
                        "stress_probability"
                    ],

                "classification":
                    result[
                        "classification"
                    ],

                "readiness_status":
                    result.get(
                        "readiness_status"
                    ),

                "biometric_score":
                    result.get(
                        "biometric_score"
                    ),

                "lifestyle_score":
                    result.get(
                        "lifestyle_score"
                    ),

                "final_score":
                    result.get(
                        "final_score"
                    ),

                "encrypted_clinical_data":
                    encrypt_blob(
                        clinical_payload
                    ),

                "status":
                    "completed",

                "completed_at":
                    datetime.now(
                        timezone.utc
                    ),
            }
        }
    )

    # =====================================================
    # Critical alert
    # =====================================================

    if (
        result["classification"]
        == CRITICAL_LABEL
    ):

        send_critical_alert(

            personnel_id=
                personnel_id,

            session_id=
                payload.session_id,

            unit_id=
                current_user.get(
                    "unit_id"
                ),
        )

    # =====================================================
    # Candidate response
    # =====================================================

    if (
        current_user.get(
            "role"
        )
        == "candidate"
    ):

        return {

            "session_id":
                payload.session_id,

            "personnel_id":
                personnel_id,

            "readiness_status":
                result.get(
                    "readiness_status"
                ),

            "classification":
                result.get(
                    "classification"
                ),

            "stress_probability":
                result[
                    "stress_probability"
                ],

            "biometric_score":
                result.get(
                    "biometric_score"
                ),

            "lifestyle_score":
                result.get(
                    "lifestyle_score"
                ),

            "final_score":
                result.get(
                    "final_score"
                ),

            "subscore_breakdown":
                result.get(
                    "subscore_breakdown"
                ),

            "shap_attribution":
                result[
                    "shap_attribution"
                ],

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

    # =====================================================
    # Commander / other role response
    # =====================================================

    return {

        "session_id":
            payload.session_id,

        "personnel_id":
            personnel_id,

        "features_used":
            video_features,

        "biometric_score":
            result.get(
                "biometric_score"
            ),

        "lifestyle_score":
            result.get(
                "lifestyle_score"
            ),

        "final_score":
            result.get(
                "final_score"
            ),

        **result,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


# =========================================================
# Candidate History
# =========================================================

@router.get(
    "/my-history"
)
def get_my_history(

    db=Depends(get_db),

    current_user: dict = Depends(
        get_current_user
    ),
):

    personnel_id = (
        current_user["Username"]
    )

    sessions = list(

        db.assessment_sessions.find(

            {
                "personnel_id":
                    personnel_id,

                "status":
                    "completed",
            }
        )

        .sort(
            "created_at",
            -1
        )

        .limit(100)
    )

    history = []

    critical_count = 0

    for s in sessions:

        clinical = decrypt_blob(
            s.get(
                "encrypted_clinical_data",
                ""
            )
        )

        if (
            s.get(
                "classification"
            )
            == CRITICAL_LABEL
        ):

            critical_count += 1

        history.append({

            "session_id":
                s["_id"],

            "classification":
                s.get(
                    "classification"
                ),

            "stress_probability":
                s.get(
                    "stress_probability"
                ),

            "biometric_score":
                s.get(
                    "biometric_score"
                ),

            "lifestyle_score":
                s.get(
                    "lifestyle_score"
                ),

            "final_score":
                s.get(
                    "final_score"
                ),

            "duty_hours_streak":
                s.get(
                    "duty_hours_streak"
                ),

            "relax_hours_preceding":
                s.get(
                    "relax_hours_preceding"
                ),

            "hr_bpm":
                clinical.get(
                    "hr_bpm"
                ),

            "rmssd_ms":
                clinical.get(
                    "rmssd_ms"
                ),

            "shap_attribution":
                clinical.get(
                    "shap_attribution"
                ),

            "timestamp":
                s[
                    "created_at"
                ].isoformat(),
        })

    return {

        "personnel_id":
            personnel_id,

        "total_assessments":
            len(history),

        "critical_count":
            critical_count,

        "history":
            history,
    }


# =========================================================
# Commander Roster
# =========================================================

@router.get(
    "/commander/roster"
)
def get_commander_roster(

    db=Depends(get_db),

    _current_user=Depends(
        require_role(
            "commander",
            "medical_officer"
        )
    ),
):

    latest_sessions = list(

        db.assessment_sessions.find(
            {
                "status":
                    "completed"
            }
        )

        .sort(
            "created_at",
            -1
        )

        .limit(50)
    )

    seen = set()

    roster = []

    for s in latest_sessions:

        pid = s[
            "personnel_id"
        ]

        if pid in seen:
            continue

        seen.add(pid)

        anonymized_id = (
            hash_identifier(
                pid
            )
        )

        roster.append({

            "candidate_id":
                anonymized_id,

            "readiness_tag":
                (
                    "Mandatory Rest Required"

                    if s.get(
                        "classification"
                    )
                    == CRITICAL_LABEL

                    else
                    "Fit for Duty"
                ),
        })

    return {

        "total_evaluated":
            len(roster),

        "roster":
            roster,
    }


# =========================================================
# Welfare Triage
# =========================================================

@router.get(
    "/welfare/triage"
)
def get_welfare_triage(

    db=Depends(get_db),

    _current_user=Depends(
        require_role(
            "medical_officer"
        )
    ),
):

    critical_sessions = list(

        db.assessment_sessions.find(

            {
                "classification":
                    CRITICAL_LABEL,

                "status":
                    "completed",
            }
        )

        .sort(
            "created_at",
            -1
        )

        .limit(20)
    )

    pending = []

    for s in critical_sessions:

        clinical = decrypt_blob(
            s.get(
                "encrypted_clinical_data",
                ""
            )
        )

        shap_attr = (
            clinical.get(
                "shap_attribution"
            )
            or []
        )

        top_driver = (

            shap_attr[0][
                "description"
            ]

            if shap_attr

            else
            "High stress probability"
        )

        pending.append({

            "session_id":
                s["_id"],

            "personnel_id":
                s["personnel_id"],

            "risk_tier":
                "Critical",

            "primary_shap_driver":
                top_driver,

            "full_shap_attribution":
                shap_attr,

            "duty_hours_streak":
                s.get(
                    "duty_hours_streak"
                ),

            "relax_hours_preceding":
                s.get(
                    "relax_hours_preceding"
                ),

            "workload_trend":
                _get_workload_trend(
                    db,
                    s[
                        "personnel_id"
                    ]
                ),

            "suggested_action":
                "Clinical rest order & psychological check-in.",
        })

    return {

        "pending_triages":
            pending
    }


# =========================================================
# Welfare Intervention
# =========================================================

@router.post(
    "/welfare/interventions"
)
def log_welfare_intervention(

    data: InterventionPayload,

    db=Depends(get_db),

    current_user=Depends(
        require_role(
            "medical_officer"
        )
    ),
):

    doc = {

        "_id":
            gen_id(),

        "personnel_id":
            data.personnel_id,

        "session_id":
            data.session_id,

        "action_type":
            data.action_type,

        "status":
            data.status,

        "notes":
            data.notes,

        "scheduled_time":
            data.scheduled_time,

        "logged_by_hash":
            hash_identifier(
                current_user[
                    "Username"
                ]
            ),

        "created_at":
            datetime.now(
                timezone.utc
            ),
    }

    db.welfare_interventions.insert_one(
        doc
    )

    return {

        "status":
            "success",

        "message":
            (
                f"Intervention "
                f"'{data.action_type}' "
                f"recorded for "
                f"{data.personnel_id}."
            ),
    }