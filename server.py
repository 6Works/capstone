"""
FastAPI Server Gateway for the Decoupled Microservices Pipeline.
Follows SOLID: Dependency Inversion via PipelineFactory.
Allows dynamic swapping of ASR, NLP, and TTS service providers at runtime.
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from orchestrator.factory import PipelineFactory
from services.contracts import AudioPayload
from services.roles import role_service
from services.llm.gateway import llm_gateway
from services.blueprints import blueprint_service, TelemetryEvent, ArtifactBundle
from services.context import context_manager, ContextUpdateRequest

app = FastAPI(title="Decoupled 6-Stage Conversational Voice Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("web", exist_ok=True)
os.makedirs("web/pages", exist_ok=True)
os.makedirs("web/static", exist_ok=True)
os.makedirs("storage/temp_audio", exist_ok=True)

app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Active configuration state
active_config = {
    "asr_provider": "google",
    "tts_provider": "pyttsx3",
    "nlp_provider": "llm",
    "dialogue_provider": "llm",
    "input_provider": "file",
    "output_provider": "web",
    "bot_name": "Mira"
}

# Instantiate decoupled pipeline via factory
pipeline = PipelineFactory.create_pipeline(**active_config)


class TextInputRequest(BaseModel):
    text: str
    play_audio_on_server: bool = False


class ProviderConfigRequest(BaseModel):
    asr_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    nlp_provider: Optional[str] = None
    dialogue_provider: Optional[str] = None


@app.get("/api/health")
@app.get("/api/health/")
def health_check():
    """Health check endpoint for platform monitoring."""
    return {
        "status": "healthy",
        "service": "agentic-ai-interviewer",
        "llm_provider": llm_gateway.active_provider_id,
        "llm_model": llm_gateway.active_model
    }


@app.get("/api/pipeline/providers")
def get_available_providers():
    """Lists available pluggable providers for each stage."""
    llm_info = llm_gateway.get_status()
    return {
        "active": active_config,
        "llm_gateway": {
            "active_provider": llm_info.get("active_provider"),
            "active_model": llm_info.get("active_model"),
            "fallback_enabled": llm_info.get("fallback_enabled")
        },
        "available_providers": {
            "asr": [
                {"id": "google", "name": "Google Web Speech (Fast cloud free)", "status": "active"},
                {"id": "whisper", "name": "Whisper Transformer (Local / HuggingFace)", "status": "ready"},
                {"id": "mock", "name": "Mock ASR (Synthetic deterministic)", "status": "testing"}
            ],
            "nlp": [
                {"id": "llm", "name": "LLM Intent & Tool Engine (Gemini / OpenAI / Ollama)", "status": "active"},
                {"id": "rule_based", "name": "Rule-Based Intent & Entity NLU", "status": "active"}
            ],
            "dialogue": [
                {"id": "llm", "name": "Generative LLM Dialogue Manager (Gemini / OpenAI / Ollama)", "status": "active"},
                {"id": "modular", "name": "Decoupled Memory & State Machine (Deterministic)", "status": "active"}
            ],
            "tts": [
                {"id": "pyttsx3", "name": "pyttsx3 SAPI5 (Offline native Windows)", "status": "active"},
                {"id": "elevenlabs", "name": "ElevenLabs (Human Cadence & Natural Pauses)", "status": "ready"},
                {"id": "gtts", "name": "Google Text-To-Speech (MP3)", "status": "active"}
            ],
            "audio_output": [
                {"id": "web", "name": "Browser HTML5 Audio Element", "status": "active"},
                {"id": "speaker", "name": "System Speaker Hardware", "status": "active"}
            ]
        }
    }


@app.post("/api/pipeline/configure")
def configure_providers(req: ProviderConfigRequest):
    """Dynamically reconfigures microservice providers at runtime without restart."""
    global pipeline, active_config
    if req.asr_provider:
        active_config["asr_provider"] = req.asr_provider
    if req.tts_provider:
        active_config["tts_provider"] = req.tts_provider
    if req.nlp_provider:
        active_config["nlp_provider"] = req.nlp_provider
    if req.dialogue_provider:
        active_config["dialogue_provider"] = req.dialogue_provider

    # Re-assemble pipeline with new providers (Dependency Injection)
    pipeline = PipelineFactory.create_pipeline(**active_config)
    return {"status": "success", "active_config": active_config}


@app.get("/api/pipeline/status")
def get_pipeline_status():
    """Returns the decoupled status of all 6 stages."""
    return {
        "status": "ready",
        "architecture": "microservices_solid",
        "active_config": active_config,
        "stages": [
            {"id": 1, "name": "Audio Input", "provider": active_config["input_provider"], "color": "#60a5fa"},
            {"id": 2, "name": "Speech Recognition (ASR)", "provider": active_config["asr_provider"], "color": "#4ade80"},
            {"id": 3, "name": "Natural Language Processing", "provider": active_config["nlp_provider"], "color": "#facc15"},
            {"id": 4, "name": "Dialogue Management", "provider": "modular_memory_state", "color": "#c084fc"},
            {"id": 5, "name": "Text-to-Speech (TTS)", "provider": active_config["tts_provider"], "color": "#f472b6"},
            {"id": 6, "name": "Audio Output", "provider": active_config["output_provider"], "color": "#fb923c"},
        ],
        "dialogue_state": pipeline.dialogue.state,
        "turns_completed": pipeline.dialogue.turn_count
    }


@app.post("/api/pipeline/process-text")
def process_text_input(req: TextInputRequest):
    """Processes text query through stages 3 to 6."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    telemetry = pipeline.run_turn_from_text(req.text, play_audio=req.play_audio_on_server)
    
    tts_meta = telemetry.get("stages", {}).get("stage5_tts", {})
    file_path = tts_meta.get("file_path")
    audio_url = None
    if file_path and os.path.exists(file_path):
        filename = os.path.basename(file_path)
        dest = os.path.join("storage/temp_audio", filename)
        shutil.copyfile(file_path, dest)
        audio_url = f"/api/pipeline/audio/{filename}"

    telemetry["audio_url"] = audio_url
    return telemetry


@app.post("/api/pipeline/process-audio")
async def process_audio_upload(file: UploadFile = File(...)):
    """Processes audio upload through decoupled 6-stage microservices."""
    try:
        temp_wav_path = os.path.join("storage/temp_audio", f"upload_{file.filename}")
        with open(temp_wav_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if os.path.getsize(temp_wav_path) < 44:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Audio signal too short or empty. Please speak clearly into the microphone."}
            )

        # Stage 1: File Audio Input Service
        payload = pipeline.audio_input.capture(file_path=temp_wav_path)

        # Run Stages 2 -> 6
        telemetry = pipeline.run_turn_from_audio_payload(payload, play_audio=False)

        # Expose audio URL for browser playback
        tts_meta = telemetry.get("stages", {}).get("stage5_tts", {})
        file_path = tts_meta.get("file_path")
        audio_url = None
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            dest = os.path.join("storage/temp_audio", filename)
            shutil.copyfile(file_path, dest)
            audio_url = f"/api/pipeline/audio/{filename}"

        telemetry["audio_url"] = audio_url
        return telemetry

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": f"Service execution error: {str(e)}"}
        )


@app.get("/api/pipeline/audio/{filename}")
def get_audio_file(filename: str):
    path = os.path.join("storage/temp_audio", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media_type)


@app.post("/api/pipeline/reset")
def reset_conversation():
    pipeline.dialogue.reset()
    return {"status": "success", "message": "Dialogue session and memory store reset"}


# ============================================================================
# Individual Stage Test Endpoints (For Deep Dive Pages)
# ============================================================================

class NLPTestRequest(BaseModel):
    text: str


class TTSTestRequest(BaseModel):
    text: str
    provider: Optional[str] = "pyttsx3"
    speech_rate: Optional[int] = 175


class DialogueTestRequest(BaseModel):
    text: str
    intent: Optional[str] = None


@app.post("/api/test/stage1-audio-input")
def test_stage1_audio():
    from services.audio_input import SyntheticAudioInput
    synth = SyntheticAudioInput()
    payload = synth.capture(duration_sec=1.5)
    return payload.model_dump()


@app.post("/api/test/stage2-asr")
def test_stage2_asr(req: NLPTestRequest):
    # Simulated transcription testbench
    from services.contracts import AudioPayload
    payload = AudioPayload(source_type="test", format="wav", duration_sec=1.2)
    res = pipeline.asr.transcribe(payload)
    if not res.transcript:
        res.transcript = req.text
        res.provider += " (testbench)"
    return res.model_dump()


@app.post("/api/test/stage3-nlp")
def test_stage3_nlp(req: NLPTestRequest):
    res = pipeline.nlp.parse(req.text)
    return res.model_dump()


@app.post("/api/test/stage4-dialogue")
def test_stage4_dialogue(req: DialogueTestRequest):
    nlp_res = pipeline.nlp.parse(req.text)
    if req.intent:
        nlp_res.intent = req.intent
    turn_res = pipeline.dialogue.process_turn(nlp_res)
    return {
        "turn": turn_res.model_dump(),
        "memory_history": pipeline.dialogue.memory.turns,
        "active_context": pipeline.dialogue.memory.context
    }


@app.post("/api/test/stage5-tts")
def test_stage5_tts(req: TTSTestRequest):
    from services.tts import Pyttsx3TTSService, ElevenLabsTTSService, GTTSService
    if req.provider == "elevenlabs":
        service = ElevenLabsTTSService()
    elif req.provider == "gtts":
        service = GTTSService()
    else:
        service = Pyttsx3TTSService(speech_rate=req.speech_rate or 175)

    res = service.synthesize(req.text)
    audio_url = None
    if res.file_path and os.path.exists(res.file_path):
        filename = os.path.basename(res.file_path)
        dest = os.path.join("storage/temp_audio", filename)
        shutil.copyfile(res.file_path, dest)
        audio_url = f"/api/pipeline/audio/{filename}"

    data = res.model_dump()
    data["audio_url"] = audio_url
    return data


# ============================================================================
# Central LLM Model Gateway & Orchestrator Endpoints
# ============================================================================

class LLMConfigureRequest(BaseModel):
    provider_id: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    fallback_enabled: Optional[bool] = None


class LLMTestRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


@app.get("/api/llm/status")
def get_llm_status(force_refresh: bool = False):
    """Returns diagnostic health and metadata across all 5 LLM providers."""
    return llm_gateway.get_status(force_refresh=force_refresh)


@app.post("/api/llm/configure")
def configure_llm_gateway(req: LLMConfigureRequest):
    """Hot-swaps active provider, model, API keys, or endpoints at runtime."""
    return llm_gateway.configure(
        provider_id=req.provider_id,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        temperature=req.temperature,
        fallback_enabled=req.fallback_enabled
    )


@app.post("/api/llm/test")
def test_llm_gateway(req: LLMTestRequest):
    """Executes a live test prompt across the active LLM Gateway."""
    return llm_gateway.test_completion(req.prompt, model=req.model)


# ============================================================================
# Role & Interviewer Engine Endpoints (Solo & Co-Pilot Modes)
# ============================================================================

class RoleConfigureRequest(BaseModel):
    role_id: str
    mode_id: Optional[str] = None


class InterviewerTurnRequest(BaseModel):
    candidate_text: str
    mode: Optional[str] = None


@app.get("/api/roles")
def get_roles():
    """Lists all available personas and sub-modes."""
    return role_service.list_roles()


@app.get("/api/roles/active")
def get_active_role():
    """Returns the currently active persona and sub-mode configuration."""
    return role_service.get_active_role()


@app.post("/api/roles/configure")
def configure_role(req: RoleConfigureRequest):
    """Sets the active persona and sub-mode."""
    return role_service.configure_role(req.role_id, req.mode_id)


@app.post("/api/roles/interviewer/turn")
def process_interviewer_turn(req: InterviewerTurnRequest):
    """Processes candidate utterance under either Solo or Co-Pilot mode."""
    return role_service.process_interviewer_turn(req.candidate_text, req.mode)


@app.post("/api/roles/interviewer/reset")
def reset_interview():
    """Resets the interviewer question progression and turn history."""
    role_service.reset_interview()
    return {"status": "success", "message": "Interview session reset"}


class CodeRunRequest(BaseModel):
    code: str
    language: Optional[str] = "python"


@app.post("/api/interviewer/run-code")
def run_interviewer_code(req: CodeRunRequest):
    """Executes candidate code in an isolated subprocess environment, captures standard output, and updates context."""
    import sys, subprocess, time, shutil
    lang = (req.language or "python").lower()
    t0 = time.time()
    
    try:
        if lang in ["python", "py"]:
            cmd = [sys.executable, "-c", req.code]
        elif lang in ["javascript", "js", "typescript", "ts"]:
            node_bin = shutil.which("node") or "node"
            cmd = [node_bin, "-e", req.code]
        elif lang in ["bash", "sh"]:
            cmd = ["powershell", "-NoProfile", "-Command", req.code]
        else:
            cmd = [sys.executable, "-c", req.code]
            
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=12
        )
        duration_ms = round((time.time() - t0) * 1000, 1)
        stdout = proc.stdout
        stderr = proc.stderr
        
        output = stdout
        if stderr:
            output = f"{stdout}\n[stderr]: {stderr}" if stdout else stderr
        if not output.strip():
            output = f"Code executed successfully (exit code {proc.returncode}) with no stdout."
            
        status = "success" if proc.returncode == 0 else "error"
        
        # Telemetry to active blueprint & Context Manager
        try:
            blueprint_service.record_event(
                plugin_id="CodeEditor",
                event_type="code_execution",
                payload={
                    "language": lang,
                    "returncode": proc.returncode,
                    "duration_ms": duration_ms,
                    "output_preview": output[:200]
                }
            )
            context_manager.update_workspace_state("CodeEditor", {
                "code": req.code,
                "language": lang,
                "output": output,
                "status": status,
                "returncode": proc.returncode,
                "duration_ms": duration_ms
            })
        except Exception:
            pass

        return {
            "status": status,
            "output": output,
            "returncode": proc.returncode,
            "duration_ms": duration_ms,
            "runtime": f"Subprocess ({lang.capitalize()})"
        }
    except subprocess.TimeoutExpired:
        timeout_msg = "Execution timed out after 12.0 seconds. Possible infinite loop detected."
        context_manager.update_workspace_state("CodeEditor", {
            "code": req.code,
            "language": lang,
            "output": timeout_msg,
            "status": "timeout"
        })
        return {
            "status": "error",
            "output": timeout_msg,
            "returncode": 124,
            "runtime": f"Subprocess ({lang.capitalize()})"
        }
    except Exception as e:
        err_msg = f"Runtime Sandbox Exception: {str(e)}"
        return {
            "status": "error",
            "output": err_msg,
            "returncode": 1,
            "runtime": f"Subprocess ({lang.capitalize()})"
        }


# ============================================================================
# Three-Layer Plug-and-Play Role Blueprint Endpoints
# ============================================================================

class BlueprintProfileRequest(BaseModel):
    jd_text: str
    seniority: Optional[str] = "Senior Staff"
    resume_text: Optional[str] = None


class BlueprintSelectRequest(BaseModel):
    blueprint_id: str


@app.get("/api/blueprints")
def get_blueprints():
    """Lists all available Role Blueprints across domains."""
    return blueprint_service.list_blueprints()


@app.get("/api/blueprints/active")
def get_active_blueprint():
    """Returns currently active Role Blueprint and current Phase layout."""
    return blueprint_service.get_active_blueprint()


@app.post("/api/blueprints/select")
def select_blueprint(req: BlueprintSelectRequest):
    """Sets active Role Blueprint for the interview session."""
    bp = blueprint_service.set_active_blueprint(req.blueprint_id)
    return {"status": "success", "blueprint": bp.model_dump()}


@app.post("/api/blueprints/phase/advance")
def advance_blueprint_phase():
    """Advances session to the next progressive disclosure phase."""
    return blueprint_service.advance_phase()


@app.post("/api/blueprints/profile")
def profile_job_description(req: BlueprintProfileRequest):
    """Stage 1: Synthesizes a tailored Blueprint from JD and Seniority using LLMGateway."""
    bp = blueprint_service.profile_job_description(
        jd_text=req.jd_text,
        seniority=req.seniority or "Senior",
        resume_text=req.resume_text
    )
    return bp.model_dump()


class BlueprintCustomizeLayoutRequest(BaseModel):
    left_plugins: List[str]
    right_plugins: List[str]
    ratio: Optional[str] = "50-50"


@app.post("/api/blueprints/customize-layout")
def customize_blueprint_layout(req: BlueprintCustomizeLayoutRequest):
    """Dynamically plugs or unplugs workspace plugins into the active interview layout."""
    return blueprint_service.customize_active_layout(
        left_plugins=req.left_plugins,
        right_plugins=req.right_plugins,
        ratio=req.ratio or "50-50"
    )


@app.post("/api/blueprints/events")
def record_blueprint_event(event: TelemetryEvent):
    """Stage 3: Records events from the Unified EventBus across mounted plugins and syncs to Context."""
    try:
        context_manager.update_workspace_state(event.plugin_id, event.payload)
    except Exception:
        pass
    return blueprint_service.record_event(event)


@app.post("/api/blueprints/evaluate")
def evaluate_blueprint_artifacts(bundle: ArtifactBundle):
    """Stage 4: Evaluates exported plugin snapshots against declarative Blueprint rubric."""
    return blueprint_service.evaluate_artifacts(bundle)


# ============================================================================
# Universal Session Context Management Endpoints
# ============================================================================

@app.get("/api/context/active")
def get_active_context():
    """Returns real-time session context snapshot (dialogue trace, workspace state, radar)."""
    return context_manager.get_snapshot().model_dump()


@app.post("/api/context/reset")
def reset_session_context():
    """Resets multi-turn conversation memory and workspace state buffers."""
    context_manager.reset_session()
    role_service.reset_interview()
    return {"status": "success", "message": "Session context reset."}


@app.post("/api/context/update")
def update_session_context(req: ContextUpdateRequest):
    """Updates live context state manually or via telemetry hooks."""
    if req.plugin_id and req.workspace_data:
        context_manager.update_workspace_state(req.plugin_id, req.workspace_data)
    if req.candidate_claim:
        snap = context_manager.get_snapshot()
        if req.candidate_claim not in snap.discovered_claims:
            snap.discovered_claims.append(req.candidate_claim)
    return context_manager.get_snapshot().model_dump()


# ============================================================================
# Multi-Page Navigation Routes (Strict SOLID Separation)
# ============================================================================

@app.get("/")
def serve_dashboard():
    return FileResponse("web/index.html")

@app.get("/roles")
@app.get("/roles/")
@app.get("/role")
@app.get("/interview/roles")
def serve_roles():
    return FileResponse("web/pages/roles.html")

@app.get("/interviewer-mode")
@app.get("/interviewer-mode/")
@app.get("/interviewer/mode")
@app.get("/interview-mode")
def serve_interviewer_mode():
    return FileResponse("web/pages/interviewer_mode.html")

@app.get("/interviewer/solo")
@app.get("/interviewer/solo/")
@app.get("/interview/solo")
def serve_interviewer_solo():
    return FileResponse("web/pages/interviewer_solo.html")

@app.get("/interviewer/copilot")
@app.get("/interviewer/copilot/")
@app.get("/interview/copilot")
def serve_interviewer_copilot():
    return FileResponse("web/pages/interviewer_copilot.html")

@app.get("/interview/profiling")
@app.get("/interview/profiling/")
@app.get("/profiling")
def serve_interview_profiling():
    """Middle Layer: Intelligent Role Profiling & Blueprint Selection."""
    return FileResponse("web/pages/interview_profiling.html")

@app.get("/interview/workspace")
@app.get("/interview/workspace/")
@app.get("/workspace")
def serve_interview_workspace():
    """Universal 3-Layer Plug-and-Play Interview Workspace."""
    return FileResponse("web/pages/interview_workspace.html")

@app.get("/interview/call")
@app.get("/interview/call/")
@app.get("/call")
def serve_interview_call():
    """Universal Zoom-Style Collaborative Interview Suite with on-demand tools."""
    return FileResponse("web/pages/interview_call.html")

@app.get("/audio-input")
def serve_stage1_audio_input():
    return FileResponse("web/pages/audio_input.html")

@app.get("/asr")
def serve_stage2_asr():
    return FileResponse("web/pages/asr.html")

@app.get("/nlp")
def serve_stage3_nlp():
    return FileResponse("web/pages/nlp.html")

@app.get("/dialogue")
def serve_stage4_dialogue():
    return FileResponse("web/pages/dialogue.html")

@app.get("/tts")
def serve_stage5_tts():
    return FileResponse("web/pages/tts.html")

@app.get("/audio-output")
def serve_stage6_audio_output():
    return FileResponse("web/pages/audio_output.html")

@app.get("/architecture")
def serve_stage7_architecture():
    return FileResponse("web/pages/architecture.html")


if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if os.getenv("PORT"):
        target_port = int(os.environ["PORT"])
    elif is_port_in_use(8000):
        target_port = 8001
        print(f"[Notice] Port 8000 is already in use by another service. Selecting port {target_port}...")
    else:
        target_port = 8000

    print(f"Starting server on http://127.0.0.1:{target_port}")
    uvicorn.run("server:app", host="127.0.0.1", port=target_port, reload=True)
