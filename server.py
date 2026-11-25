"""
FastAPI Server Gateway for the Decoupled Microservices Pipeline.
Follows SOLID: Dependency Inversion via PipelineFactory.
Allows dynamic swapping of ASR, NLP, and TTS service providers at runtime.
"""

import os
import time
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
    "asr_provider": "whisper",
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
                {"id": "whisper", "name": "faster-whisper Small.en INT8 (High Accuracy - Name Optimized)", "status": "active"},
                {"id": "whisper_tiny", "name": "faster-whisper Tiny.en INT8 (Offline Ultra-Fast ~200ms)", "status": "active"},
                {"id": "google", "name": "Google Web Speech (Fast cloud free)", "status": "active"},
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
                {"id": "neural_vits", "name": "Neural VITS & Voice Cloning (Zero-Shot Local ONNX)", "status": "active"},
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
    provider: Optional[str] = "neural_vits"
    speech_rate: Optional[int] = 175
    profile_id: Optional[str] = "mira_neural"
    with_pauses: Optional[bool] = True
    with_normalization: Optional[bool] = True


class DialogueTestRequest(BaseModel):
    text: str
    intent: Optional[str] = None


@app.post("/api/test/stage1-audio-input")
def test_stage1_audio():
    from services.audio_input import SyntheticAudioInput
    synth = SyntheticAudioInput()
    payload = synth.capture(duration_sec=1.5)
    return payload.model_dump()


class ASRTestRequest(BaseModel):
    text: Optional[str] = "What time is it right now?"
    provider: Optional[str] = "whisper"
    initial_prompt: Optional[str] = None


@app.post("/api/test/stage2-asr")
def test_stage2_asr(req: ASRTestRequest):
    """Executes live ASR testbench using specified provider (faster-whisper or Google)."""
    import time
    from services.contracts import AudioPayload
    from services.asr import GoogleWebSpeechASR, WhisperTransformerASR, MockASR

    target_provider = req.provider or active_config.get("asr_provider", "whisper")
    if target_provider in ["whisper", "whisper_small"]:
        asr_svc = WhisperTransformerASR(model_size="small.en", compute_type="int8", initial_prompt=req.initial_prompt)
    elif target_provider == "whisper_tiny":
        asr_svc = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", initial_prompt=req.initial_prompt)
    elif target_provider == "google":
        asr_svc = GoogleWebSpeechASR(language="en-US")
    else:
        asr_svc = MockASR(fixed_transcript=req.text or "Mock Speech")

    # Synthesize live acoustic waveform with pyttsx3 to exercise end-to-end Whisper inference
    temp_wav = os.path.join(tempfile.gettempdir(), f"asr_bench_{os.getpid()}_{int(time.time()*1000)}.wav")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(req.text or "Hello Mira", temp_wav)
        engine.runAndWait()

        payload = AudioPayload(file_path=temp_wav, source_type="test", format="wav", duration_sec=1.5)
        res = asr_svc.transcribe(payload)
    except Exception as e:
        # Fallback if audio device synthesis is locked
        payload = AudioPayload(source_type="test", format="wav", duration_sec=1.2)
        res = asr_svc.transcribe(payload)
        if not res.transcript:
            res.transcript = req.text
            res.provider += " (testbench)"
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

    return res.model_dump()


@app.post("/api/asr/transcribe-file")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    provider: str = Form("whisper"),
    vad_filter: bool = Form(True),
    beam_size: int = Form(5),
    language: str = Form("en"),
    initial_prompt: Optional[str] = Form(None)
):
    """
    Direct Stage 2 ASR Execution on uploaded audio (mic recording or file).
    Returns real transcription, latency, confidence, word segments, and audio metadata.
    """
    import time
    from services.contracts import AudioPayload
    from services.asr import GoogleWebSpeechASR, WhisperTransformerASR, MockASR

    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_wav_path = os.path.join("storage/temp_audio", f"asr_live_{os.getpid()}_{int(time.time()*1000)}{ext}")

    try:
        content = await file.read()
        with open(temp_wav_path, "wb") as f:
            f.write(content)

        file_size_bytes = len(content)
        if file_size_bytes < 44:
            raise HTTPException(status_code=400, detail="Audio file too small or empty")

        if provider in ["whisper", "whisper_small"]:
            asr_svc = WhisperTransformerASR(
                model_size="small.en",
                compute_type="int8",
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                initial_prompt=initial_prompt
            )
        elif provider == "whisper_tiny":
            asr_svc = WhisperTransformerASR(
                model_size="tiny.en",
                compute_type="int8",
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                initial_prompt=initial_prompt
            )
        elif provider == "google":
            asr_svc = GoogleWebSpeechASR(language="en-US" if language == "en" else language)
        else:
            asr_svc = MockASR(fixed_transcript="Sample recognized audio transcription")

        payload = AudioPayload(
            file_path=temp_wav_path,
            source_type="live_test",
            format=ext.lstrip(".")
        )
        res = asr_svc.transcribe(payload)
        data = res.model_dump()
        data["file_size_bytes"] = file_size_bytes
        data["file_name"] = file.filename
        return data

    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass


class ASRBenchmarkRequest(BaseModel):
    text: Optional[str] = "The quick brown fox jumps over the lazy dog"
    providers: Optional[List[str]] = ["whisper_small", "whisper_tiny", "google", "mock"]
    initial_prompt: Optional[str] = None


@app.post("/api/asr/benchmark")
def benchmark_asr_models(req: ASRBenchmarkRequest):
    """
    Benchmarking Arena for Stage 2 ASR.
    Races multiple models on the exact same synthesized audio and computes comparative metrics.
    """
    import time
    from services.contracts import AudioPayload
    from services.asr import GoogleWebSpeechASR, WhisperTransformerASR, MockASR
    import pyttsx3

    temp_wav = os.path.join(tempfile.gettempdir(), f"asr_bench_race_{os.getpid()}_{int(time.time()*1000)}.wav")
    try:
        engine = pyttsx3.init()
        engine.save_to_file(req.text or "The quick brown fox jumps over the lazy dog", temp_wav)
        engine.runAndWait()

        results = []
        providers_to_test = req.providers or ["whisper_small", "whisper_tiny", "google", "mock"]

        for pid in providers_to_test:
            if pid in ["whisper_small", "whisper"]:
                svc = WhisperTransformerASR(model_size="small.en", compute_type="int8", initial_prompt=req.initial_prompt)
                label = "Whisper Small.en (INT8 - Name Optimized)"
            elif pid == "whisper_tiny":
                svc = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", initial_prompt=req.initial_prompt)
                label = "Whisper Tiny.en (INT8 - Fast)"
            elif pid == "google":
                svc = GoogleWebSpeechASR(language="en-US")
                label = "Google Web Speech (Cloud)"
            else:
                svc = MockASR(fixed_transcript=req.text or "Sample")
                label = "Mock ASR (Synthetic)"

            payload = AudioPayload(file_path=temp_wav, source_type="benchmark", format="wav")
            res = svc.transcribe(payload)
            item = res.model_dump()
            item["display_name"] = label
            item["word_count"] = len(res.transcript.split()) if res.transcript else 0
            item["char_count"] = len(res.transcript)
            results.append(item)

        results.sort(key=lambda x: x["latency_sec"])
        return {
            "reference_text": req.text,
            "results": results,
            "fastest": results[0]["display_name"] if results else None
        }

    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


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
    try:
        from services.tts import (
            Pyttsx3TTSService,
            ElevenLabsTTSService,
            GTTSService,
            NeuralVoiceCloningTTSService
        )
        if req.provider in ["neural_vits", "neural", "cloned"]:
            service = NeuralVoiceCloningTTSService(
                speaker_profile_id=req.profile_id or "mira_neural",
                speech_rate=req.speech_rate or 175,
                with_human_pauses=req.with_pauses if req.with_pauses is not None else True,
                with_normalization=req.with_normalization if req.with_normalization is not None else True
            )
        elif req.provider == "elevenlabs":
            service = ElevenLabsTTSService()
        elif req.provider == "gtts":
            service = GTTSService()
        else:
            service = Pyttsx3TTSService(speech_rate=req.speech_rate or 175)

        res = service.synthesize(req.text or "")
        audio_url = None
        spectrogram_data = None
        if res.file_path and os.path.exists(res.file_path):
            filename = os.path.basename(res.file_path)
            dest = os.path.join("storage/temp_audio", filename)
            shutil.copyfile(res.file_path, dest)
            audio_url = f"/api/pipeline/audio/{filename}"

            # Module 1: Automatically compute 80-band Mel-Spectrogram & F0 Contour
            try:
                from services.tts import AcousticAnalyzer
                spectrogram_data = AcousticAnalyzer.analyze_wav(dest)
            except Exception as spec_err:
                print(f"[AcousticAnalyzer] Spectrogram extraction skipped: {spec_err}")

        data = res.model_dump()
        data["audio_url"] = audio_url
        data["spectrogram"] = spectrogram_data
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": f"TTS synthesis failed: {str(e)}"}
        )


# ============================================================================
# Neural TTS & Voice Cloning Course Studio Endpoints
# ============================================================================

class TextNormalizeRequest(BaseModel):
    text: str


@app.get("/api/tts/voices")
def get_tts_voices():
    """Lists available speaker profiles including zero-shot cloned voices."""
    from services.tts import voice_cloner
    return {
        "status": "success",
        "voices": voice_cloner.list_profiles()
    }


@app.post("/api/tts/normalize-text")
def normalize_and_phonemize_text(req: TextNormalizeRequest):
    """
    Chapter 3 Front-End Inspector:
    Expands Non-Standard Words (NSW) and computes Grapheme-to-Phoneme (G2P) breakdown.
    """
    from services.tts import TextNormalizer, G2PPhonemizer
    norm = TextNormalizer.normalize(req.text)
    g2p = G2PPhonemizer.phonemize_sentence(norm["normalized_text"])
    return {
        "status": "success",
        "original_text": norm["original_text"],
        "normalized_text": norm["normalized_text"],
        "changes_detected": norm["changes_detected"],
        "phonetics": g2p
    }


@app.post("/api/tts/clone-voice")
async def clone_voice_from_reference(
    file: UploadFile = File(...),
    name: Optional[str] = Form("Custom Cloned Voice")
):
    """
    Chapter 4 Zero-Shot Voice Cloning:
    Extracts acoustic speaker profile (F0, spectral centroid, 128-D embedding) from audio.
    Supports any audio container: WAV, WebM (browser mic), MP3, OGG, Opus, AAC.
    """
    os.makedirs("storage/temp_audio", exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".audio"
    temp_path = os.path.join("storage/temp_audio", f"ref_clone_{int(time.time()*1000)}{ext}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from services.tts import VoiceCloner, voice_cloner
        profile = VoiceCloner.extract_acoustic_profile_from_wav(temp_path, profile_name=name or "Cloned Voice")
        voice_cloner.save_cloned_profile(profile)
        return {
            "status": "success",
            "message": f"Successfully extracted acoustic profile and created cloned voice '{name}'",
            "profile": profile.to_dict()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": f"Acoustic profiling failed: {str(e)}"}
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/tts/benchmark")
def benchmark_tts_engines(req: TextNormalizeRequest):
    """
    Course Arena: Compares Neural VITS Cloned vs pyttsx3 vs ElevenLabs Cadence vs gTTS.
    """
    from services.tts import (
        Pyttsx3TTSService,
        ElevenLabsTTSService,
        GTTSService,
        NeuralVoiceCloningTTSService
    )

    results = []

    # 1. Neural VITS Cloned
    vits = NeuralVoiceCloningTTSService(speaker_profile_id="mira_neural")
    r1 = vits.synthesize(req.text)
    results.append({
        "engine": "Neural VITS (Zero-Shot ONNX)",
        "provider_id": "neural_vits",
        "latency_ms": int(r1.latency_sec * 1000),
        "sample_rate": 16000,
        "format": r1.audio_format.upper(),
        "status": r1.status,
        "human_cadence": "High (Adaptive Prosody)",
        "offline": True
    })

    # 2. pyttsx3 SAPI5
    sapi = Pyttsx3TTSService()
    r2 = sapi.synthesize(req.text)
    results.append({
        "engine": "pyttsx3 Windows SAPI5",
        "provider_id": "pyttsx3",
        "latency_ms": int(r2.latency_sec * 1000),
        "sample_rate": 22050,
        "format": r2.audio_format.upper(),
        "status": r2.status,
        "human_cadence": "Low (Flat Formant)",
        "offline": True
    })

    # 3. ElevenLabs Cadence
    eleven = ElevenLabsTTSService()
    r3 = eleven.synthesize(req.text)
    results.append({
        "engine": "ElevenLabs Cadence Model",
        "provider_id": "elevenlabs",
        "latency_ms": int(r3.latency_sec * 1000),
        "sample_rate": 44100 if eleven.api_key else 22050,
        "format": r3.audio_format.upper(),
        "status": r3.status,
        "human_cadence": "Highest (Breaths & Micro-pauses)",
        "offline": eleven.api_key is None
    })

    # Clean up temp audio files
    for r in [r1, r2, r3]:
        if r.file_path and os.path.exists(r.file_path):
            try:
                os.remove(r.file_path)
            except Exception:
                pass

    return {
        "status": "success",
        "benchmark_text": req.text,
        "engines": results
    }


class VoiceMorphRequest(BaseModel):
    profile_id_a: str = "mira_neural"
    profile_id_b: str = "alex_interviewer"
    alpha: float = 0.5
    name: Optional[str] = None


class FormantSynthRequest(BaseModel):
    vowel: str = "a"
    f0: float = 130.0
    duration: float = 0.8


class SpectrogramRequest(BaseModel):
    filename: str


@app.post("/api/tts/morph-voices")
def morph_voices(req: VoiceMorphRequest):
    """
    Module 5: Acoustic Latent Space Interpolation.
    Blends two speaker embedding vectors: z_blend = (1-a)*z_A + a*z_B.
    """
    from services.tts import voice_cloner
    morphed = voice_cloner.morph_profiles(
        profile_id_a=req.profile_id_a,
        profile_id_b=req.profile_id_b,
        alpha=req.alpha,
        custom_name=req.name
    )
    return {
        "status": "success",
        "message": f"Successfully morphed voices with alpha={req.alpha:.2f}",
        "profile": morphed.to_dict()
    }


@app.post("/api/tts/spectrogram")
def get_audio_spectrogram(req: SpectrogramRequest):
    """
    Module 1: Returns 80-band Mel-Spectrogram & F0 pitch contour for any audio file.
    """
    from services.tts import AcousticAnalyzer
    path = os.path.join("storage/temp_audio", os.path.basename(req.filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found in storage")

    data = AcousticAnalyzer.analyze_wav(path)
    return data


@app.post("/api/tts/formant-synth")
def synthesize_formant_vowel(req: FormantSynthRequest):
    """
    Module 3: Klatt Cascade Formant Synthesizer Simulator.
    Generates rule-based formant acoustic speech (vocal tract resonators).
    """
    from services.tts import FormantSynthesizer, AcousticAnalyzer
    wav_path = FormantSynthesizer.synthesize_vowel(
        vowel_key=req.vowel,
        f0=req.f0,
        duration=req.duration
    )
    filename = os.path.basename(wav_path)
    dest = os.path.join("storage/temp_audio", filename)
    shutil.copyfile(wav_path, dest)
    audio_url = f"/api/pipeline/audio/{filename}"

    # Extract spectrogram of synthesized formant
    spec = AcousticAnalyzer.analyze_wav(dest)

    return {
        "status": "success",
        "vowel": req.vowel,
        "f0_hz": req.f0,
        "audio_url": audio_url,
        "spectrogram": spec
    }


@app.get("/api/tts/homographs")
def list_supported_homographs():
    """
    Module 2: Lists homographs and context-sensitive pronunciation rules.
    """
    from services.tts import HomographDisambiguator
    return {
        "status": "success",
        "homographs": HomographDisambiguator.list_homographs()
    }


@app.get("/api/tts/formant-vowels")
def list_formant_vowels():
    """
    Module 3: Lists vowel formant targets (F1, F2, F3) for educational simulation.
    """
    from services.tts import FormantSynthesizer
    return {
        "status": "success",
        "vowels": FormantSynthesizer.list_vowels()
    }


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
@app.get("/asr/")
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
