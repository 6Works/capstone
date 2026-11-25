"""
Unit and Integration Test Suite for Decoupled 6-Stage Microservices Architecture.
Verifies SOLID Principles: Dependency Inversion, Liskov Substitution, and Interfaces.
"""

import os
import pytest

from services.contracts import (
    AudioPayload,
    ASRResult,
    NLPResult,
    DialogueTurnResult,
    TTSResult,
    PlaybackResult,
)
from services.audio_input import SyntheticAudioInput, FileAudioInput
from services.asr import GoogleWebSpeechASR, MockASR, WhisperTransformerASR
from services.nlp import RuleBasedNLPService, LLMNLPService
from services.dialogue import ModularDialogueManager, ConversationMemoryStore
from services.tts import (
    Pyttsx3TTSService,
    ElevenLabsTTSService,
    NeuralVoiceCloningTTSService,
    TextNormalizer,
    G2PPhonemizer,
    HomographDisambiguator,
    voice_cloner,
    VoiceCloner,
    AcousticAnalyzer,
    FormantSynthesizer,
)
from services.audio_output import SystemSpeakerOutput, WebStreamAudioOutput
from orchestrator.factory import PipelineFactory
from orchestrator.pipeline import DecoupledVoicePipeline


class TestStage1AudioInputService:
    def test_synthetic_input_generation(self):
        service = SyntheticAudioInput()
        payload = service.capture(duration_sec=0.5)
        assert isinstance(payload, AudioPayload)
        assert payload.source_type == "synthetic"
        assert os.path.exists(payload.file_path)

    def test_file_audio_input(self):
        synth = SyntheticAudioInput()
        synth_payload = synth.capture(duration_sec=0.5)
        
        file_service = FileAudioInput()
        payload = file_service.capture(file_path=synth_payload.file_path)
        assert payload.source_type == "file_upload"
        assert payload.format == "wav"


class TestStage2ASRService:
    def test_mock_asr_liskov_substitution(self):
        """Tests that MockASR satisfies IASRService contract seamlessly."""
        asr = MockASR(fixed_transcript="testing whisper upgrade")
        payload = AudioPayload(source_type="test", format="wav")
        res = asr.transcribe(payload)
        assert isinstance(res, ASRResult)
        assert res.transcript == "testing whisper upgrade"
        assert res.provider == "mock_asr"

    def test_whisper_transformer_contract(self):
        """Verifies Whisper Transformer provider interface, INT8 quantization, and configuration."""
        asr = WhisperTransformerASR(model_size="tiny.en", compute_type="int8")
        assert asr.model_size == "tiny.en"
        assert asr.compute_type == "int8"
        assert asr.device == "cpu"

    def test_whisper_transcription_live(self):
        """Verifies faster-whisper CTranslate2 transcription on acoustic audio."""
        import tempfile, pyttsx3
        temp_wav = os.path.join(tempfile.gettempdir(), "test_whisper_transcribe.wav")
        try:
            engine = pyttsx3.init()
            engine.save_to_file("Hello Mira what is the time", temp_wav)
            engine.runAndWait()

            asr = WhisperTransformerASR(model_size="tiny.en", compute_type="int8")
            payload = AudioPayload(file_path=temp_wav, source_type="test", format="wav")
            res = asr.transcribe(payload)

            assert isinstance(res, ASRResult)
            assert res.status == "success"
            assert "hello" in res.transcript.lower() or "mira" in res.transcript.lower() or "time" in res.transcript.lower()
            assert res.confidence > 0.0
            assert "whisper_tiny.en" in res.provider
            assert res.latency_sec < 5.0
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


class TestStage3NLPService:
    def setup_method(self):
        self.nlp = RuleBasedNLPService()

    def test_intent_and_entity_extraction(self):
        res = self.nlp.parse("what is the current time")
        assert isinstance(res, NLPResult)
        assert res.intent == "time_query"
        assert res.confidence >= 0.9

    def test_calculation_with_tool_call(self):
        res = self.nlp.parse("calculate 12 * 8")
        assert res.intent == "calculation_query"
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["tool"] == "calculator"
        assert res.tool_calls[0]["args"]["expression"] == "12 * 8"


class TestStage4DialogueService:
    def test_memory_store_isolation(self):
        memory = ConversationMemoryStore(max_turns=10)
        memory.append_turn("hi", "hello", "greeting", "greet")
        assert len(memory.turns) == 1
        assert memory.turns[0]["intent"] == "greeting"
        memory.clear()
        assert len(memory.turns) == 0

    def test_dialogue_manager_turn(self):
        dm = ModularDialogueManager(bot_name="Mira")
        nlp_res = NLPResult(
            cleaned_text="calculate 50 + 50",
            intent="calculation_query",
            entities={"math_expression": "50 + 50"},
            provider="test"
        )
        turn_res = dm.process_turn(nlp_res)
        assert isinstance(turn_res, DialogueTurnResult)
        assert "100" in turn_res.response_text
        assert turn_res.turn == 1


class TestStage5TTSService:
    def test_elevenlabs_human_cadence_pause_injection(self):
        service = ElevenLabsTTSService()
        raw_text = "Hello, I am ready. Let us begin."
        enhanced = service.add_human_pauses(raw_text)
        # Should inject natural cadence pauses at punctuation
        assert ", ... " in enhanced
        assert ". ... " in enhanced

    def test_pyttsx3_synthesis(self):
        tts = Pyttsx3TTSService()
        res = tts.synthesize("Unit test speech synthesis")
        assert isinstance(res, TTSResult)
        assert res.status == "success"
        assert os.path.exists(res.file_path)

    def test_text_normalizer_nsw_expansion(self):
        """Verifies Non-Standard Words (NSW) expansion based on course Chapter 3."""
        norm = TextNormalizer.normalize("Item costs $4.50 with 50% discount at 10:30am on 2026 for the ASR API.")
        assert norm["changes_detected"] is True
        text = norm["normalized_text"]
        assert "four dollars and fifty cents" in text
        assert "fifty percent" in text
        assert "ten thirty A M" in text
        assert "twenty twenty-six" in text
        assert "A S R" in text
        assert "A P I" in text

    def test_g2p_phonemization_breakdown(self):
        """Verifies Grapheme-to-Phoneme (G2P) phonetic tokenization into IPA."""
        phon = G2PPhonemizer.phonemize_sentence("Hello Mira acoustic vocoder")
        assert phon["token_count"] == 4
        assert len(phon["ipa_string"]) > 0
        assert "həˈloʊ" in phon["ipa_string"]
        assert "ˈmɪərə" in phon["ipa_string"]
        assert "voʊˈkoʊdər" in phon["ipa_string"]

    def test_speaker_profile_catalog(self):
        """Verifies archetypal speaker profiles (Mira, Alex, Sarah, David)."""
        profiles = voice_cloner.list_profiles()
        assert len(profiles) >= 4
        ids = [p["profile_id"] for p in profiles]
        assert "mira_neural" in ids
        assert "alex_interviewer" in ids
        assert "sarah_executive" in ids
        assert "david_architect" in ids
        mira = voice_cloner.get_profile("mira_neural")
        assert mira.fundamental_freq_f0 > 200.0
        assert len(mira.embedding_vector) == 128

    def test_neural_voice_cloning_tts_synthesis(self):
        """Verifies VITS neural synthesis with speaker profile conditioning."""
        tts = NeuralVoiceCloningTTSService(speaker_profile_id="alex_interviewer")
        res = tts.synthesize("Testing neural voice cloning architecture.")
        assert isinstance(res, TTSResult)
        assert res.status == "success"
        assert "alex_interviewer" in res.provider
        assert res.sample_rate == 16000
        assert os.path.exists(res.file_path)
        assert res.latency_sec < 5.0
        assert "normalized_text" in res.metadata

    def test_zero_shot_acoustic_extraction(self):
        """Verifies extracting F0, centroid, and 128-D embedding from reference audio."""
        import tempfile, wave, numpy as np
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Synthetic 220 Hz harmonic signal (female F0 range)
        signal_data = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm_data = (signal_data * 32767).astype(np.int16)

        temp_wav = os.path.join(tempfile.gettempdir(), f"test_acoustic_{int(np.random.randint(10000))}.wav")
        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm_data.tobytes())

            profile = VoiceCloner.extract_acoustic_profile_from_wav(temp_wav, profile_name="Unit Test Clone")
            assert profile.fundamental_freq_f0 > 150.0
            assert profile.spectral_centroid > 200.0
            assert len(profile.embedding_vector) == 128
            assert -8.0 <= profile.pitch_shift_semitones <= 8.0
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def test_factory_neural_vits_provider(self):
        """Verifies PipelineFactory dynamically injects NeuralVoiceCloningTTSService."""
        pipeline = PipelineFactory.create_pipeline(
            asr_provider="mock",
            tts_provider="neural_vits",
            output_provider="web"
        )
        assert isinstance(pipeline.tts, NeuralVoiceCloningTTSService)

    def test_homograph_disambiguation_rules(self):
        """Verifies context-aware homograph resolution based on Course Chapter 3."""
        past_read = HomographDisambiguator.disambiguate("read", prev_word="have")
        assert past_read["ipa"] == "rɛd"
        assert past_read["sense"] == "past"

        pres_read = HomographDisambiguator.disambiguate("read", prev_word="will")
        assert pres_read["ipa"] == "riːd"
        assert pres_read["sense"] == "present"

        live_adj = HomographDisambiguator.disambiguate("live", next_word="stream")
        assert live_adj["ipa"] == "laɪv"
        assert live_adj["sense"] == "adjective"

        lead_metal = HomographDisambiguator.disambiguate("lead", next_word="pipe")
        assert lead_metal["ipa"] == "lɛd"
        assert lead_metal["sense"] == "metal_noun"

    def test_acoustic_analyzer_mel_spectrogram(self):
        """Verifies extracting 80-band Mel-Spectrogram and F0 pitch contour."""
        import tempfile, wave, numpy as np
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)

        temp_wav = os.path.join(tempfile.gettempdir(), f"test_spec_{int(np.random.randint(9999))}.wav")
        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio.tobytes())

            res = AcousticAnalyzer.analyze_wav(temp_wav)
            assert res["status"] == "success"
            assert res["num_mel_bands"] == 80
            assert res["num_time_frames"] > 0
            assert len(res["mel_spectrogram"]) == 80
            assert 180.0 <= res["mean_f0_hz"] <= 220.0
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def test_formant_synthesizer_klatt_vowels(self):
        """Verifies Klatt digital cascade formant synthesis based on Course Module 3."""
        wav_path = FormantSynthesizer.synthesize_vowel("a", f0=130.0, duration=0.3)
        assert os.path.exists(wav_path)
        assert os.path.getsize(wav_path) > 1000
        try:
            os.remove(wav_path)
        except Exception:
            pass

    def test_latent_space_voice_morphing(self):
        """Verifies interpolating speaker embeddings and acoustic profiles."""
        morphed = voice_cloner.morph_profiles("mira_neural", "alex_interviewer", alpha=0.5)
        assert morphed.profile_id.startswith("morph_")
        assert len(morphed.embedding_vector) == 128
        # Mira is 218Hz, Alex is 124Hz -> 50/50 blend should be around 171Hz
        assert 160.0 <= morphed.fundamental_freq_f0 <= 180.0

    def test_neural_tts_multi_sentence_synthesis(self):
        """Verifies multi-sentence synthesis joins all chunks and preserves speech duration."""
        from services.tts import NeuralVoiceCloningTTSService
        import wave
        service = NeuralVoiceCloningTTSService(speaker_profile_id="mira_neural")
        text = "Hello! This is sentence two. And this is sentence three."
        res = service.synthesize(text)
        assert res.status == "success"
        assert res.file_path and os.path.exists(res.file_path)
        with wave.open(res.file_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            # 3 sentences should take at least 1.5 seconds
            assert dur > 1.5
        try:
            os.remove(res.file_path)
        except Exception:
            pass

    def test_clone_voice_api_endpoint(self):
        """Verifies POST /api/tts/clone-voice handles file upload and returns JSON response."""
        from fastapi.testclient import TestClient
        from server import app
        import tempfile, wave, numpy as np

        client = TestClient(app)
        sr = 16000
        audio = (0.5 * np.sin(2 * np.pi * 210 * np.linspace(0, 0.5, int(sr * 0.5))) * 32767).astype(np.int16)
        temp_wav = os.path.join(tempfile.gettempdir(), f"test_api_clone_{int(np.random.randint(9999))}.wav")
        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio.tobytes())

            with open(temp_wav, "rb") as f:
                resp = client.post("/api/tts/clone-voice", files={"file": ("sample.wav", f, "audio/wav")}, data={"name": "Test Cloned Person"})

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert "profile" in data
            assert data["profile"]["name"] == "Test Cloned Person"
            assert len(data["profile"]["embedding_vector"]) == 128
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


class TestStage6AudioOutputService:
    def test_web_stream_audio_sink(self):
        sink = WebStreamAudioOutput()
        res = sink.play("non_existent_file.wav")
        assert isinstance(res, PlaybackResult)
        assert res.status == "error"


class TestDecoupledPipelineOrchestration:
    def test_factory_dependency_injection(self):
        """Verifies assembling decoupled pipeline with custom providers."""
        pipeline = PipelineFactory.create_pipeline(
            asr_provider="mock",
            tts_provider="pyttsx3",
            nlp_provider="rule_based",
            output_provider="web"
        )
        assert isinstance(pipeline, DecoupledVoicePipeline)
        
        telemetry = pipeline.run_turn_from_text("What time is it?", play_audio=False)
        assert "stages" in telemetry
        assert telemetry["stages"]["stage2_asr"]["transcript"] == "What time is it?"
        assert telemetry["stages"]["stage3_nlp"]["intent"] == "time_query"
        assert "The current time is" in telemetry["stages"]["stage4_dialogue_management"]["response_text"]
        assert telemetry["stages"]["stage5_tts"]["status"] == "success"


class TestRoleServiceAndInterviewerModes:
    """Verifies Role/Persona selection and Dual-Mode Interviewer Engine."""

    def test_role_list_and_configuration(self):
        from services.roles import RoleService
        service = RoleService()
        roles = service.list_roles()
        assert len(roles) >= 4
        role_ids = [r["id"] for r in roles]
        assert "interviewer" in role_ids
        assert "tech_screener" in role_ids
        
        # Verify dual sub-modes in Interviewer
        interviewer = next(r for r in roles if r["id"] == "interviewer")
        mode_ids = [m["id"] for m in interviewer["modes"]]
        assert "solo" in mode_ids
        assert "copilot" in mode_ids

    def test_solo_interviewer_evaluation(self):
        from services.roles import RoleService
        service = RoleService()
        res = service.process_interviewer_turn(
            "I built an event-driven architecture with Kafka partitioned logs and Redis cache.",
            mode="solo"
        )
        assert res["mode"] == "solo"
        assert "bot_response" in res
        assert "evaluation" in res
        assert res["evaluation"]["overall_score"] >= 6
        assert len(res["evaluation"]["strengths"]) > 0

    def test_copilot_interviewer_hud(self):
        from services.roles import RoleService
        service = RoleService()
        res = service.process_interviewer_turn(
            "We used consistent hashing for Redis distributed caching.",
            mode="copilot"
        )
        assert res["mode"] == "copilot"
        assert "copilot_hud" in res
        assert len(res["copilot_hud"]["suggested_probes"]) == 3
        assert "fact_check" in res["copilot_hud"]
        assert "competency_scores" in res["copilot_hud"]


class TestLLMGatewayAndProviders:
    """Verifies Central LLM Gateway, Model Switching, and Graceful Fallback."""

    def test_rule_based_provider_generation(self):
        from services.llm.providers import RuleBasedLLMProvider
        from services.llm.llm_contracts import LLMMessage
        provider = RuleBasedLLMProvider(bot_name="Mira")
        msg = LLMMessage(role="user", content="What is your architecture?")
        res = provider.generate([msg])
        assert res.provider == "rule_based"
        assert len(res.content) > 10
        assert not res.fallback_used

    def test_llm_gateway_status_and_models(self):
        from services.llm.gateway import llm_gateway
        status = llm_gateway.get_status()
        assert "active_provider" in status
        assert "providers" in status
        assert len(status["providers"]) == 5
        provider_ids = [p["provider_id"] for p in status["providers"]]
        assert "gemini" in provider_ids
        assert "openai" in provider_ids
        assert "ollama" in provider_ids
        assert "local_openai" in provider_ids
        assert "rule_based" in provider_ids

    def test_llm_gateway_graceful_fallback(self):
        """Verifies that missing cloud credentials gracefully trigger deterministic fallback."""
        from services.llm.gateway import llm_gateway
        from services.llm.llm_contracts import LLMMessage

        # Ensure fallback is enabled
        llm_gateway.configure(provider_id="gemini", fallback_enabled=True)
        res = llm_gateway.generate([LLMMessage(role="user", content="Hello test")])
        assert res is not None
        assert res.fallback_used is True
        assert "rule_based" in res.provider
        assert len(res.content) > 0

    def test_llm_gateway_runtime_reconfiguration(self):
        from services.llm.gateway import llm_gateway
        new_status = llm_gateway.configure(provider_id="rule_based", model="deterministic-v2", temperature=0.5)
        assert new_status["active_provider"] == "rule_based"
        assert new_status["active_model"] == "deterministic-v2"
        assert new_status["temperature"] == 0.5

    def test_llm_nlp_service_with_gateway(self):
        from services.nlp import LLMNLPService
        nlp = LLMNLPService()
        res = nlp.parse("Hello, how are you today?")
        assert res.intent in ["greeting", "general_query"]
        assert res.cleaned_text == "Hello, how are you today?"
        assert res.confidence > 0.0

    def test_llm_dialogue_manager_with_gateway(self):
        from services.dialogue import LLMDialogueManager
        from services.contracts import NLPResult
        dm = LLMDialogueManager(bot_name="Mira")
        nlp_res = NLPResult(cleaned_text="Who designed this system?", intent="identity_query", provider="test")
        turn_res = dm.process_turn(nlp_res)
        assert turn_res.response_text is not None
        assert turn_res.turn == 1
        assert len(dm.memory.turns) == 1


class TestBlueprintServiceAndIntelligentProfiling:
    """Verifies Three-Layer Plug-and-Play Architecture, Intelligent JD Profiler, and Artifact Rubrics."""

    def test_preset_blueprints_catalog(self):
        from services.blueprints import blueprint_service, RuntimeTier
        blueprints = blueprint_service.list_blueprints()
        assert len(blueprints) >= 6
        bp_ids = [b["blueprint_id"] for b in blueprints]
        assert "sde-dsa-v1" in bp_ids
        assert "ml-datascience-v1" in bp_ids
        assert "fullstack-app-v1" in bp_ids
        assert "cybersecurity-netsec-v1" in bp_ids
        assert "mba-strategy-case-v1" in bp_ids
        assert "system-design-pm-v1" in bp_ids

        # Verify tiers
        ml = next(b for b in blueprints if b["blueprint_id"] == "ml-datascience-v1")
        assert ml["runtime"]["tier"] == RuntimeTier.TIER_2_STATEFUL_SANDBOX

        mba = next(b for b in blueprints if b["blueprint_id"] == "mba-strategy-case-v1")
        assert mba["runtime"]["tier"] == RuntimeTier.TIER_0_ZERO_COMPUTE

    def test_phase_advancement_and_layout(self):
        from services.blueprints import blueprint_service
        blueprint_service.set_active_blueprint("mba-strategy-case-v1")
        state0 = blueprint_service.get_active_blueprint()
        assert state0["active_phase_index"] == 0
        assert "CaseBriefReader" in state0["active_phase"]["layout"]["left"]

        # Advance to phase 2 (Financial Spreadsheet)
        state1 = blueprint_service.advance_phase()
        assert state1["active_phase_index"] == 1
        assert "SpreadsheetGrid" in state1["active_phase"]["layout"]["right"]

    def test_unified_event_bus_telemetry(self):
        from services.blueprints import blueprint_service, TelemetryEvent
        ev = TelemetryEvent(
            session_id="test_sess",
            plugin_id="SpreadsheetGrid",
            event_type="formula_update",
            payload={"cell": "D5", "formula": "=SUM(D3:D4)"}
        )
        res = blueprint_service.record_event(ev)
        assert res["status"] == "recorded"
        assert "=SUM(D3:D4)" in res["context_hint"]

    def test_artifact_rubric_evaluation(self):
        from services.blueprints import blueprint_service, ArtifactBundle
        bundle = ArtifactBundle(
            session_id="test_eval",
            blueprint_id="sde-dsa-v1",
            timestamp=123456789.0,
            plugin_snapshots={
                "CodeEditor": {"code": "def solve(): return True", "language": "python"},
                "StructuredNotes": {"notes": "Edge cases considered: N=0, N=10^6"}
            }
        )
        report = blueprint_service.evaluate_artifacts(bundle)
        assert report["blueprint_id"] == "sde-dsa-v1"
        assert report["composite_score"] >= 7.0
        assert len(report["criteria_breakdown"]) == 4

    def test_intelligent_jd_profiler(self):
        from services.blueprints import blueprint_service, RuntimeTier
        custom_bp = blueprint_service.profile_job_description(
            jd_text="Seeking Senior PyTorch Machine Learning Engineer with experience in deep learning training dynamics and embedding models.",
            seniority="Staff Applied Scientist"
        )
        assert custom_bp is not None
        assert custom_bp.seniority == "Staff Applied Scientist"
        assert custom_bp.runtime.tier in [RuntimeTier.TIER_2_STATEFUL_SANDBOX, RuntimeTier.TIER_1_STATELESS_EXEC]
        assert len(custom_bp.phases) >= 3

    def test_dynamic_layout_customization(self):
        from services.blueprints import blueprint_service
        blueprint_service.set_active_blueprint("sde-dsa-v1")
        res = blueprint_service.customize_active_layout(
            left_plugins=["CaseBriefReader", "SQLWorkbench"],
            right_plugins=["CodeEditor", "LinuxTerminal"],
            ratio="40-60"
        )
        assert res["active_phase"]["layout"]["left"] == ["CaseBriefReader", "SQLWorkbench"]
        assert res["active_phase"]["layout"]["right"] == ["CodeEditor", "LinuxTerminal"]
        assert res["active_phase"]["layout"]["ratio"] == "40-60"
        assert "SQLWorkbench" in res["blueprint"]["all_plugins"]


