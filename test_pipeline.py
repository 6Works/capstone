"""
Unit and Integration Test Suite for Decoupled 6-Stage Microservices Architecture.
Verifies SOLID Principles: Dependency Inversion, Liskov Substitution, and Interfaces.
"""

import os
import time
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
            assert len(profile.embedding_vector) in (128, 512)
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
        assert len(morphed.embedding_vector) in (128, 512)
        # Mira is 218Hz, Alex is 124Hz -> 50/50 blend should be around 171Hz
        assert 160.0 <= morphed.fundamental_freq_f0 <= 180.0

    def test_morph_voices_api_endpoint(self):
        """Verifies POST /api/tts/morph-voices returns structured JSON and valid profile."""
        from fastapi.testclient import TestClient
        from server import app
        client = TestClient(app)
        resp = client.post("/api/tts/morph-voices", json={
            "profile_id_a": "mira_neural",
            "profile_id_b": "alex_interviewer",
            "alpha": 0.5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "profile" in data
        assert data["profile"]["profile_id"].startswith("morph_")

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
            assert len(data["profile"]["embedding_vector"]) in (128, 512)
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def test_delete_and_rename_cloned_voice_api(self):
        """Verifies deleting and renaming cloned voice profiles while protecting presets."""
        from fastapi.testclient import TestClient
        from server import app
        from services.tts import voice_cloner, SpeakerProfile

        client = TestClient(app)
        test_id = f"test_delete_{int(time.time()*1000)}"
        dummy_profile = SpeakerProfile(
            profile_id=test_id,
            name="Temporary Voice To Delete",
            description="Testing deletion",
            gender_estimate="male",
            fundamental_freq_f0=130.0,
            pitch_variability_std=15.0,
            spectral_centroid=1800.0,
            spectral_flatness=0.03,
            pitch_shift_semitones=-1.8,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            embedding_vector=[0.01] * 128
        )
        voice_cloner.save_cloned_profile(dummy_profile)
        assert test_id in voice_cloner.profiles

        # 1. Rename profile
        rename_resp = client.patch(f"/api/tts/voices/{test_id}", json={"name": "Newly Renamed Voice"})
        assert rename_resp.status_code == 200
        assert rename_resp.json()["profile"]["name"] == "Newly Renamed Voice"
        assert voice_cloner.get_profile(test_id).name == "Newly Renamed Voice"

        # 2. Prevent deleting preset
        preset_del_resp = client.delete("/api/tts/voices/mira_neural")
        assert preset_del_resp.status_code == 400

        # 3. Delete custom profile
        del_resp = client.delete(f"/api/tts/voices/{test_id}")
        assert del_resp.status_code == 200
        assert test_id not in voice_cloner.profiles
        file_path = os.path.join(voice_cloner.profiles_dir, f"{test_id}.json")
        assert not os.path.exists(file_path)

    def test_edge_tts_service_and_engine_api(self):
        """Verifies Azure EdgeTTSService, engine catalog endpoint, and reference audio endpoints."""
        from fastapi.testclient import TestClient
        from server import app
        from services.tts import EdgeTTSService
        import tempfile

        client = TestClient(app)

        # 1. Verify GET /api/tts/engines returns available multi-engine catalog
        eng_resp = client.get("/api/tts/engines")
        assert eng_resp.status_code == 200
        eng_data = eng_resp.json()
        assert eng_data["status"] == "success"
        engine_ids = [e["id"] for e in eng_data["engines"]]
        assert "edge_tts" in engine_ids
        assert "elevenlabs" in engine_ids
        assert "neural_vits" in engine_ids

        # 2. Test EdgeTTSService directly with Indian English male voice
        edge_svc = EdgeTTSService(voice="en-IN-PrabhatNeural")
        tmp_out = os.path.join(tempfile.gettempdir(), f"test_edge_{int(time.time()*1000)}.mp3")
        try:
            res = edge_svc.synthesize("Hello Sachin and Rahul, this is technical interviewer testing.", output_path=tmp_out)
            assert res.status == "success"
            assert res.file_path and os.path.exists(res.file_path)
            assert os.path.getsize(res.file_path) > 100
        finally:
            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass

        # 3. Test API endpoint POST /api/test/stage5-tts with edge_tts
        synth_resp = client.post("/api/test/stage5-tts", json={
            "text": "Testing multi-engine switching with Azure neural voice.",
            "provider": "edge_tts",
            "voice": "en-US-GuyNeural"
        })
        assert synth_resp.status_code == 200
        data = synth_resp.json()
        assert data["status"] == "success"
        assert "audio_url" in data


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


class TestMLEvaluationMetrics:
    """Verifies Session 1 research evaluation primitives and ONNX research runtime availability."""

    def test_asr_and_slu_metrics(self):
        from ml.metrics import compute_asr_metrics, compute_slu_metrics

        asr_m = compute_asr_metrics(
            references=["Hello Sachin, tell me about PyTorch", "My name is Aradhya"],
            hypotheses=["Hello Sachin tell me about PyTorch", "My name is Aradhya"],
            target_terms=[["Sachin", "PyTorch"], ["Aradhya"]],
        )
        assert asr_m["wer"] == 0.0
        assert asr_m["cer"] == 0.0
        assert asr_m["target_term_recall"] == 1.0

        slu_m = compute_slu_metrics(
            y_true=["greeting", "project_query", "greeting", "farewell"],
            y_pred=["greeting", "project_query", "general_query", "farewell"],
            schema_valid_flags=[True, True, False, True],
        )
        assert slu_m["accuracy"] == 0.75
        assert slu_m["schema_valid_rate"] == 0.75
        assert 0.0 < slu_m["macro_f1"] < 1.0

    def test_speaker_similarity_bootstrap_and_stoi(self):
        import tempfile
        import wave
        import numpy as np
        import sherpa_onnx
        from ml.metrics import (
            compute_speaker_similarity,
            compute_audio_intelligibility,
            paired_bootstrap_test,
            summarize_seeds,
            estimate_cost_per_1k_turns,
        )

        assert sherpa_onnx is not None
        assert compute_speaker_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0
        assert compute_speaker_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

        sig = paired_bootstrap_test(
            baseline_scores=[0.60, 0.62, 0.58, 0.61, 0.59, 0.60],
            candidate_scores=[0.85, 0.87, 0.84, 0.86, 0.88, 0.85],
            n_resamples=500,
            seed=42,
            higher_is_better=True,
        )
        assert sig["observed_improvement"] > 0.2
        assert sig["p_value"] < 0.05

        s = summarize_seeds([0.80, 0.82, 0.84])
        assert s["mean"] == 0.82
        assert s["n_seeds"] == 3
        assert estimate_cost_per_1k_turns(avg_latency_sec=1.0) > 0.0

        # Verify STOI cleanly on a synthetic speech-like tone
        sr = 16000
        t = np.linspace(0, 0.6, int(sr * 0.6))
        wav_pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        p1 = os.path.join(tempfile.gettempdir(), f"stoi_test_{int(time.time()*1000)}.wav")
        try:
            with wave.open(p1, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(wav_pcm.tobytes())
            score = compute_audio_intelligibility(p1, p1)
            assert score >= 0.99
        finally:
            if os.path.exists(p1):
                os.remove(p1)

    def test_campp_neural_speaker_encoder_and_morph_compat(self):
        """Verifies WeSpeaker CAM++ 512-D embeddings discriminate distinct voices and morph safely with 128-D presets."""
        from services.tts import VoiceCloner, NeuralVoiceCloningTTSService, voice_cloner
        from ml.metrics import compute_speaker_similarity

        if not os.path.exists(VoiceCloner.DEFAULT_SPEAKER_ENCODER_PATH):
            pytest.skip("WeSpeaker CAM++ ONNX weights not present locally")

        tts_f = NeuralVoiceCloningTTSService(speaker_profile_id="mira_neural")
        tts_m = NeuralVoiceCloningTTSService(speaker_profile_id="alex_interviewer")
        f_res1 = tts_f.synthesize("Testing consistent voice verification sample one.")
        f_res2 = tts_f.synthesize("Another spoken sentence in the same female voice.")
        m_res = tts_m.synthesize("Deep baritone technical interviewer speaker sample.")

        paths = [r.file_path for r in (f_res1, f_res2, m_res) if r.file_path]
        try:
            a1, sr1 = VoiceCloner.load_audio_signal(f_res1.file_path, target_sr=16000)
            a2, sr2 = VoiceCloner.load_audio_signal(f_res2.file_path, target_sr=16000)
            am, srm = VoiceCloner.load_audio_signal(m_res.file_path, target_sr=16000)

            e_f1, b_f1 = VoiceCloner.extract_speaker_embedding(a1, sr1, use_neural=True)
            e_f2, b_f2 = VoiceCloner.extract_speaker_embedding(a2, sr2, use_neural=True)
            e_m, b_m = VoiceCloner.extract_speaker_embedding(am, srm, use_neural=True)

            assert b_f1 == "wespeaker_campp_lm" and len(e_f1) == 512
            assert b_f2 == "wespeaker_campp_lm" and b_m == "wespeaker_campp_lm"

            sim_same = compute_speaker_similarity(e_f1, e_f2)
            sim_diff = compute_speaker_similarity(e_f1, e_m)
            assert sim_same > sim_diff, f"Expected same-speaker ({sim_same}) > diff-speaker ({sim_diff})"

            # Also verify 512-D extracted profile morphs cleanly with 128-D preset without shape mismatch
            prof_cam = VoiceCloner.extract_acoustic_profile_from_wav(f_res1.file_path, profile_name="Temp CAM++")
            assert prof_cam.encoder_backend == "wespeaker_campp_lm"
            assert len(prof_cam.embedding_vector) == 512
            voice_cloner.save_cloned_profile(prof_cam)
            morphed = voice_cloner.morph_profiles("mira_neural", prof_cam.profile_id, alpha=0.5)
            assert len(morphed.embedding_vector) == 512
            voice_cloner.delete_profile(prof_cam.profile_id)
            voice_cloner.delete_profile(morphed.profile_id)
        finally:
            for p in paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass




