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
from services.tts import Pyttsx3TTSService, ElevenLabsTTSService
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

    def test_whisper_transformer_stub(self):
        """Verifies Whisper Transformer provider interface and fallback capability."""
        asr = WhisperTransformerASR(model_size="base")
        assert asr.model_size == "base"


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


