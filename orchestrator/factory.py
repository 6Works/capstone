"""
Service Factory for Dynamic Dependency Injection.
Allows assembling any combination of microservices conforming to the 6-stage contracts.
"""

import os
from typing import Optional

from services.contracts import (
    IAudioInputService,
    IASRService,
    INLPService,
    IDialogueManagerService,
    ITTSService,
    IAudioOutputService,
)
from services.audio_input import MicrophoneAudioInput, FileAudioInput, SyntheticAudioInput
from services.asr import (
    GoogleWebSpeechASR,
    WhisperTransformerASR,
    MockASR,
    build_dynamic_bias_prompt,
)
from services.nlp import RuleBasedNLPService, LLMNLPService
from services.dialogue import ModularDialogueManager, LLMDialogueManager
from services.tts import (
    Pyttsx3TTSService,
    GTTSService,
    EdgeTTSService,
    ElevenLabsTTSService,
    NeuralVoiceCloningTTSService,
    PocketTTSCloningService,
)
from services.audio_output import SystemSpeakerOutput, WebStreamAudioOutput
from .pipeline import DecoupledVoicePipeline


DEFAULT_DOMAIN_LEXICON = [
    "FastAPI", "Redis", "PyTorch", "LoRA", "PostgreSQL", "CUDA", "ONNX", "gRPC", "Kafka",
    "CTranslate2", "Kubernetes", "OAuth", "TensorRT", "OpenVINO", "Silero", "Whisper",
    "WeSpeaker", "Kyutai", "GraphQL", "Qwen", "Llama", "SOLID", "Mira",
    "Sachin", "Aradhya", "Nandu", "Bablu", "Rahul", "Priya", "Vikram", "Ananya",
    "Karthik", "Rohan", "Sneha", "Deepa", "Aditya", "Siddharth", "Krishnendu", "Shivendra",
]


class PipelineFactory:
    """Factory to instantiate and inject decoupled services into the pipeline."""

    @staticmethod
    def build_asr_service(
        asr_provider: str = "whisper",
        language: str = "en",
        vad_filter: bool = True,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        fixed_transcript: str = "Hello Mira, what is the time?",
    ) -> IASRService:
        """Instantiates an IASRService with Layer A contextual biasing and anti-loop safeguards."""
        dyn_prompt = initial_prompt if initial_prompt is not None else build_dynamic_bias_prompt(DEFAULT_DOMAIN_LEXICON)
        if asr_provider in ["whisper", "whisper_small"]:
            return WhisperTransformerASR(
                model_size="small.en",
                compute_type="int8",
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                initial_prompt=dyn_prompt,
                hotwords=" ".join(DEFAULT_DOMAIN_LEXICON),
                word_timestamps=True,
                repair_lexicon=DEFAULT_DOMAIN_LEXICON,
            )
        if asr_provider == "whisper_tiny":
            return WhisperTransformerASR(
                model_size="tiny.en",
                compute_type="int8",
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                initial_prompt=dyn_prompt,
                hotwords=None,
                word_timestamps=True,
                repair_lexicon=DEFAULT_DOMAIN_LEXICON,
                condition_on_previous_text=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
                hallucination_silence_threshold=0.4,
            )
        if asr_provider == "mock":
            return MockASR(fixed_transcript=fixed_transcript)
        lang_code = "en-US" if language == "en" else language
        return GoogleWebSpeechASR(language=lang_code)

    @staticmethod
    def create_pipeline(
        asr_provider: str = "google",
        tts_provider: str = "pyttsx3",
        nlp_provider: str = "rule_based",
        dialogue_provider: str = "modular",
        input_provider: str = "microphone",
        output_provider: str = "speaker",
        bot_name: str = "Mira",
        elevenlabs_api_key: Optional[str] = None,
        speaker_profile_id: str = "mira_neural",
        slu_model_name: Optional[str] = None,
    ) -> DecoupledVoicePipeline:

        # 1. Resolve Audio Input
        if input_provider == "file":
            audio_input: IAudioInputService = FileAudioInput()
        elif input_provider == "synthetic":
            audio_input = SyntheticAudioInput()
        else:
            audio_input = MicrophoneAudioInput()

        # 2. Resolve ASR (Session 5 capacity-gated prompt, hotwords & phonetic repair defaults)
        asr = PipelineFactory.build_asr_service(asr_provider=asr_provider)

        # 3. Resolve NLP (Session 6 shared JSON repair + single-retry + ASR pre-repair defaults)
        if nlp_provider == "llm":
            resolved_model = slu_model_name or os.getenv("CAPSTONE_SLU_MODEL", None)
            nlp: INLPService = LLMNLPService(
                model_name=resolved_model,
                repair_lexicon=DEFAULT_DOMAIN_LEXICON,
                enable_schema_repair=True,
                enable_repair_retry=True,
                use_json_mode=True,
            )
        else:
            nlp = RuleBasedNLPService(repair_lexicon=DEFAULT_DOMAIN_LEXICON)

        # 4. Resolve Dialogue Manager
        if dialogue_provider == "llm":
            dialogue: IDialogueManagerService = LLMDialogueManager(bot_name=bot_name)
        else:
            dialogue = ModularDialogueManager(bot_name=bot_name)

        # 5. Resolve TTS (Session 3/7 Pocket TTS INT8 default cloning with automatic Piper fallback)
        if tts_provider in ["pocket_tts", "pocket", "zero_shot"]:
            tts: ITTSService = PocketTTSCloningService(speaker_profile_id=speaker_profile_id)
        elif tts_provider in ["edge_tts", "azure"]:
            tts = EdgeTTSService()
        elif tts_provider in ["neural_vits", "neural", "cloned", "voice_cloning"]:
            tts = NeuralVoiceCloningTTSService(speaker_profile_id=speaker_profile_id)
        elif tts_provider == "elevenlabs":
            tts = ElevenLabsTTSService(api_key=elevenlabs_api_key)
        elif tts_provider == "gtts":
            tts = GTTSService()
        else:
            tts = Pyttsx3TTSService()

        # 6. Resolve Audio Output
        if output_provider == "web":
            audio_output: IAudioOutputService = WebStreamAudioOutput()
        else:
            audio_output = SystemSpeakerOutput()

        return DecoupledVoicePipeline(
            audio_input=audio_input,
            asr=asr,
            nlp=nlp,
            dialogue=dialogue,
            tts=tts,
            audio_output=audio_output,
        )

    @staticmethod
    def create_preset(tier: str = "full", **overrides) -> DecoupledVoicePipeline:
        """
        Instantiates empirically validated Pareto pipeline profiles from Session 8:
          - 'full': small.en + hotwords/repair + qwen2.5:3b/repaired + Pocket TTS INT8 (peak accuracy)
          - 'edge': tiny.en + anti-loop/repair + qwen2.5:3b/repaired + Pocket TTS INT8 (48% lower cost)
        """
        chosen = (tier or os.getenv("CAPSTONE_PIPELINE_TIER", "full")).lower()
        defaults = {
            "asr_provider": "whisper_tiny" if chosen == "edge" else "whisper",
            "tts_provider": "pocket_tts",
            "nlp_provider": "llm",
            "dialogue_provider": "llm",
            "input_provider": "file",
            "output_provider": "web",
            "slu_model_name": overrides.pop("slu_model_name", os.getenv("CAPSTONE_SLU_MODEL", "qwen2.5:3b")),
        }
        defaults.update(overrides)
        return PipelineFactory.create_pipeline(**defaults)
