"""
Service Factory for Dynamic Dependency Injection.
Allows assembling any combination of microservices conforming to the 6-stage contracts.
"""

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
from services.asr import GoogleWebSpeechASR, WhisperTransformerASR, MockASR
from services.nlp import RuleBasedNLPService, LLMNLPService
from services.dialogue import ModularDialogueManager, LLMDialogueManager
from services.tts import Pyttsx3TTSService, GTTSService, ElevenLabsTTSService, NeuralVoiceCloningTTSService
from services.audio_output import SystemSpeakerOutput, WebStreamAudioOutput
from .pipeline import DecoupledVoicePipeline


class PipelineFactory:
    """Factory to instantiate and inject decoupled services into the pipeline."""

    @staticmethod
    def create_pipeline(
        asr_provider: str = "google",
        tts_provider: str = "pyttsx3",
        nlp_provider: str = "rule_based",
        dialogue_provider: str = "modular",
        input_provider: str = "microphone",
        output_provider: str = "speaker",
        bot_name: str = "Mira",
        elevenlabs_api_key: Optional[str] = None
    ) -> DecoupledVoicePipeline:
        
        # 1. Resolve Audio Input
        if input_provider == "file":
            audio_input: IAudioInputService = FileAudioInput()
        elif input_provider == "synthetic":
            audio_input = SyntheticAudioInput()
        else:
            audio_input = MicrophoneAudioInput()

        # 2. Resolve ASR
        if asr_provider in ["whisper", "whisper_small"]:
            asr: IASRService = WhisperTransformerASR(model_size="small.en", compute_type="int8")
        elif asr_provider == "whisper_tiny":
            asr = WhisperTransformerASR(model_size="tiny.en", compute_type="int8")
        elif asr_provider == "mock":
            asr = MockASR()
        else:
            asr = GoogleWebSpeechASR(language="en-US")

        # 3. Resolve NLP
        if nlp_provider == "llm":
            nlp: INLPService = LLMNLPService()
        else:
            nlp = RuleBasedNLPService()

        # 4. Resolve Dialogue Manager
        if dialogue_provider == "llm":
            dialogue: IDialogueManagerService = LLMDialogueManager(bot_name=bot_name)
        else:
            dialogue = ModularDialogueManager(bot_name=bot_name)

        # 5. Resolve TTS
        if tts_provider in ["neural_vits", "neural", "cloned", "voice_cloning"]:
            tts: ITTSService = NeuralVoiceCloningTTSService()
        elif tts_provider == "elevenlabs":
            tts: ITTSService = ElevenLabsTTSService(api_key=elevenlabs_api_key)
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
            audio_output=audio_output
        )
