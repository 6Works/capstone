"""
Decoupled Voice Agent Orchestrator.
Demonstrates the Dependency Inversion Principle (DIP).
High-level pipeline depends ONLY on abstract interfaces, never on concrete implementations.
"""

import time
from typing import Dict, Any, Optional

from services.contracts import (
    IAudioInputService,
    IASRService,
    INLPService,
    IDialogueManagerService,
    ITTSService,
    IAudioOutputService,
    AudioPayload,
    ASRResult,
    NLPResult,
    DialogueTurnResult,
    TTSResult,
    PlaybackResult,
)


class DecoupledVoicePipeline:
    """
    Decoupled Orchestrator for the 6-Stage Conversational Voice Agent.
    Zero tight coupling to any specific ASR, NLP, or TTS library.
    """

    def __init__(
        self,
        audio_input: IAudioInputService,
        asr: IASRService,
        nlp: INLPService,
        dialogue: IDialogueManagerService,
        tts: ITTSService,
        audio_output: IAudioOutputService,
    ):
        self.audio_input = audio_input
        self.asr = asr
        self.nlp = nlp
        self.dialogue = dialogue
        self.tts = tts
        self.audio_output = audio_output

    def run_turn_from_audio_payload(
        self,
        audio_payload: AudioPayload,
        play_audio: bool = True
    ) -> Dict[str, Any]:
        """Runs the pipeline starting from an audio payload (Stages 2 -> 6)."""
        turn_start = time.time()
        telemetry: Dict[str, Any] = {
            "timestamp": turn_start,
            "stages": {
                "stage1_audio_input": audio_payload.model_dump()
            }
        }

        # Stage 2: ASR
        asr_res: ASRResult = self.asr.transcribe(audio_payload)
        telemetry["stages"]["stage2_asr"] = asr_res.model_dump()

        if not asr_res.transcript:
            telemetry["total_latency_sec"] = round(time.time() - turn_start, 3)
            return telemetry

        # Stage 3: NLP
        nlp_res: NLPResult = self.nlp.parse(asr_res.transcript)
        telemetry["stages"]["stage3_nlp"] = nlp_res.model_dump()

        # Stage 4: Dialogue Management
        dm_res: DialogueTurnResult = self.dialogue.process_turn(nlp_res)
        telemetry["stages"]["stage4_dialogue_management"] = dm_res.model_dump()

        # Stage 5: TTS
        tts_res: TTSResult = self.tts.synthesize(dm_res.response_text)
        telemetry["stages"]["stage5_tts"] = tts_res.model_dump()

        # Stage 6: Audio Output
        if play_audio and tts_res.file_path:
            play_res: PlaybackResult = self.audio_output.play(tts_res.file_path, block=True)
            telemetry["stages"]["stage6_audio_output"] = play_res.model_dump()
        else:
            telemetry["stages"]["stage6_audio_output"] = {
                "status": "skipped",
                "file": tts_res.file_path,
                "provider": "skipped_or_web"
            }

        telemetry["total_latency_sec"] = round(time.time() - turn_start, 3)
        return telemetry

    def run_turn_from_text(self, text: str, play_audio: bool = False) -> Dict[str, Any]:
        """Runs the pipeline starting directly from text (Simulated Stages 1 & 2 -> 6)."""
        turn_start = time.time()
        
        simulated_input = AudioPayload(
            source_type="simulated_text",
            format="none",
            duration_sec=0.0,
            metadata={"text_injected": text}
        )
        simulated_asr = ASRResult(
            transcript=text,
            confidence=1.0,
            latency_sec=0.001,
            provider="direct_text_injector"
        )

        telemetry: Dict[str, Any] = {
            "timestamp": turn_start,
            "stages": {
                "stage1_audio_input": simulated_input.model_dump(),
                "stage2_asr": simulated_asr.model_dump()
            }
        }

        # Stage 3: NLP
        nlp_res: NLPResult = self.nlp.parse(text)
        telemetry["stages"]["stage3_nlp"] = nlp_res.model_dump()

        # Stage 4: Dialogue
        dm_res: DialogueTurnResult = self.dialogue.process_turn(nlp_res)
        telemetry["stages"]["stage4_dialogue_management"] = dm_res.model_dump()

        # Stage 5: TTS
        tts_res: TTSResult = self.tts.synthesize(dm_res.response_text)
        telemetry["stages"]["stage5_tts"] = tts_res.model_dump()

        # Stage 6: Audio Output
        if play_audio and tts_res.file_path:
            play_res: PlaybackResult = self.audio_output.play(tts_res.file_path, block=True)
            telemetry["stages"]["stage6_audio_output"] = play_res.model_dump()
        else:
            telemetry["stages"]["stage6_audio_output"] = {
                "status": "skipped",
                "file": tts_res.file_path,
                "provider": "web_or_client"
            }

        telemetry["total_latency_sec"] = round(time.time() - turn_start, 3)
        return telemetry

    def run_turn_from_mic(self, play_audio: bool = True) -> Dict[str, Any]:
        """Captures from configured Stage 1 input and runs the full pipeline."""
        payload = self.audio_input.capture()
        return self.run_turn_from_audio_payload(payload, play_audio=play_audio)
