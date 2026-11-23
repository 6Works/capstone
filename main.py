"""
CLI Runner for the Decoupled 6-Stage Microservices Voice Agent.
Supports dynamic dependency injection via CLI arguments.
"""

import sys
import argparse
from orchestrator.factory import PipelineFactory


def main():
    parser = argparse.ArgumentParser(description="Decoupled 6-Stage Conversational Voice Agent")
    parser.add_argument(
        "--mode",
        choices=["demo", "mic", "text"],
        default="demo",
        help="Run mode: 'demo' (automated simulation), 'mic' (live microphone loop), or 'text'"
    )
    parser.add_argument(
        "--asr",
        choices=["google", "whisper", "mock"],
        default="google",
        help="ASR Microservice Provider (default: google)"
    )
    parser.add_argument(
        "--tts",
        choices=["pyttsx3", "gtts", "elevenlabs"],
        default="pyttsx3",
        help="TTS Microservice Provider (default: pyttsx3)"
    )
    parser.add_argument(
        "--nlp",
        choices=["rule_based", "llm"],
        default="rule_based",
        help="NLP Microservice Provider (default: rule_based)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello Mira, can you explain your SOLID microservices architecture?",
        help="Input query text when using --mode text"
    )

    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Decoupled 6-Stage Conversational Voice Agent (SOLID)")
    print(f"  [ASR: {args.asr}]  [NLP: {args.nlp}]  [TTS: {args.tts}]")
    print("="*65 + "\n")

    pipeline = PipelineFactory.create_pipeline(
        asr_provider=args.asr,
        tts_provider=args.tts,
        nlp_provider=args.nlp,
        input_provider="microphone" if args.mode == "mic" else "synthetic",
        output_provider="speaker",
        bot_name="Mira"
    )

    if args.mode == "demo":
        print("=== Running Automated End-to-End Simulation ===")
        test_queries = [
            "Hello Mira, who are you?",
            "What is the current time?",
            "Can you explain your 6 stages?",
            "Calculate 35 plus 25",
            "Goodbye"
        ]

        for q in test_queries:
            print(f"\n--> User Query: \"{q}\"")
            telemetry = pipeline.run_turn_from_text(q, play_audio=True)
            stages = telemetry.get("stages", {})
            asr_p = stages.get("stage2_asr", {}).get("provider")
            nlp_p = stages.get("stage3_nlp", {}).get("provider")
            tts_p = stages.get("stage5_tts", {}).get("provider")
            reply = stages.get("stage4_dialogue_management", {}).get("response_text")
            print(f"  [ASR: {asr_p}] -> [NLP: {nlp_p}] -> [TTS: {tts_p}]")
            print(f"  Bot: \"{reply}\"")
            print(f"  Turn Latency: {telemetry['total_latency_sec']}s")

        print("\nAll simulation turns completed successfully!")

    elif args.mode == "mic":
        print("Starting Live Microphone Loop. Speak into your microphone... (Ctrl+C to stop)")
        while True:
            try:
                res = pipeline.run_turn_from_mic(play_audio=True)
                dm = res.get("stages", {}).get("stage4_dialogue_management", {})
                if dm.get("state") == "CLOSING":
                    print("\n[Conversation Ended by user]")
                    break
            except KeyboardInterrupt:
                print("\nVoice assistant stopped. Goodbye!")
                break
            except Exception as e:
                print(f"Error during turn: {e}")

    elif args.mode == "text":
        res = pipeline.run_turn_from_text(args.text, play_audio=True)
        reply = res.get("stages", {}).get("stage4_dialogue_management", {}).get("response_text")
        print(f"\nResponse: \"{reply}\"")
        print(f"Total Latency: {res['total_latency_sec']}s")


if __name__ == "__main__":
    main()
