"""
Role Management & Interviewer Engine Service.
Encapsulates Persona management and dual-mode Interviewer processing.
Following SOLID: Single Responsibility, Open/Closed, and Dependency Inversion.
"""

import time
import re
from typing import Dict, Any, List, Optional
from services.roles.role_contracts import (
    RoleProfile, RoleMode, InterviewEvaluation, CopilotHUD
)
from services.llm.gateway import llm_gateway
from services.llm.llm_contracts import LLMMessage
from services.llm.json_utils import repair_and_parse_with_retry
from services.context import context_manager


class RoleService:
    """
    Decoupled Role Service managing AI personas and sub-mode behaviors.
    """

    def __init__(self):
        self.roles: Dict[str, RoleProfile] = self._init_roles()
        self.active_role_id: str = "interviewer"
        self.active_mode_id: str = "solo"
        self.current_interview_question_idx: int = 0
        self.interview_turns_history: List[Dict[str, Any]] = []

    def _init_roles(self) -> Dict[str, RoleProfile]:
        return {
            "interviewer": RoleProfile(
                id="interviewer",
                name="Intelligent Human-Like Interviewer",
                tagline="Conducts technical & behavioral interviews or assists a human interviewer live",
                category="Hiring & Talent Assessment",
                icon="fa-user-tie",
                accent_color="#4f46e5",
                description="Advanced dual-mode interviewer. Can operate either as an autonomous solo interviewer or as an augmented real-time co-pilot assisting an actual human interviewer.",
                modes=[
                    RoleMode(
                        id="solo",
                        name="Autonomous Solo Interviewer",
                        description="Conducts the full interview autonomously. Mira asks structured questions, evaluates responses, provides vocal feedback, and tracks competency scorecards.",
                        badge="Primary AI Interviewer",
                        icon="fa-microphone"
                    ),
                    RoleMode(
                        id="copilot",
                        name="Human Interviewer Co-Pilot & Assistant",
                        description="Assists a human interviewer sitting in the room. Silently transcribes candidate speech, suggests follow-up probes, performs real-time fact checks, and drafts notes.",
                        badge="Interviewer Co-Pilot HUD",
                        icon="fa-user-astronaut"
                    )
                ],
                system_prompt="You are an empathetic, insightful senior engineering hiring manager evaluating technical depth, clarity, and system design acumen."
            ),
            "tech_screener": RoleProfile(
                id="tech_screener",
                name="Technical Architecture Screener",
                tagline="Probes algorithms, concurrency, microservices, and system boundaries",
                category="Engineering Assessment",
                icon="fa-network-wired",
                accent_color="#059669",
                description="Specialized technical screening bot focusing exclusively on low-level systems, data structures, and high-throughput architectural trade-offs.",
                modes=[
                    RoleMode(
                        id="screening",
                        name="Architecture Deep Dive",
                        description="Presents complex distributed systems problems (CAP theorem, caching, consensus) and challenges trade-offs.",
                        badge="Deep Dive",
                        icon="fa-code-branch"
                    )
                ],
                system_prompt="You are a principal systems architect scrutinizing distributed design trade-offs and performance bottlenecks."
            ),
            "executive_coach": RoleProfile(
                id="executive_coach",
                name="Executive Leadership & Behavioral Coach",
                tagline="Prepares candidates using the STAR methodology for behavioral leadership",
                category="Executive Development",
                icon="fa-chess-knight",
                accent_color="#d97706",
                description="Socratic behavioral coach focusing on conflict resolution, stakeholder management, cross-functional prioritization, and crisis leadership.",
                modes=[
                    RoleMode(
                        id="socratic",
                        name="STAR Method Drill",
                        description="Evaluates Situation, Task, Action, and Result structures in candidate responses.",
                        badge="STAR Rubric",
                        icon="fa-bullseye"
                    )
                ],
                system_prompt="You are an executive coach helping leaders articulate their impact, vision, and organizational achievements."
            ),
            "customer_support": RoleProfile(
                id="customer_support",
                name="Customer Success & Empathy Specialist",
                tagline="Handles high-stakes customer inquiries and de-escalation scenarios",
                category="Customer Operations",
                icon="fa-headset",
                accent_color="#dc2626",
                description="Simulates distressed enterprise customers requiring high emotional intelligence, prompt technical troubleshooting, and structured SLA resolution.",
                modes=[
                    RoleMode(
                        id="simulation",
                        name="Live Escalation Simulation",
                        description="Presents realistic customer outage scenarios and evaluates agent de-escalation strategies.",
                        badge="Empathy Metric",
                        icon="fa-heart-pulse"
                    )
                ],
                system_prompt="You simulate demanding enterprise customer scenarios to train and assess client communication."
            )
        }

    def list_roles(self) -> List[Dict[str, Any]]:
        return [r.model_dump() for r in self.roles.values()]

    def get_active_role(self) -> Dict[str, Any]:
        role = self.roles.get(self.active_role_id, self.roles["interviewer"])
        active_mode = next((m for m in role.modes if m.id == self.active_mode_id), role.modes[0] if role.modes else None)
        return {
            "role": role.model_dump(),
            "active_mode": active_mode.model_dump() if active_mode else None,
            "question_index": self.current_interview_question_idx
        }

    def configure_role(self, role_id: str, mode_id: Optional[str] = None) -> Dict[str, Any]:
        if role_id not in self.roles:
            raise ValueError(f"Unknown role_id: {role_id}")
        self.active_role_id = role_id
        role = self.roles[role_id]
        if mode_id:
            valid_modes = [m.id for m in role.modes]
            if mode_id in valid_modes:
                self.active_mode_id = mode_id
            elif role.modes:
                self.active_mode_id = role.modes[0].id
        elif role.modes:
            self.active_mode_id = role.modes[0].id
        return self.get_active_role()

    def reset_interview(self) -> None:
        self.current_interview_question_idx = 0
        self.interview_turns_history.clear()
        context_manager.reset_session()

    # ========================================================================
    # Intelligent Interviewer Execution Engine (Solo vs Co-Pilot)
    # ========================================================================

    INTERVIEW_STAGES = [
        {
            "stage_name": "Stage 1: Background & Architecture Overview",
            "question": "Welcome! To begin, could you introduce yourself and walk me through a distributed or high-impact system you designed recently?"
        },
        {
            "stage_name": "Stage 2: Scalability, Caching & Bottlenecks",
            "question": "That's very interesting. If read traffic suddenly scaled up by 50x during peak hours, how would you design your caching layers, invalidation policies, and read-replica replication?"
        },
        {
            "stage_name": "Stage 3: Fault Tolerance & Data Consistency",
            "question": "How do you guarantee consistency when a downstream microservice or database partition fails mid-transaction? Walk me through your idempotency and saga/2PC patterns."
        },
        {
            "stage_name": "Stage 4: Incident Response & Engineering Leadership",
            "question": "Tell me about a high-severity production outage you navigated. How did you triage the root cause, communicate with stakeholders, and prevent recurrence?"
        }
    ]

    def process_interviewer_turn(
        self,
        candidate_text: str,
        mode: Optional[str] = None,
        model_name: Optional[str] = None,
        use_json_mode: bool = True,
        enable_schema_repair: bool = True,
        enable_repair_retry: bool = True,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes an interview evaluation turn under either 'solo' or 'copilot' mode.
        Leverages LLMGateway for real LLM reasoning with seamless fallback.
        """
        mode = mode or self.active_mode_id
        idx = self.current_interview_question_idx
        current_stage = self.INTERVIEW_STAGES[min(idx, len(self.INTERVIEW_STAGES) - 1)]

        # Next question index
        next_idx = min(len(self.INTERVIEW_STAGES) - 1, idx + 1)
        next_stage = self.INTERVIEW_STAGES[next_idx]

        # Analyze candidate utterance
        text_lower = candidate_text.lower()
        word_count = len(candidate_text.split())

        # Heuristic base keywords
        tech_keywords = [
            "cache", "redis", "kafka", "queue", "sharding", "replica",
            "microservice", "latency", "throughput", "consistency", "acid",
            "idempotent", "database", "load balancer", "failover", "metric",
            "observability", "distributed", "solid", "decoupled"
        ]
        matched_keywords = [kw for kw in tech_keywords if kw in text_lower]

        # Heuristic scoring baseline
        depth_score = min(10, max(5, 5 + len(matched_keywords) // 2))
        clarity_score = 9 if word_count >= 15 else (6 if word_count < 6 else 7)
        design_score = min(10, max(6, 6 + (1 if "cache" in text_lower or "queue" in text_lower else 0) + (1 if "scale" in text_lower or "distribut" in text_lower else 0)))
        overall_score = round((depth_score + clarity_score + design_score) / 3)

        strengths = [
            f"Demonstrated domain familiarity mentioning {', '.join(matched_keywords[:3])}." if matched_keywords else "Articulated architectural structure clearly."
        ]
        growth_areas = [
            "Consider detailing quantitative metrics (e.g. p99 latency, SLA targets)." if "metric" not in text_lower else "Highlight cross-region failure trade-offs."
        ]
        fact_check = (
            f"Validated: Technical concepts ({', '.join(matched_keywords[:3]) if matched_keywords else 'distributed architecture'}) align with standard engineering patterns."
        )

        bot_reply = (
            f"Thank you for sharing that. I appreciate your focus on {matched_keywords[0] if matched_keywords else 'the architecture'}. "
            f"Let's move to the next area: {next_stage['question']}"
        )

        suggested_probes = [
            f"Can you dive deeper into how you benchmarked {matched_keywords[0] if matched_keywords else 'your components'}?",
            "What would happen to your system if the network between availability zones degraded?",
            "If you were to re-architect this from scratch today, what would you do differently?"
        ]

        llm_model_used = "heuristic_baseline"
        schema_diag: Dict[str, Any] = {
            "raw_valid": True,
            "schema_valid": True,
            "repaired": False,
            "retried": False,
            "fixes": [],
        }

        # Update Context Manager with current stage metadata
        context_manager.update_metadata(
            mode=mode,
            stage_name=current_stage["stage_name"],
            stage_index=idx
        )
        context_prompt_block = context_manager.format_llm_context_prompt(mode=mode)

        # Attempt deep LLM reasoning via LLMGateway
        try:
            sys_prompt = f"""You are Mira, a Senior Staff Technical Interviewer evaluating a candidate's response.
Current Interview Stage: {current_stage['stage_name']}
Current Question Asked: "{current_stage['question']}"
Next Scheduled Stage: "{next_stage['stage_name']}" - "{next_stage['question']}"

{context_prompt_block}

Analyze the candidate's answer and produce ONLY a valid JSON object matching this schema:
{{
  "overall_score": <int 1-10>,
  "clarity_score": <int 1-10>,
  "depth_score": <int 1-10>,
  "system_design_score": <int 1-10>,
  "strengths": [<1-2 bullet points>],
  "areas_for_growth": [<1-2 bullet points>],
  "fact_check_note": "<1 crisp sentence on technical validity>",
  "bot_response": "<Conversational response from Mira in 2-3 spoken sentences. Acknowledge candidate's points, incorporate any active code/workspace context if relevant, give technical feedback, and bridge to asking the next question: '{next_stage['question']}'>",
  "suggested_probes": [<3 sharp follow-up technical questions for an interviewer co-pilot>]
}}
Return ONLY valid JSON."""

            llm_msg = LLMMessage(role="user", content=candidate_text)
            gen_kwargs: Dict[str, Any] = {}
            if seed is not None:
                gen_kwargs["seed"] = seed
            llm_res = llm_gateway.generate(
                messages=[llm_msg],
                model=model_name,
                temperature=temperature,
                system_prompt=sys_prompt,
                json_mode=use_json_mode,
                max_tokens=650,
                **gen_kwargs,
            )

            def _retry_eval(raw_bad: str, keys):
                r_prompt = f"Convert this interview evaluation into ONLY valid JSON with keys [{', '.join(keys)}]."
                return llm_gateway.generate(
                    messages=[LLMMessage(role="user", content=raw_bad[:1200])],
                    model=model_name,
                    temperature=0.0,
                    system_prompt=r_prompt,
                    json_mode=True,
                    max_tokens=500,
                    **gen_kwargs,
                ).content

            repair_out = repair_and_parse_with_retry(
                llm_res.content,
                expected_keys=("overall_score", "clarity_score", "depth_score", "system_design_score"),
                enable_repair=enable_schema_repair,
                retry_fn=_retry_eval if (enable_schema_repair and enable_repair_retry and not llm_res.fallback_used) else None,
            )
            schema_diag = {
                "raw_valid": repair_out["raw_valid"],
                "schema_valid": repair_out["schema_valid"],
                "repaired": repair_out["repaired"],
                "retried": repair_out["retried"],
                "fixes": repair_out["fixes"],
            }
            parsed = repair_out["parsed"]
            if parsed is None:
                raise ValueError("Invalid interview JSON after schema repair")

            overall_score = int(parsed.get("overall_score", overall_score))
            clarity_score = int(parsed.get("clarity_score", clarity_score))
            depth_score = int(parsed.get("depth_score", depth_score))
            design_score = int(parsed.get("system_design_score", design_score))
            if parsed.get("strengths"):
                strengths = parsed["strengths"]
            if parsed.get("areas_for_growth"):
                growth_areas = parsed["areas_for_growth"]
            if parsed.get("fact_check_note"):
                fact_check = parsed["fact_check_note"]
            if parsed.get("bot_response"):
                bot_reply = parsed["bot_response"]
            if parsed.get("suggested_probes"):
                suggested_probes = parsed["suggested_probes"]

            llm_model_used = f"{llm_res.provider}_{llm_res.model}"
            if llm_res.fallback_used:
                llm_model_used += " (fallback)"

        except Exception as e:
            llm_model_used = f"fallback ({str(e)[:30]})"
            schema_diag = {
                "raw_valid": False,
                "schema_valid": False,
                "repaired": False,
                "retried": enable_schema_repair and enable_repair_retry,
                "fixes": [],
            }

        # Record Turn in Unified Context Manager
        context_manager.record_turn(
            candidate_utterance=candidate_text,
            bot_response=bot_reply,
            stage_name=current_stage["stage_name"],
            scores={
                "overall_score": overall_score,
                "clarity_score": clarity_score,
                "depth_score": depth_score,
                "system_design_score": design_score
            },
            key_claims=matched_keywords
        )
        context_snapshot = context_manager.get_snapshot().model_dump()

        evaluation = InterviewEvaluation(
            overall_score=overall_score,
            clarity_score=clarity_score,
            depth_score=depth_score,
            system_design_score=design_score,
            strengths=strengths,
            areas_for_growth=growth_areas,
            fact_check_note=fact_check,
            executive_summary=f"Candidate evaluated via {llm_model_used} on {current_stage['stage_name']} ({overall_score}/10)."
        )

        # Progress question index
        self.current_interview_question_idx = next_idx

        if mode == "solo":
            result = {
                "mode": "solo",
                "stage": current_stage["stage_name"],
                "candidate_text": candidate_text,
                "bot_response": bot_reply,
                "evaluation": evaluation.model_dump(),
                "next_question": next_stage["question"],
                "next_stage": next_stage["stage_name"],
                "turn_index": idx + 1,
                "llm_engine": llm_model_used,
                "schema_diagnostics": schema_diag,
                "context": context_snapshot
            }
        else:
            copilot_hud = CopilotHUD(
                suggested_probes=suggested_probes,
                fact_check=fact_check,
                competency_scores={
                    "Technical Depth": depth_score,
                    "Communication Clarity": clarity_score,
                    "System Architecture": design_score
                },
                key_claims=[f"Discussed {current_stage['stage_name']}", f"Keywords: {', '.join(matched_keywords) if matched_keywords else 'Architectural design'}"],
                interviewer_notes=f"Turn #{idx+1}: Evaluated via {llm_model_used}. Scores: Depth {depth_score}/10, Clarity {clarity_score}/10, System Design {design_score}/10."
            )
            result = {
                "mode": "copilot",
                "stage": current_stage["stage_name"],
                "candidate_text": candidate_text,
                "copilot_hud": copilot_hud.model_dump(),
                "evaluation": evaluation.model_dump(),
                "suggested_probes": suggested_probes,
                "turn_index": idx + 1,
                "llm_engine": llm_model_used,
                "schema_diagnostics": schema_diag,
                "context": context_snapshot
            }

        self.interview_turns_history.append(result)
        return result


# Singleton instance
role_service = RoleService()
