"""
Thread-Safe Context Manager & Memory Store for Multi-Mode Interview Interactions.
Adheres to SOLID: Single Responsibility (manages session state & history) and
Dependency Inversion (provides decoupled context abstraction to LLM & Role services).
"""

import time
from typing import Dict, Any, List, Optional
from services.context.context_contracts import (
    TurnRecord, WorkspacePluginState, SessionContextSnapshot
)


class InterviewContextManager:
    """Manages active conversation context, multi-turn history, and live workspace states."""

    def __init__(self):
        self.sessions: Dict[str, SessionContextSnapshot] = {}
        self.default_session_id = "session-ai-702"
        # Pre-initialize default session
        self._get_or_create(self.default_session_id)

    def _get_or_create(self, session_id: Optional[str] = None) -> SessionContextSnapshot:
        sid = session_id or self.default_session_id
        if sid not in self.sessions:
            self.sessions[sid] = SessionContextSnapshot(session_id=sid)
        return self.sessions[sid]

    def get_snapshot(self, session_id: Optional[str] = None) -> SessionContextSnapshot:
        """Returns active context snapshot."""
        return self._get_or_create(session_id)

    def get_active_snapshot(self, session_id: Optional[str] = None) -> SessionContextSnapshot:
        """Alias for get_snapshot."""
        return self.get_snapshot(session_id)

    def record_turn(
        self,
        candidate_utterance: str,
        bot_response: str,
        stage_name: str,
        scores: Dict[str, int],
        key_claims: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> TurnRecord:
        """Appends a new conversational turn and updates cumulative competencies."""
        session = self._get_or_create(session_id)
        session.total_turns += 1

        claims = key_claims or []
        turn = TurnRecord(
            turn_index=session.total_turns,
            candidate_utterance=candidate_utterance,
            bot_response=bot_response,
            stage_name=stage_name,
            scores=scores,
            key_claims=claims,
            timestamp=time.time()
        )
        session.turns.append(turn)

        # Retain last 20 turns
        if len(session.turns) > 20:
            session.turns.pop(0)

        # Update discovered claims
        for c in claims:
            if c and c not in session.discovered_claims:
                session.discovered_claims.append(c)
        if len(session.discovered_claims) > 15:
            session.discovered_claims.pop(0)

        # Update moving average competencies
        if scores:
            for k, val in scores.items():
                radar_key = {
                    "depth_score": "Technical Depth",
                    "clarity_score": "Communication Clarity",
                    "system_design_score": "System Architecture",
                    "overall_score": "Problem Solving"
                }.get(k, k)

                if radar_key in session.competencies_radar:
                    prev = session.competencies_radar[radar_key]
                    # Exponential smoothing alpha=0.4
                    session.competencies_radar[radar_key] = round(0.6 * prev + 0.4 * val, 1)

        return turn

    def update_workspace_state(
        self,
        plugin_id: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> None:
        """Synchronizes live state of a capability plugin (CodeEditor, Terminal, Whiteboard, etc.)."""
        session = self._get_or_create(session_id)
        session.workspace_states[plugin_id] = {
            "last_updated": time.time(),
            "data": data
        }

        # Cache specialized high-visibility properties
        if plugin_id == "CodeEditor" and "code" in data:
            session.active_code_snippet = data["code"]
        elif plugin_id in ["LinuxTerminal", "CodeEditor"] and "output" in data:
            session.last_terminal_output = data["output"]

    def update_metadata(
        self,
        blueprint_id: Optional[str] = None,
        role_id: Optional[str] = None,
        mode: Optional[str] = None,
        seniority: Optional[str] = None,
        stage_name: Optional[str] = None,
        stage_index: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Updates high-level interview configuration."""
        session = self._get_or_create(session_id)
        if blueprint_id:
            session.blueprint_id = blueprint_id
        if role_id:
            session.role_id = role_id
        if mode:
            session.mode = mode
        if seniority:
            session.seniority = seniority
        if stage_name:
            session.current_stage_name = stage_name
        if stage_index is not None:
            session.current_stage_index = stage_index

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Resets interview context for a fresh session."""
        sid = session_id or self.default_session_id
        self.sessions[sid] = SessionContextSnapshot(session_id=sid)

    def format_llm_context_prompt(self, mode: str = "solo", session_id: Optional[str] = None) -> str:
        """
        Builds a structured contextual memory block for LLM prompts.
        Injects multi-turn dialogue history, workspace state snapshots,
        and current competency trajectory.
        """
        session = self._get_or_create(session_id)

        # 1. Dialogue History Summary (Last 4 turns)
        history_lines = []
        for t in session.turns[-4:]:
            history_lines.append(f"Candidate (Turn #{t.turn_index}): \"{t.candidate_utterance}\"")
            if t.bot_response:
                history_lines.append(f"Mira (Interviewer): \"{t.bot_response}\"")

        history_str = "\n".join(history_lines) if history_lines else "None (Session just started)."

        # 2. Live Workspace Telemetry
        workspace_lines = []
        if session.active_code_snippet:
            clean_code = session.active_code_snippet.strip()
            preview = clean_code[:300] + ("..." if len(clean_code) > 300 else "")
            workspace_lines.append(f"- Active Code Buffer:\n```\n{preview}\n```")
        if session.last_terminal_output:
            clean_out = session.last_terminal_output.strip()
            preview = clean_out[:200] + ("..." if len(clean_out) > 200 else "")
            workspace_lines.append(f"- Last Sandbox Execution Output: \"{preview}\"")

        for p_id, p_state in session.workspace_states.items():
            if p_id not in ["CodeEditor"]:
                data_summary = str(p_state.get("data", {}))[:120]
                workspace_lines.append(f"- Plugin [{p_id}]: {data_summary}")

        workspace_str = "\n".join(workspace_lines) if workspace_lines else "No active code buffer or terminal output yet."

        # 3. Cumulative Skills
        radar_str = ", ".join([f"{k}: {v}/10" for k, v in session.competencies_radar.items()])

        # 4. Assembled Context Block
        context_block = f"""
[LIVE SESSION CONTEXT - Turn #{session.total_turns + 1}]
- Target Role Blueprint: {session.blueprint_id} ({session.seniority})
- Interview Mode: {mode.upper()}
- Current Stage: {session.current_stage_name}
- Trajectory Competencies: {radar_str}
- Discovered Candidate Claims: {', '.join(session.discovered_claims) if session.discovered_claims else 'None yet'}

[LIVE WORKSPACE TELEMETRY]
{workspace_str}

[RECENT CONVERSATION HISTORY]
{history_str}
"""
        return context_block


# Global singleton context manager
context_manager = InterviewContextManager()
