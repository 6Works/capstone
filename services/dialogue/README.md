# 💬 Stage 4: Dialogue Management Microservice

The **Dialogue Management Microservice** governs conversation state, session memory, tool execution, and response policy formulation. It decides what the assistant says next based on current and prior turns.

---

## 📋 Contract Specification

**Abstract Interface**: `IDialogueManagerService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class IDialogueManagerService(ABC):
    @abstractmethod
    def process_turn(self, nlp_result: NLPResult) -> DialogueTurnResult:
        """Processes an NLPResult, updates memory, and decides next dialogue turn."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets dialogue state and wipes session memory."""
        pass
```

### Output Data Transfer Object (`DialogueTurnResult`)

```python
class DialogueTurnResult(BaseModel):
    response_text: str            # The bot's reply text
    dialogue_action: str          # Action ("greet", "inform", "calculate", "clarify", etc.)
    state: str                    # Current state machine state ("IN_CONVERSATION", "CLOSING")
    turn: int                     # Turn count for current session
    memory_context: Dict[str, Any]# Snapshot of active session context
    latency_sec: float            # Decision latency in seconds
    provider: str                 # Engine name ("modular_dialogue_manager")
```

---

## 🧠 Memory Sub-module (`ConversationMemoryStore`)

Memory is decoupled into a dedicated `ConversationMemoryStore` class in [services/dialogue/dialogue_services.py](file:///c:/Users/shive/projects/capstonedemo/services/dialogue/dialogue_services.py):
- **Short-Term Turn Buffer**: Maintains ordered list of `(turn, user_text, bot_response, intent, action, timestamp)`.
- **Entity Context Store**: Key-value mapping preserving extracted parameters across multiple conversational turns.
- **Buffer Truncation**: Configurable `max_turns` window to prevent memory leaks in long-running sessions.

---

## 💻 Standalone Usage Example

```python
from services.contracts import NLPResult
from services.dialogue import ModularDialogueManager

dm = ModularDialogueManager(bot_name="Mira")

# 1. Turn 1: Greeting
nlp_turn1 = NLPResult(cleaned_text="Hello", intent="greeting", provider="test")
res1 = dm.process_turn(nlp_turn1)
print(f"Turn 1 ({res1.state}): {res1.response_text}")

# 2. Turn 2: Calculation
nlp_turn2 = NLPResult(
    cleaned_text="calculate 15 + 35",
    intent="calculation_query",
    entities={"math_expression": "15 + 35"},
    provider="test"
)
res2 = dm.process_turn(nlp_turn2)
print(f"Turn 2 ({res2.dialogue_action}): {res2.response_text}")

# 3. Reset Session
dm.reset()
```

---

## 🔮 Upgrading Memory & Tool Execution

This microservice is built to easily plug into:
1. **Vector Database Long-Term Memory**:
   - Swap `ConversationMemoryStore` with a ChromaDB, FAISS, or Pinecone backed retriever for semantic search over past conversations or custom documentation.
2. **Autonomous Tool Execution**:
   - The dialogue engine can inspect `nlp_result.tool_calls`, invoke external Python APIs or MCP tools, and inject the result directly into `response_text`.
3. **LangGraph / StateGraph Transitions**:
   - Replace the internal state machine with a DAG workflow (e.g., triage $\rightarrow$ research $\rightarrow$ answer).
