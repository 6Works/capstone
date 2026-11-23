# ⚙️ Stage 3: Natural Language Processing (NLP) Microservice

The **NLP Microservice** takes textual transcripts produced by ASR and extracts semantic meaning: identifying user intent, parsing named entities, evaluating sentiment, and defining tool/function calls.

---

## 📋 Contract Specification

**Abstract Interface**: `INLPService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class INLPService(ABC):
    @abstractmethod
    def parse(self, text: str) -> NLPResult:
        """Parses transcript text and returns structured NLPResult."""
        pass
```

### Output Data Transfer Object (`NLPResult`)

```python
class NLPResult(BaseModel):
    cleaned_text: str             # Normalized input string
    intent: str                   # Classified intent ("greeting", "time_query", etc.)
    confidence: float             # Metric between 0.0 and 1.0
    entities: Dict[str, Any]      # Extracted parameters (e.g. math expressions, dates)
    sentiment: str                # "positive" | "neutral" | "negative"
    tool_calls: List[Dict[str, Any]] # Requested function invocations [{"tool": "...", "args": {...}}]
    latency_sec: float            # Processing duration in seconds
    provider: str                 # Engine name ("rule_based_nlp", "llm_gemini", etc.)
```

---

## 🛠️ Available Providers

| Provider Class | Description | Capabilities |
| :--- | :--- | :--- |
| `RuleBasedNLPService` | High-speed regex pattern engine | Zero latency, deterministic intent & entity extraction, offline. |
| `LLMNLPService` | Large Language Model service | Context-aware reasoning, function calling, extensible to Gemini / Ollama / OpenAI. |

---

## 💻 Standalone Usage Example

```python
from services.nlp import RuleBasedNLPService

nlp = RuleBasedNLPService()
result = nlp.parse("Please calculate 45 * 12")

print(f"Intent: {result.intent} (confidence: {result.confidence})")
print(f"Entities: {result.entities}")
print(f"Tool Calls: {result.tool_calls}")
# Output:
# Intent: calculation_query (confidence: 0.95)
# Entities: {'math_expression': '45 * 12'}
# Tool Calls: [{'tool': 'calculator', 'args': {'expression': '45 * 12'}}]
```

---

## 🔮 Upgrading to LLMs & Function Calling

To upgrade this microservice to an LLM provider:

1. Configure an LLM client (e.g. Google Generative AI or Ollama) in [services/nlp/nlp_services.py](file:///c:/Users/shive/projects/capstonedemo/services/nlp/nlp_services.py):
   ```python
   import google.generativeai as genai

   class GeminiNLPService(INLPService):
       def __init__(self, api_key: str):
           genai.configure(api_key=api_key)
           self.model = genai.GenerativeModel("gemini-1.5-flash")

       def parse(self, text: str) -> NLPResult:
           prompt = f"Analyze intent, entities, sentiment for: '{text}'. Return JSON."
           resp = self.model.generate_content(prompt)
           # Parse structured JSON response into NLPResult
           ...
   ```
2. Any new tool definitions added here flow directly into Stage 4 (Dialogue Management) through `NLPResult.tool_calls`.
