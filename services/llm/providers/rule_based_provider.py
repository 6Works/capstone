"""
Deterministic Rule-Based / Mock Fallback LLM Provider.
Implements ILanguageModelProvider for 100% offline availability.
Guarantees the system never crashes with 500 errors when cloud API keys or local servers are absent.
"""

import json
import time
import re
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)


class RuleBasedLLMProvider(ILanguageModelProvider):
    """
    Intelligent Mock & Rule-Based Fallback Provider.
    Simulates structured JSON outputs and conversational dialogue responses with 0 latency.
    """

    def __init__(self, bot_name: str = "Mira"):
        self.bot_name = bot_name

    def get_provider_id(self) -> str:
        return "rule_based"

    def get_display_name(self) -> str:
        return "Rule-Based Deterministic Engine (Offline Safe)"

    def list_models(self) -> List[str]:
        return ["deterministic-v2", "heuristic-evaluator"]

    def check_health(self) -> LLMHealthStatus:
        return LLMHealthStatus(
            provider_id="rule_based",
            provider_name="Rule-Based Deterministic Engine (Offline Safe)",
            is_available=True,
            status_message="Always operational (100% offline availability)",
            available_models=["deterministic-v2", "heuristic-evaluator"],
            requires_api_key=False,
            has_api_key=True,
            endpoint_url="local://in-memory"
        )

    def generate(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()

        # Combine user messages
        user_texts = [m.content for m in messages if m.role == "user"]
        last_user_text = user_texts[-1] if user_texts else ""
        lower = last_user_text.lower()
        sys_lower = (system_prompt or "").lower()

        # 1. JSON Schema Output simulation (e.g. for NLP parse or Interview Evaluation)
        if json_mode or "json" in sys_lower:
            # Check if this is an NLP parse request
            if "intent" in sys_lower or "entity" in sys_lower:
                # Basic regex intent classification
                intent = "general_query"
                confidence = 0.95
                entities: Dict[str, Any] = {}
                tool_calls: List[Dict[str, Any]] = []

                if re.search(r"\b(hello|hi|hey)\b", lower):
                    intent = "greeting"
                elif re.search(r"\b(bye|exit|quit)\b", lower):
                    intent = "farewell"
                elif re.search(r"\b(who are you|your name)\b", lower):
                    intent = "identity_query"
                elif re.search(r"\b(time|what time)\b", lower):
                    intent = "time_query"
                elif re.search(r"\b(weather|forecast)\b", lower):
                    intent = "weather_query"
                elif re.search(r"\b(calculate|what is)\s+([0-9\+\-\*\/\^\.\s]+)", lower):
                    intent = "calculation_query"
                    m = re.search(r"([0-9\+\-\*\/\^\.\s]+)", lower)
                    if m:
                        entities["expression"] = m.group(1).strip()
                        tool_calls.append({"tool": "calculator", "args": {"expression": m.group(1).strip()}})

                pos_words = {"good", "great", "awesome", "excellent", "happy"}
                neg_words = {"bad", "sad", "terrible", "poor", "wrong"}
                words = set(re.findall(r"\b\w+\b", lower))
                sentiment = "positive" if len(words.intersection(pos_words)) > len(words.intersection(neg_words)) else "neutral"

                content = json.dumps({
                    "cleaned_text": last_user_text.strip(),
                    "intent": intent,
                    "confidence": confidence,
                    "entities": entities,
                    "sentiment": sentiment,
                    "tool_calls": tool_calls
                }, indent=2)

            elif "interview" in sys_lower or "evaluation" in sys_lower:
                # Interviewer Evaluation JSON
                tech_keywords = [
                    "cache", "redis", "kafka", "queue", "sharding", "replica",
                    "microservice", "latency", "throughput", "consistency", "acid",
                    "idempotent", "database", "load balancer", "failover", "metric"
                ]
                found = [k for k in tech_keywords if k in lower]
                depth = min(10, max(5, 5 + len(found)))
                clarity = 9 if len(last_user_text.split()) > 10 else 7
                design = min(10, max(6, 6 + (2 if "cache" in lower or "queue" in lower else 0)))
                overall = round((depth + clarity + design) / 3)

                content = json.dumps({
                    "overall_score": overall,
                    "clarity_score": clarity,
                    "depth_score": depth,
                    "system_design_score": design,
                    "strengths": [
                        f"Demonstrated domain understanding citing {', '.join(found[:2])}." if found else "Concisely structured architectural overview."
                    ],
                    "areas_for_growth": [
                        "Include quantitative SLAs or latency metrics (p99)." if "p99" not in lower else "Explore multi-region failure semantics."
                    ],
                    "fact_check_note": "Engineering concepts align with industry distributed patterns.",
                    "suggested_probes": [
                        f"How would you address cache invalidation for {found[0]}?" if found else "What database consistency model does your design guarantee?",
                        "What is your strategy if network partitions isolate a secondary zone?",
                        "How do you monitor and alert on anomalous latency spikes?"
                    ],
                    "interviewer_reply": f"Thank you for walking through that. Your explanation of {found[0] if found else 'the architecture'} is clear. Let's delve into how you handle unexpected network partitions and consistency."
                }, indent=2)

            else:
                content = json.dumps({
                    "status": "success",
                    "message": f"Simulated structured response for: {last_user_text[:50]}"
                })

        else:
            # 2. Natural Conversational Dialogue & Technical Explanations
            if any(k in lower for k in ["consistent hash", "hash ring", "re-shard", "reshard", "virtual node"]):
                content = (
                    "Consistent hashing maps both cluster nodes and keys to positions on a 360-degree virtual ring. "
                    "When nodes join or leave the topology, only k/N keys (where k is the total keys and N is cluster size) "
                    "need relocation to neighboring nodes rather than remapping the entire database. "
                    "Adding virtual replicas per physical host ensures uniform key distribution and eliminates hotspots."
                )
            elif any(k in lower for k in ["cache", "caching", "redis", "memcached", "write-behind", "cache-aside"]):
                content = (
                    "Distributed caching minimizes database read load and latency by keeping active working sets in RAM. "
                    "In a cache-aside pattern, the application queries cache first and lazily populates from the database on a miss, "
                    "while write-behind queuing defers durable persistence to absorb bursts. "
                    "Techniques like TTL jitter, mutex locks, and distributed leases prevent cache stampedes and thundering herd failures."
                )
            elif any(k in lower for k in ["kafka", "queue", "message broker", "rabbitmq", "partition", "event-driven"]):
                content = (
                    "Event-driven message backbones like Apache Kafka decouple producers and consumers through immutable partitioned commit logs. "
                    "Partition keys ensure strict intra-partition message ordering while allowing consumer groups to scale read concurrency horizontally. "
                    "Backpressure buffering and dead-letter queues safeguard downstream microservices during traffic spikes."
                )
            elif any(k in lower for k in ["database", "acid", "postgres", "sql", "sharding", "replica", "isolation"]):
                content = (
                    "Modern transactional databases balance ACID guarantees with horizontal scalability. "
                    "Relational stores rely on Write-Ahead Logging (WAL) and B+ Trees for fast point and range queries, "
                    "while NoSQL architectures use Log-Structured Merge (LSM) trees to optimize write throughput. "
                    "Read replicas offload analytical queries, and range or hash-based sharding partitions massive multi-terabyte datasets."
                )
            elif any(k in lower for k in ["concurrency", "lock", "race condition", "idempotent", "idempotency"]):
                content = (
                    "Safe concurrency requires isolating critical sections through optimistic locking (version timestamps) "
                    "or distributed leases (e.g., Redlock or ZooKeeper ephemerals). "
                    "For distributed state updates, unique idempotency keys stored in an atomic key-value store guarantee that "
                    "network retries do not cause duplicate financial transactions or duplicate message processing."
                )
            elif any(k in lower for k in ["cap theorem", "pacelc", "system design", "latency", "p99", "availability"]):
                content = (
                    "In distributed system design, the CAP theorem states that under a network partition (P), "
                    "systems must choose between strong consistency (CP) or high availability (AP). "
                    "The PACELC theorem extends this: even when operating normally without partitions (E), "
                    "systems trade latency (L) for consistency (C). Production architectures optimize p99 latency "
                    "via connection pooling, read replicas, and asynchronous execution."
                )
            elif any(k in lower for k in ["solid", "microservice", "architecture", "decoupled"]):
                content = (
                    "A SOLID decoupled microservices architecture separates business concerns across dedicated boundaries. "
                    "Using abstract interfaces (Single Responsibility, Dependency Inversion) enables hot-swapping infrastructure "
                    "such as cloud providers, speech pipelines, and execution runtimes without impacting core interview orchestration."
                )
            elif any(k in lower for k in ["who are you", "what is your name"]):
                content = (
                    f"I am {self.bot_name}, your AI Technical Interviewer and Evaluation Co-Pilot. "
                    "I conduct deep-dive architectural assessments, code sandbox challenges, and competency evaluations across fullstack, systems, and data engineering."
                )
            elif "interviewer" in sys_lower or "interview" in sys_lower:
                content = (
                    f"That is a clear technical explanation. Looking closer at production scenarios: "
                    f"how would your design handle an unexpected surge in p99 latency or node failure during peak traffic? "
                    f"Walk me through your recovery strategy and any tradeoffs you would make between consistency and availability."
                )
            else:
                # Intelligent, articulate contextual synthesis
                topics = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", last_user_text) if w.lower() not in {"what", "when", "where", "which", "could", "would", "should", "explain", "about", "please", "using"}]
                top_topic = " ".join(topics[:3]) if topics else "the specified architecture"
                content = (
                    f"Regarding {top_topic}: in scalable production architectures, the primary focus is minimizing tail latency (p99) "
                    f"and preventing cascading failures across service boundaries. Key considerations include establishing explicit idempotency contracts, "
                    f"configuring backpressure controls, and adopting asynchronous queuing or distributed caching where appropriate. "
                    f"When evaluating tradeoffs, prioritize maintainability, fault-tolerance, and resilient degradation under partitioned network conditions."
                )

        latency = round(time.time() - start_time, 4)
        return LLMResponse(
            content=content,
            provider="rule_based",
            model="deterministic-v2",
            latency_sec=latency,
            tokens_prompt=len(last_user_text.split()),
            tokens_completion=len(content.split()),
            fallback_used=False
        )
