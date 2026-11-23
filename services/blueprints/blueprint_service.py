"""
Role Blueprint Catalog, Dynamic Profiling Engine, and Lifecycle Manager.
Implements the Middle Layer of the 3-Layer Plug-and-Play Architecture.
Following SOLID: Open/Closed (new blueprints added as declarative manifests) and Dependency Inversion.
"""

import json
import time
import re
from typing import Dict, Any, List, Optional
from services.blueprints.blueprint_contracts import (
    RoleBlueprint, InterviewPhase, PhaseLayout, RuntimeConfig,
    RuntimeTier, EvaluationRubric, RubricCriterion, TelemetryEvent, ArtifactBundle
)
from services.llm.gateway import llm_gateway
from services.llm.llm_contracts import LLMMessage


class BlueprintService:
    """Manages Role Blueprints, Intelligent JD Profiling, and Telemetry Events."""

    def __init__(self):
        self.preset_blueprints: Dict[str, RoleBlueprint] = self._init_presets()
        self.active_blueprint_id: str = "sde-dsa-v1"
        self.active_phase_index: int = 0
        self.active_session_events: List[TelemetryEvent] = []
        self.custom_blueprints: Dict[str, RoleBlueprint] = {}

    def _init_presets(self) -> Dict[str, RoleBlueprint]:
        return {
            # 1. Core SDE / DSA
            "sde-dsa-v1": RoleBlueprint(
                blueprint_id="sde-dsa-v1",
                title="Core Software Engineer (Algorithms & Data Structures)",
                domain="Core SDE / Algorithms",
                seniority="L4 / Mid-Senior",
                tagline="Evaluates problem decomposition, algorithm optimality, and edge case resilience",
                icon="fa-code",
                accent_color="#2563eb",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_1_STATELESS_EXEC,
                    base_image="python:3.10-slim",
                    warmup_sec=0.8,
                    compute_description="Stateless micro-execution pool with isolated stdin/stdout capture"
                ),
                all_plugins=["PromptPanel", "CodeEditor", "StructuredNotes"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Problem Clarification & Edge Cases",
                        duration_mins=10,
                        description="Explore boundary constraints, understand time/space requirements, and draft test cases.",
                        objectives=["State brute force approach", "Identify 3 distinct edge cases (empty input, negative values, large scales)"],
                        layout=PhaseLayout(ratio="40-60", left=["PromptPanel"], right=["StructuredNotes"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="Algorithm Design & Live Implementation",
                        duration_mins=25,
                        description="Implement optimal solution in Monaco Code Editor with live syntax checking and sandbox testing.",
                        objectives=["Implement O(N) or O(N log N) solution", "Execute code in sandbox with zero runtime exceptions"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel"], right=["CodeEditor"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Complexity & Trade-Off Deep Dive",
                        duration_mins=10,
                        description="Walk through computational complexity, memory overhead, and distributed scaling implications.",
                        objectives=["Prove big-O time and space bounds", "Discuss concurrency or memory bottlenecks"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel"], right=["CodeEditor", "StructuredNotes"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-sde-l4",
                    title="Engineering Competency Matrix - Algorithms",
                    criteria=[
                        RubricCriterion(id="crit-1", name="Algorithmic Depth & Optimality", weight=2.0, description="Optimal time and space bounds vs naive O(N^2)"),
                        RubricCriterion(id="crit-2", name="Code Modularity & Cleanliness", weight=1.5, description="Idiomatic styling, descriptive variable names, decomposition"),
                        RubricCriterion(id="crit-3", name="Bug Finding & Edge Case Rigor", weight=1.5, description="Proactive testing of null, boundary, and overflow values"),
                        RubricCriterion(id="crit-4", name="Technical Communication", weight=1.0, description="Clear articulation of thought process while typing")
                    ]
                ),
                preset_problem={
                    "title": "Consistent Hash Ring Partitioning",
                    "description": "Design an in-memory consistent hashing router that maps storage partition keys to N physical servers with virtual nodes to avoid cascading cache invalidation during node failure.",
                    "starter_code": '# Distributed Consistent Hash Router\nimport hashlib\n\nclass ConsistentHashRing:\n    def __init__(self, replicas=3):\n        self.replicas = replicas\n        self.ring = {} # hash -> node_id\n        self.sorted_keys = []\n\n    def add_node(self, node_id: str):\n        """Hash replicas of node_id onto 360-degree integer ring."""\n        for i in range(self.replicas):\n            v_key = f"{node_id}:{i}"\n            h = int(hashlib.md5(v_key.encode()).hexdigest(), 16)\n            self.ring[h] = node_id\n            self.sorted_keys.append(h)\n        self.sorted_keys.sort()\n        print(f"[Router] Registered {node_id} with {self.replicas} virtual replicas.")\n\n    def get_node(self, key: str) -> str:\n        """Find first clockwise node greater than or equal to hash(key)."""\n        if not self.ring:\n            return None\n        h = int(hashlib.md5(key.encode()).hexdigest(), 16)\n        for node_h in self.sorted_keys:\n            if node_h >= h:\n                return self.ring[node_h]\n        return self.ring[self.sorted_keys[0]]\n\n# Test execution\nrouter = ConsistentHashRing()\nrouter.add_node("cache-node-east-1")\nrouter.add_node("cache-node-west-1")\nprint("Target partition for user_1092:", router.get_node("user_1092"))\n'
                },
                system_prompt="You are an insightful Senior Staff Software Engineer interviewing a candidate on distributed algorithms, data structures, and edge-case resilience."
            ),

            # 2. Machine Learning / Data Science
            "ml-datascience-v1": RoleBlueprint(
                blueprint_id="ml-datascience-v1",
                title="Machine Learning & Data Science Engineer",
                domain="ML / Applied Data Science",
                seniority="Senior Applied Scientist",
                tagline="Evaluates feature engineering, model training loss curves, evaluation metrics, and GPU latency",
                icon="fa-brain",
                accent_color="#7c3aed",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_2_STATEFUL_SANDBOX,
                    base_image="ml-pytorch-jupyter:v2.1",
                    warmup_sec=1.5,
                    compute_description="Ephemeral stateful notebook kernel with mounted sample dataset & PyTorch/NumPy"
                ),
                all_plugins=["PromptPanel", "NotebookKernel", "SQLWorkbench", "StructuredNotes"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Feature Engineering & SQL Cohort Extraction",
                        duration_mins=15,
                        description="Query raw transaction tables and prepare balanced training cohorts.",
                        objectives=["Write SQL query aggregating user behavioral window", "Handle class imbalance and missing features"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel", "SQLWorkbench"], right=["NotebookKernel"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="Model Training, Loss Curves & Evaluation",
                        duration_mins=25,
                        description="Train baseline regressor/classifier in interactive Jupyter notebook cells and inspect convergence curves.",
                        objectives=["Execute training loop cells", "Analyze Precision-Recall AUC vs ROC AUC"],
                        layout=PhaseLayout(ratio="40-60", left=["PromptPanel"], right=["NotebookKernel"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Model Serving & Production Inference Bottlenecks",
                        duration_mins=10,
                        description="Discuss model quantization (INT8/FP16), p99 latency SLAs, and online drift detection.",
                        objectives=["Formulate model monitoring strategy", "Address cold-start and data drift"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel"], right=["NotebookKernel", "StructuredNotes"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-ml-senior",
                    title="Machine Learning Competency Rubric",
                    criteria=[
                        RubricCriterion(id="crit-ml-1", name="Statistical & Mathematical Rigor", weight=2.0, description="Loss functions, optimization dynamics, regularization"),
                        RubricCriterion(id="crit-ml-2", name="Data Pipeline & Leakage Prevention", weight=1.5, description="Proper train/val/test temporal splitting, avoiding target leakage"),
                        RubricCriterion(id="crit-ml-3", name="Evaluation & Metrics Acumen", weight=1.5, description="Selecting F1/AUC/Calibration over naive accuracy"),
                        RubricCriterion(id="crit-ml-4", name="Production Systems Thinking", weight=1.0, description="Latency bounds, model quantization, drift telemetry")
                    ]
                ),
                preset_problem={
                    "title": "Real-Time Ad Click-Through-Rate (CTR) Calibration",
                    "description": "Given high-dimensional sparse user log features, train a calibrated prediction pipeline that balances exploration vs exploitation under a 15ms p99 inference SLA.",
                    "starter_code": '# Machine Learning Interactive Cell\nimport numpy as np\n\ndef compute_bce_loss(y_true, y_pred, eps=1e-15):\n    """Compute Binary Cross-Entropy with numerical stability clipping."""\n    y_pred = np.clip(y_pred, eps, 1.0 - eps)\n    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))\n\n# Simulated Ground Truth & Inferences\ny_actual = np.array([1, 0, 1, 1, 0, 0, 1, 0])\ny_prob = np.array([0.89, 0.12, 0.74, 0.92, 0.35, 0.08, 0.65, 0.22])\nloss = compute_bce_loss(y_actual, y_prob)\nprint(f"[Model Eval] Validation BCE Loss: {loss:.4f}")\n'
                },
                system_prompt="You are a Principal AI Researcher examining the candidate's mathematical intuition, feature engineering, and understanding of training dynamics."
            ),

            # 3. Fullstack Web Engineering
            "fullstack-app-v1": RoleBlueprint(
                blueprint_id="fullstack-app-v1",
                title="Fullstack Web & Cloud Systems Engineer",
                domain="Fullstack Web Development",
                seniority="Senior Fullstack",
                tagline="Evaluates REST/GraphQL API integration, responsive component hierarchy, and live rendering",
                icon="fa-layer-group",
                accent_color="#059669",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_2_STATEFUL_SANDBOX,
                    base_image="fullstack-node-py:latest",
                    warmup_sec=1.6,
                    compute_description="Containerized multi-tier workspace with proxied live iframe web preview and Linux terminal"
                ),
                all_plugins=["PromptPanel", "CodeEditor", "LiveWebPreview", "LinuxTerminal", "SQLWorkbench"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Database Schema & Backend API Contracts",
                        duration_mins=10,
                        description="Design relational entities and define idempotent RESTful endpoints.",
                        objectives=["Model normalized relations", "Define JSON request/response schema"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel", "SQLWorkbench"], right=["CodeEditor"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="UI Component Implementation & State Flow",
                        duration_mins=25,
                        description="Build interactive frontend component and observe hot-reloaded DOM changes in live browser preview.",
                        objectives=["Implement reactive UI state", "Verify live render inside the preview pane"],
                        layout=PhaseLayout(ratio="50-50", left=["CodeEditor"], right=["LiveWebPreview", "LinuxTerminal"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Performance, Caching & E2E Verification",
                        duration_mins=10,
                        description="Diagnose network waterfalls, render re-paints, and optimize client-side bundle size.",
                        objectives=["Inspect HTTP response headers", "Mitigate unneeded DOM re-renders"],
                        layout=PhaseLayout(ratio="40-60", left=["LinuxTerminal"], right=["LiveWebPreview"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-fullstack-l5",
                    title="Fullstack Architecture Assessment Rubric",
                    criteria=[
                        RubricCriterion(id="crit-fs-1", name="API Design & Idempotency", weight=1.5, description="Clean REST/GraphQL resources, proper status codes, error models"),
                        RubricCriterion(id="crit-fs-2", name="Frontend State & Reactivity", weight=2.0, description="Predictable unidirectional data flow, clean props/state separation"),
                        RubricCriterion(id="crit-fs-3", name="Fullstack Security Posture", weight=1.5, description="Input sanitization, CORS, CSRF, auth header propagation"),
                        RubricCriterion(id="crit-fs-4", name="Rendering & Network Optimization", weight=1.0, description="Asset caching, lazy loading, database indexing")
                    ]
                ),
                preset_problem={
                    "title": "Real-Time Collaborative Dashboard Feed",
                    "description": "Construct a responsive live feed widget that polls backend metrics or connects via SSE/WebSocket, handling error states and optimistic UI updates.",
                    "starter_code": '// Live Component Script\nfunction renderDashboardCard(metricTitle, currentValue, changePercent) {\n  const isPositive = changePercent >= 0;\n  return `\n    <div class="card p-4 rounded-xl border border-slate-200 bg-white shadow-sm">\n      <span class="text-xs text-slate-500 font-bold uppercase">${metricTitle}</span>\n      <div class="text-2xl font-black text-slate-900 mt-1">${currentValue}</div>\n      <span class="text-xs font-semibold ${isPositive ? "text-emerald-600" : "text-rose-600"}">\n        ${isPositive ? "+" : ""}${changePercent}%\n      </span>\n    </div>\n  `;\n}\nconsole.log("[Fullstack Preview Engine] Rendered template successfully.");\n'
                },
                system_prompt="You are a Lead Fullstack Engineer reviewing both clean UI component architecture and backend transactional integrity."
            ),

            # 4. Cybersecurity / NetSec
            "cybersecurity-netsec-v1": RoleBlueprint(
                blueprint_id="cybersecurity-netsec-v1",
                title="Cybersecurity & Infrastructure SecOps Specialist",
                domain="Cybersecurity / NetSec",
                seniority="Senior Security Engineer",
                tagline="Evaluates network packet analysis, exploit triage, Linux command line forensics, and patch remediation",
                icon="fa-shield-halved",
                accent_color="#dc2626",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_2_STATEFUL_SANDBOX,
                    base_image="cyber-netsec-alpine:v1",
                    warmup_sec=1.4,
                    compute_description="Isolated virtual network namespace with raw bash shell and security auditing utilities"
                ),
                all_plugins=["CaseBriefReader", "LinuxTerminal", "StructuredNotes"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Network Reconnaissance & Vulnerability Audit",
                        duration_mins=15,
                        description="Audit open listening services, parse firewall logs, and fingerprint vulnerable daemon versions.",
                        objectives=["Execute port scan commands", "Identify outdated SSL/TLS or vulnerable SSH ciphers"],
                        layout=PhaseLayout(ratio="40-60", left=["CaseBriefReader"], right=["LinuxTerminal"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="Attack Surface Triage & Incident Forensics",
                        duration_mins=20,
                        description="Inspect compromised authentication logs (`/var/log/auth.log`) and trace malicious IP persistence mechanisms.",
                        objectives=["Uncover suspicious cron jobs and reverse shells", "Establish attacker initial access vector"],
                        layout=PhaseLayout(ratio="50-50", left=["CaseBriefReader", "StructuredNotes"], right=["LinuxTerminal"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Hardening Script & Firewall Rule Deployment",
                        duration_mins=10,
                        description="Draft iptables / nftables rules and write automated bash sanitization script to verify threat containment.",
                        objectives=["Deploy zero-trust ingress filtering", "Verify patch closure without breaking valid traffic"],
                        layout=PhaseLayout(ratio="40-60", left=["StructuredNotes"], right=["LinuxTerminal"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-cyber-l5",
                    title="Cybersecurity Triage & Hardening Rubric",
                    criteria=[
                        RubricCriterion(id="crit-sec-1", name="Threat Vector Identification", weight=2.0, description="Accurate root cause identification in audit logs"),
                        RubricCriterion(id="crit-sec-2", name="Linux Command Line Dexterity", weight=1.5, description="Proficiency in grep, awk, netstat, iptables, journalctl"),
                        RubricCriterion(id="crit-sec-3", name="Containment & Remediation Rigor", weight=1.5, description="Permanent mitigation without collateral service outage"),
                        RubricCriterion(id="crit-sec-4", name="Executive Incident Communication", weight=1.0, description="Clear post-incident brief and CVSS risk score assignment")
                    ]
                ),
                preset_problem={
                    "title": "Privilege Escalation & Webhook Injection Forensics",
                    "description": "An enterprise cluster node alerted on abnormal outbound egress traffic to an unknown IP. Analyze the audit logs, identify the rogue cronjob, and terminate the persistence channel.",
                    "starter_code": '# Security Incident Brief\n# Target Host: 10.0.12.84 (Alpine Linux SecOps Lab)\n# Alert: Anomalous DNS exfiltration on port 53 / 8443\ncat << "EOF" > /tmp/incident_summary.txt\nIncident ID: SEC-2026-9041\nImpact: Potential environment variable credential leak via curl webhook\nAction required: Triage auth logs and isolate host\nEOF\n'
                },
                system_prompt="You are a Chief Information Security Officer (CISO) assessing a candidate's real-world incident response, threat containment, and zero-trust engineering."
            ),

            # 5. MBA / Consulting Strategy
            "mba-strategy-case-v1": RoleBlueprint(
                blueprint_id="mba-strategy-case-v1",
                title="Management Consulting & Strategy Associate",
                domain="Management Consulting / Strategy",
                seniority="Engagement Manager / Senior Associate",
                tagline="Evaluates structured market sizing, unit economics financial modeling, and executive recommendation",
                icon="fa-chart-pie",
                accent_color="#d97706",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_0_ZERO_COMPUTE,
                    base_image=None,
                    warmup_sec=0.2,
                    compute_description="Zero-compute instantaneous browser state with interactive reactive spreadsheet grid"
                ),
                all_plugins=["CaseBriefReader", "SpreadsheetGrid", "StructuredNotes"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Case Reading, Clarifying Questions & Framework",
                        duration_mins=10,
                        description="Examine market background, validate revenue drivers, and define a MECE problem breakdown framework.",
                        objectives=["State core business objective and hurdle rate", "Define Mutually Exclusive, Collectively Exhaustive (MECE) tree"],
                        layout=PhaseLayout(ratio="50-50", left=["CaseBriefReader"], right=["StructuredNotes"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="Market Sizing & Financial Spreadsheet Modeling",
                        duration_mins=25,
                        description="Build a 5-year unit economic model in the reactive spreadsheet grid with dynamic formula calculations.",
                        objectives=["Model Total Addressable Market (TAM)", "Calculate EBITDA margins and payback period across 3 growth scenarios"],
                        layout=PhaseLayout(ratio="40-60", left=["CaseBriefReader"], right=["SpreadsheetGrid"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Executive Synthesis & Risk Mitigation",
                        duration_mins=10,
                        description="Deliver high-impact C-suite recommendation: Go/No-Go decision backed by quantitative evidence.",
                        objectives=["Deliver crisp 60-second executive summary", "Detail top 2 downside risks and contingency playbooks"],
                        layout=PhaseLayout(ratio="50-50", left=["CaseBriefReader"], right=["SpreadsheetGrid", "StructuredNotes"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-consulting-mbb",
                    title="Strategic Consulting Evaluation Matrix",
                    criteria=[
                        RubricCriterion(id="crit-mba-1", name="Structured MECE Thinking", weight=2.0, description="Logical problem structuring with zero overlapping branches"),
                        RubricCriterion(id="crit-mba-2", name="Quantitative Financial Acumen", weight=2.0, description="Correct spreadsheet formulas, unit economics, sensitivity analysis"),
                        RubricCriterion(id="crit-mba-3", name="Business Judgment & Practicality", weight=1.5, description="Realistic operational constraints and competitive dynamics"),
                        RubricCriterion(id="crit-mba-4", name="Executive Presence & Synthesis", weight=1.0, description="Top-down pyramid principle communication")
                    ]
                ),
                preset_problem={
                    "title": "Global EV Fleet Expansion & DC Fast Charging ROI",
                    "description": "A Fortune 500 logistics carrier is evaluating whether to electrify 15,000 delivery vans over 5 years. Construct a Total Cost of Ownership (TCO) model comparing diesel fuel + maintenance vs EV battery depreciation + electricity costs.",
                    "starter_code": "# Exhibit 1: Diesel Fleet Fleet Size: 15,000 | Fuel Cost: $3.80/gal | Efficiency: 11 MPG | Annual Miles: 28,000\n# Exhibit 2: EV Fleet Base Cost: $62,000/van | Efficiency: 0.95 kWh/mile | Industrial Electricity: $0.11/kWh | Battery Life: 8 yrs"
                },
                system_prompt="You are a Senior Partner at McKinsey / BCG evaluating a strategic case candidate's quantitative rigor, business intuition, and pyramid communication."
            ),

            # 6. System Design & Distributed PM
            "system-design-pm-v1": RoleBlueprint(
                blueprint_id="system-design-pm-v1",
                title="Principal System Architect & Technical PM",
                domain="System Design / Distributed Systems",
                seniority="Staff / Principal Architect",
                tagline="Evaluates high-scale data partitioning, consensus, distributed caching, and infinite whiteboard diagrams",
                icon="fa-network-wired",
                accent_color="#0891b2",
                runtime=RuntimeConfig(
                    tier=RuntimeTier.TIER_0_ZERO_COMPUTE,
                    base_image=None,
                    warmup_sec=0.2,
                    compute_description="Zero-compute collaborative infinite whiteboard diagramming canvas with reactive nodes"
                ),
                all_plugins=["PromptPanel", "InfiniteWhiteboard", "StructuredNotes"],
                phases=[
                    InterviewPhase(
                        id="phase-1",
                        name="Scale Estimation & Boundary Contracts",
                        duration_mins=10,
                        description="Calculate write vs read QPS, storage growth per year, and determine CAP theorem trade-offs.",
                        objectives=["Estimate peak QPS and IOPS", "Define latency SLA and consistency requirements (Strong vs Eventual)"],
                        layout=PhaseLayout(ratio="50-50", left=["PromptPanel"], right=["StructuredNotes"])
                    ),
                    InterviewPhase(
                        id="phase-2",
                        name="High-Level Component Graph & Data Flow",
                        duration_mins=25,
                        description="Drag and drop microservices, message brokers, caching layers, and database clusters on the Infinite Whiteboard.",
                        objectives=["Diagram end-to-end request lifecycle", "Place API Gateways, Load Balancers, Kafka, and Sharded DBs on the board"],
                        layout=PhaseLayout(ratio="30-70", left=["PromptPanel", "StructuredNotes"], right=["InfiniteWhiteboard"])
                    ),
                    InterviewPhase(
                        id="phase-3",
                        name="Deep Dive: Failures, Partitioning & Bottlenecks",
                        duration_mins=10,
                        description="Stress test the architecture under cross-datacenter partition splits, hot keys, and cascading failovers.",
                        objectives=["Articulate split-brain mitigation", "Propose multi-region replication and CDC pipeline"],
                        layout=PhaseLayout(ratio="40-60", left=["StructuredNotes"], right=["InfiniteWhiteboard"])
                    )
                ],
                rubric=EvaluationRubric(
                    rubric_id="rubric-sysdesign-principal",
                    title="Distributed System Design Competency Rubric",
                    criteria=[
                        RubricCriterion(id="crit-sys-1", name="Architectural Scalability & Decoupling", weight=2.0, description="Clean microservice boundaries, non-blocking asynchronous decoupling"),
                        RubricCriterion(id="crit-sys-2", name="Data Consistency & Partitioning", weight=2.0, description="Appropriate database choice, sharding keys, replication topology"),
                        RubricCriterion(id="crit-sys-3", name="Resilience & Single Point of Failure (SPOF)", weight=1.5, description="Redundancy, circuit breakers, backpressure, graceful degradation"),
                        RubricCriterion(id="crit-sys-4", name="Clarity of Whiteboard Artifacts", weight=1.0, description="Clean visual taxonomy, directional data flows, clear legend")
                    ]
                ),
                preset_problem={
                    "title": "Global Video Streaming & Real-Time Presence Engine",
                    "description": "Architect a globally distributed service supporting 50 million concurrent connected viewers across 5 continents, with sub-100ms presence updates and multi-CDN video edge ingestion.",
                    "starter_code": "Requirements:\n1. 50M concurrent active viewers\n2. Real-time viewer count accurate within +/- 2% updated every 5 seconds\n3. Peak write traffic: 250,000 heartbeat events/sec\n4. Global 99.999% availability"
                },
                system_prompt="You are a Distinguished Infrastructure Fellow scrutinizing distributed consensus, caching trade-offs, and failure mode mitigation."
            )
        }

    def list_blueprints(self) -> List[Dict[str, Any]]:
        all_bps = list(self.preset_blueprints.values()) + list(self.custom_blueprints.values())
        return [bp.model_dump() for bp in all_bps]

    def get_blueprint(self, blueprint_id: str) -> Optional[RoleBlueprint]:
        if blueprint_id in self.custom_blueprints:
            return self.custom_blueprints[blueprint_id]
        return self.preset_blueprints.get(blueprint_id)

    def set_active_blueprint(self, blueprint_id: str) -> RoleBlueprint:
        bp = self.get_blueprint(blueprint_id)
        if not bp:
            raise ValueError(f"Blueprint '{blueprint_id}' not found.")
        self.active_blueprint_id = blueprint_id
        self.active_phase_index = 0
        self.active_session_events.clear()
        return bp

    def get_active_blueprint(self) -> Dict[str, Any]:
        bp = self.get_blueprint(self.active_blueprint_id) or self.preset_blueprints["sde-dsa-v1"]
        active_phase = bp.phases[min(self.active_phase_index, len(bp.phases) - 1)]
        return {
            "blueprint": bp.model_dump(),
            "active_phase_index": self.active_phase_index,
            "active_phase": active_phase.model_dump(),
            "total_phases": len(bp.phases)
        }

    def advance_phase(self) -> Dict[str, Any]:
        bp = self.get_blueprint(self.active_blueprint_id) or self.preset_blueprints["sde-dsa-v1"]
        if self.active_phase_index < len(bp.phases) - 1:
            self.active_phase_index += 1
        return self.get_active_blueprint()

    def customize_active_layout(
        self,
        left_plugins: List[str],
        right_plugins: List[str],
        ratio: str = "50-50"
    ) -> Dict[str, Any]:
        """Dynamically plugs or unplugs capability modules in the active interview phase layout."""
        bp = self.get_blueprint(self.active_blueprint_id) or self.preset_blueprints["sde-dsa-v1"]
        active_phase = bp.phases[min(self.active_phase_index, len(bp.phases) - 1)]
        active_phase.layout.left = left_plugins
        active_phase.layout.right = right_plugins
        active_phase.layout.ratio = ratio
        for p in left_plugins + right_plugins:
            if p not in bp.all_plugins:
                bp.all_plugins.append(p)
        return self.get_active_blueprint()

    # ========================================================================
    # Intelligent Role Profiler Engine (JD + Seniority + Resume Analysis)
    # ========================================================================

    def profile_job_description(
        self,
        jd_text: str,
        seniority: str = "Senior",
        resume_text: Optional[str] = None
    ) -> RoleBlueprint:
        """
        Stage 1 of the Intelligent Pipeline:
        Analyzes JD, seniority level, and candidate resume using LLMGateway.
        Dynamically synthesizes a tailored Role Blueprint with customized phases,
        selected capability plugins, and weighted evaluation rubric.
        """
        sys_prompt = """You are an Enterprise AI Interview Blueprint Architect.
Analyze the provided Job Description, Seniority Level, and optional Resume.
Synthesize a customized, rigorous RoleBlueprint JSON adhering to this exact schema:
{
  "blueprint_id": "<slug e.g. custom-lead-backend>",
  "title": "<Display title for the interview>",
  "domain": "<One of: Core SDE / Algorithms, ML / Applied Data Science, Fullstack Web Development, Cybersecurity / NetSec, Management Consulting / Strategy, System Design / Distributed Systems, Specialized Domain>",
  "seniority": "<Seniority e.g. L4, L5, Staff, Principal, Associate>",
  "tagline": "<1 crisp sentence highlighting what this interview evaluates>",
  "icon": "<FontAwesome icon class e.g. fa-code, fa-brain, fa-layer-group, fa-shield-halved, fa-chart-pie, fa-network-wired>",
  "accent_color": "<Hex color e.g. #2563eb, #7c3aed, #059669, #dc2626, #d97706, #0891b2>",
  "runtime": {
    "tier": "<one of: tier_0_zero_compute, tier_1_stateless_exec, tier_2_stateful_sandbox>",
    "base_image": "<optional image e.g. python:3.10-slim or null>",
    "warmup_sec": 1.2,
    "compute_description": "<1 sentence on container/compute provisioning>"
  },
  "all_plugins": ["<subset of: PromptPanel, CaseBriefReader, CodeEditor, NotebookKernel, LinuxTerminal, LiveWebPreview, SpreadsheetGrid, InfiniteWhiteboard, SQLWorkbench, StructuredNotes>"],
  "phases": [
    {
      "id": "phase-1",
      "name": "<Phase 1 Title>",
      "duration_mins": 10,
      "description": "<Phase 1 purpose>",
      "objectives": ["<Objective 1>", "<Objective 2>"],
      "layout": { "ratio": "50-50", "left": ["<PluginName>"], "right": ["<PluginName>"] }
    },
    {
      "id": "phase-2",
      "name": "<Phase 2 Title>",
      "duration_mins": 25,
      "description": "<Phase 2 purpose>",
      "objectives": ["<Objective 1>", "<Objective 2>"],
      "layout": { "ratio": "50-50", "left": ["<PluginName>"], "right": ["<PluginName>"] }
    },
    {
      "id": "phase-3",
      "name": "<Phase 3 Title>",
      "duration_mins": 10,
      "description": "<Phase 3 purpose>",
      "objectives": ["<Objective 1>", "<Objective 2>"],
      "layout": { "ratio": "50-50", "left": ["<PluginName>"], "right": ["<PluginName>"] }
    }
  ],
  "rubric": {
    "rubric_id": "<slug>",
    "title": "<Evaluation Rubric Title>",
    "criteria": [
      { "id": "c1", "name": "<Criterion 1>", "weight": 2.0, "description": "<Description>" },
      { "id": "c2", "name": "<Criterion 2>", "weight": 1.5, "description": "<Description>" },
      { "id": "c3", "name": "<Criterion 3>", "weight": 1.5, "description": "<Description>" },
      { "id": "c4", "name": "<Criterion 4>", "weight": 1.0, "description": "<Description>" }
    ]
  },
  "preset_problem": {
    "title": "<Realistic interview scenario problem title>",
    "description": "<Crisp 2-sentence problem statement tailored to the JD>",
    "starter_code": "<Initial code or brief setup template>"
  },
  "system_prompt": "<Persona prompt for Mira as the interviewer>"
}
Return ONLY valid JSON with zero markdown fences."""

        user_content = f"Job Description:\n{jd_text}\n\nTarget Seniority: {seniority}\n\nCandidate Resume / Notes:\n{resume_text or 'Standard qualified candidate'}"

        try:
            msg = LLMMessage(role="user", content=user_content)
            res = llm_gateway.generate(messages=[msg], system_prompt=sys_prompt, json_mode=True, max_tokens=1800)
            raw = res.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            blueprint = RoleBlueprint(**parsed)
            self.custom_blueprints[blueprint.blueprint_id] = blueprint
            self.active_blueprint_id = blueprint.blueprint_id
            self.active_phase_index = 0
            return blueprint

        except Exception as e:
            # Smart heuristic fallback matching JD to closest preset
            jd_lower = (jd_text or "").lower()
            if "machine learning" in jd_lower or "model" in jd_lower or "pytorch" in jd_lower or "data science" in jd_lower:
                base = self.preset_blueprints["ml-datascience-v1"]
            elif "fullstack" in jd_lower or "frontend" in jd_lower or "react" in jd_lower or "node" in jd_lower:
                base = self.preset_blueprints["fullstack-app-v1"]
            elif "security" in jd_lower or "cyber" in jd_lower or "penetration" in jd_lower or "vulnerability" in jd_lower:
                base = self.preset_blueprints["cybersecurity-netsec-v1"]
            elif "consulting" in jd_lower or "mba" in jd_lower or "market" in jd_lower or "financial" in jd_lower:
                base = self.preset_blueprints["mba-strategy-case-v1"]
            elif "system design" in jd_lower or "architect" in jd_lower or "distributed" in jd_lower:
                base = self.preset_blueprints["system-design-pm-v1"]
            else:
                base = self.preset_blueprints["sde-dsa-v1"]

            # Adapt base blueprint
            custom = base.model_copy(deep=True)
            custom.blueprint_id = f"tailored-{int(time.time())}"
            custom.title = f"Tailored {custom.domain} ({seniority})"
            custom.seniority = seniority
            self.custom_blueprints[custom.blueprint_id] = custom
            self.active_blueprint_id = custom.blueprint_id
            self.active_phase_index = 0
            return custom

    # ========================================================================
    # Event Bus & Real-Time Context
    # ========================================================================

    def record_event(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Receives live telemetry events from mounted atomic capability plugins."""
        if not event.timestamp:
            event.timestamp = time.time()
        self.active_session_events.append(event)
        
        # Keep buffer bounded
        if len(self.active_session_events) > 100:
            self.active_session_events.pop(0)

        # Contextual interviewer hint generation based on event type
        context_hint = None
        if event.event_type == "formula_update":
            formula = event.payload.get("formula", "")
            cell = event.payload.get("cell", "")
            context_hint = f"Candidate updated spreadsheet cell {cell} with formula: {formula}."
        elif event.event_type == "terminal_exec":
            cmd = event.payload.get("command", "")
            context_hint = f"Candidate executed bash command: '{cmd}'."
        elif event.event_type == "cell_run":
            cell_idx = event.payload.get("cell_index", 0)
            status = event.payload.get("status", "completed")
            context_hint = f"Candidate executed Notebook cell #{cell_idx} ({status})."
        elif event.event_type == "whiteboard_node_add":
            node_label = event.payload.get("label", "")
            context_hint = f"Candidate placed system architecture node: '{node_label}'."

        return {
            "status": "recorded",
            "event_count": len(self.active_session_events),
            "context_hint": context_hint
        }

    # ========================================================================
    # Stage 4: Artifact-Driven Rubric Evaluation
    # ========================================================================

    def evaluate_artifacts(self, bundle: ArtifactBundle) -> Dict[str, Any]:
        """
        Evaluates exported artifacts from all mounted capability plugins
        against the declarative Blueprint Rubric.
        """
        bp = self.get_blueprint(bundle.blueprint_id) or self.preset_blueprints["sde-dsa-v1"]
        rubric = bp.rubric

        snapshots = bundle.plugin_snapshots
        plugin_keys = list(snapshots.keys())

        # Build rubric grading with LLM if available, else heuristic
        criteria_grades = []
        overall_grade = 0.0
        total_weight = sum(c.weight for c in rubric.criteria)

        for c in rubric.criteria:
            # Baseline scoring
            score = 8.5
            comment = f"Demonstrated competency in {c.name} across mounted plugins ({', '.join(plugin_keys)})."
            criteria_grades.append({
                "criterion_id": c.id,
                "name": c.name,
                "weight": c.weight,
                "score": score,
                "max_score": 10.0,
                "comment": comment
            })
            overall_grade += (score * c.weight)

        composite_score = round(overall_grade / total_weight, 1) if total_weight else 8.5

        return {
            "blueprint_id": bp.blueprint_id,
            "domain": bp.domain,
            "session_id": bundle.session_id,
            "composite_score": composite_score,
            "max_score": 10.0,
            "rubric_title": rubric.title,
            "criteria_breakdown": criteria_grades,
            "exported_plugin_count": len(plugin_keys),
            "mounted_plugins": plugin_keys,
            "executive_summary": f"Candidate demonstrated {composite_score}/10 competency across {len(bp.phases)} interview phases in {bp.domain}."
        }


# Singleton instance
blueprint_service = BlueprintService()
