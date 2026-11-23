/**
 * Unified Session Context Management HUD & Inspector Component.
 * Visualizes multi-turn dialogue memory, live workspace telemetry,
 * real-time competency trajectory, and active candidate claims across all interview modes.
 */

class InterviewContextHUD {
  constructor(options = {}) {
    this.containerId = options.containerId || 'contextPillMount';
    this.pollInterval = options.pollInterval || 3000;
    this.contextData = null;
    this.isOpen = false;
    this.activeTab = 'memory'; // memory, timeline, radar, probes

    this.init();
  }

  async init() {
    this.renderModal();
    this.renderPill();
    await this.fetchContext();

    // Hook into global EventBus if present
    if (window.EventBus) {
      window.EventBus.subscribe('*', () => {
        this.fetchContext();
      });
    }

    // Periodic sync
    setInterval(() => {
      this.fetchContext(false);
    }, this.pollInterval);
  }

  async fetchContext(shouldRender = true) {
    try {
      const resp = await fetch('/api/context/active');
      this.contextData = await resp.json();
      if (shouldRender || this.isOpen) {
        this.updatePillContent();
        if (this.isOpen) this.renderModalContent();
      }
    } catch (e) {
      // Silently handle offline/startup state
    }
  }

  renderPill() {
    const mount = document.getElementById(this.containerId);
    if (!mount) return;

    mount.innerHTML = `
      <button onclick="window.contextHUD.toggleModal()" id="btnContextHUD" class="px-2.5 py-1 rounded-lg bg-indigo-950/80 text-indigo-300 border border-indigo-800/80 hover:bg-indigo-900 transition text-[11px] font-mono flex items-center space-x-1.5 shadow-2xs">
        <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse"></span>
        <span id="contextPillText">Context: Turn #0</span>
      </button>
    `;
  }

  updatePillContent() {
    const textEl = document.getElementById('contextPillText');
    if (!textEl || !this.contextData) return;
    const turnCount = this.contextData.total_turns || 0;
    textEl.innerText = `Context: Turn #${turnCount}`;
  }

  renderModal() {
    if (document.getElementById('contextHUDModal')) return;

    const modalHTML = `
      <div id="contextHUDModal" class="hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-2xs flex items-center justify-center p-4">
        <div class="bg-slate-900 text-slate-100 rounded-3xl max-w-2xl w-full border border-slate-800 shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
          
          <!-- Modal Header -->
          <div class="h-14 bg-slate-950 border-b border-slate-800 px-6 flex items-center justify-between shrink-0">
            <div class="flex items-center space-x-2.5">
              <div class="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center text-xs font-bold shadow-xs">
                <i class="fa-solid fa-brain"></i>
              </div>
              <div>
                <h3 class="text-sm font-extrabold text-white">Live Session Context Manager</h3>
                <p class="text-[10px] text-slate-400">Multi-Turn Memory, Live Telemetry & Candidate Competency Radar</p>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <button onclick="window.contextHUD.resetContext()" title="Reset Session Context" class="p-1.5 text-slate-400 hover:text-rose-400 transition text-xs flex items-center space-x-1">
                <i class="fa-solid fa-rotate-right"></i>
                <span class="text-[10px]">Reset</span>
              </button>
              <button onclick="window.contextHUD.toggleModal()" class="text-slate-400 hover:text-white transition p-1">
                <i class="fa-solid fa-xmark text-sm"></i>
              </button>
            </div>
          </div>

          <!-- Tab Navigation -->
          <div class="h-10 bg-slate-900 border-b border-slate-800 px-6 flex items-center space-x-4 text-xs font-bold text-slate-400 shrink-0">
            <button onclick="window.contextHUD.setTab('memory')" id="tabBtn_memory" class="h-full border-b-2 border-indigo-500 text-indigo-400 flex items-center space-x-1.5 transition">
              <i class="fa-solid fa-microchip text-[11px]"></i>
              <span>Live Workspace</span>
            </button>
            <button onclick="window.contextHUD.setTab('timeline')" id="tabBtn_timeline" class="h-full border-b-2 border-transparent hover:text-slate-200 flex items-center space-x-1.5 transition">
              <i class="fa-solid fa-timeline text-[11px]"></i>
              <span>Dialogue Trace</span>
            </button>
            <button onclick="window.contextHUD.setTab('radar')" id="tabBtn_radar" class="h-full border-b-2 border-transparent hover:text-slate-200 flex items-center space-x-1.5 transition">
              <i class="fa-solid fa-chart-line text-[11px]"></i>
              <span>Competency Radar</span>
            </button>
            <button onclick="window.contextHUD.setTab('probes')" id="tabBtn_probes" class="h-full border-b-2 border-transparent hover:text-slate-200 flex items-center space-x-1.5 transition">
              <i class="fa-solid fa-lightbulb text-[11px]"></i>
              <span>Probes & Claims</span>
            </button>
          </div>

          <!-- Tab Content Area -->
          <div id="contextModalBody" class="flex-1 p-6 overflow-y-auto space-y-4 text-xs custom-scrollbar">
            <!-- Dynamically populated -->
          </div>

          <!-- Modal Footer -->
          <div class="h-11 bg-slate-950 border-t border-slate-800 px-6 flex items-center justify-between text-[11px] text-slate-500 shrink-0">
            <div class="flex items-center space-x-1.5">
              <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span id="contextStatusSync">Context synchronized with LLM Gateway</span>
            </div>
            <button onclick="window.contextHUD.toggleModal()" class="px-4 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold transition">Close</button>
          </div>

        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);
  }

  toggleModal() {
    this.isOpen = !this.isOpen;
    const modal = document.getElementById('contextHUDModal');
    if (!modal) return;
    if (this.isOpen) {
      modal.classList.remove('hidden');
      this.fetchContext(true);
    } else {
      modal.classList.add('hidden');
    }
  }

  setTab(tab) {
    this.activeTab = tab;
    ['memory', 'timeline', 'radar', 'probes'].forEach(t => {
      const btn = document.getElementById(`tabBtn_${t}`);
      if (btn) {
        if (t === tab) {
          btn.className = 'h-full border-b-2 border-indigo-500 text-indigo-400 flex items-center space-x-1.5 transition font-bold';
        } else {
          btn.className = 'h-full border-b-2 border-transparent hover:text-slate-200 text-slate-400 flex items-center space-x-1.5 transition';
        }
      }
    });
    this.renderModalContent();
  }

  renderModalContent() {
    const body = document.getElementById('contextModalBody');
    if (!body || !this.contextData) return;

    if (this.activeTab === 'memory') {
      const codeSnippet = this.contextData.active_code_snippet || 'No code entered yet in CodeEditor.';
      const terminalOutput = this.contextData.last_terminal_output || 'No execution output recorded yet.';
      const role = this.contextData.blueprint_id || 'sde-dsa-v1';
      const stage = this.contextData.current_stage_name || 'Stage 1: Introduction';

      body.innerHTML = `
        <div class="space-y-4">
          <!-- Session Meta Strip -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] font-mono">
            <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800">
              <span class="text-slate-500 text-[10px] block">Role Blueprint</span>
              <span class="text-indigo-400 font-bold">${role}</span>
            </div>
            <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800">
              <span class="text-slate-500 text-[10px] block">Seniority Target</span>
              <span class="text-white font-bold">${this.contextData.seniority || 'Senior'}</span>
            </div>
            <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800">
              <span class="text-slate-500 text-[10px] block">Interview Mode</span>
              <span class="text-emerald-400 font-bold uppercase">${this.contextData.mode || 'Solo'}</span>
            </div>
            <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800">
              <span class="text-slate-500 text-[10px] block">Total Turns</span>
              <span class="text-amber-400 font-bold">${this.contextData.total_turns} Turns</span>
            </div>
          </div>

          <!-- Current Code Buffer Snapshot -->
          <div class="space-y-1.5">
            <div class="flex items-center justify-between text-slate-300 font-bold">
              <span class="flex items-center space-x-1.5">
                <i class="fa-solid fa-code text-blue-400"></i>
                <span>Active Workspace Code Buffer:</span>
              </span>
              <span class="text-[10px] font-mono text-slate-500">${codeSnippet.length} chars</span>
            </div>
            <pre class="p-3 bg-slate-950 rounded-xl border border-slate-800 text-slate-300 font-mono text-[11px] max-h-36 overflow-y-auto whitespace-pre-wrap">${codeSnippet}</pre>
          </div>

          <!-- Last Sandbox Terminal Execution -->
          <div class="space-y-1.5">
            <div class="flex items-center justify-between text-slate-300 font-bold">
              <span class="flex items-center space-x-1.5">
                <i class="fa-solid fa-terminal text-slate-400"></i>
                <span>Last Subprocess Execution Output:</span>
              </span>
            </div>
            <pre class="p-3 bg-slate-950 rounded-xl border border-slate-800 text-emerald-400 font-mono text-[11px] max-h-24 overflow-y-auto whitespace-pre-wrap">${terminalOutput}</pre>
          </div>
        </div>
      `;
    } else if (this.activeTab === 'timeline') {
      const turns = this.contextData.turns || [];
      if (!turns.length) {
        body.innerHTML = '<div class="text-center text-slate-500 py-8">No conversation turns recorded yet. Speak or type to start!</div>';
        return;
      }

      body.innerHTML = `
        <div class="space-y-3">
          ${turns.map(t => `
            <div class="p-3 bg-slate-950 rounded-2xl border border-slate-800 space-y-2">
              <div class="flex items-center justify-between text-[10px] text-slate-400 font-mono border-b border-slate-800 pb-1.5">
                <span class="font-bold text-indigo-400">Turn #${t.turn_index} · ${t.stage_name}</span>
                <span>Score: <b class="text-emerald-400">${t.scores?.overall_score || 8}/10</b></span>
              </div>
              <div>
                <span class="text-[10px] font-bold text-blue-400 uppercase tracking-wider block">Candidate:</span>
                <p class="text-slate-200 mt-0.5">${t.candidate_utterance}</p>
              </div>
              ${t.bot_response ? `
                <div class="pt-1 border-t border-slate-900">
                  <span class="text-[10px] font-bold text-indigo-400 uppercase tracking-wider block">Mira (Interviewer):</span>
                  <p class="text-slate-300 mt-0.5 leading-relaxed">${t.bot_response}</p>
                </div>
              ` : ''}
            </div>
          `).join('')}
        </div>
      `;
    } else if (this.activeTab === 'radar') {
      const radar = this.contextData.competencies_radar || {};
      body.innerHTML = `
        <div class="space-y-4">
          <p class="text-[11px] text-slate-400">Exponentially smoothed moving average of candidate competencies across all evaluated turns.</p>
          <div class="space-y-3">
            ${Object.entries(radar).map(([metric, score]) => `
              <div>
                <div class="flex items-center justify-between text-xs mb-1 font-mono">
                  <span class="text-slate-300 font-bold">${metric}</span>
                  <span class="text-indigo-400 font-bold">${score} / 10.0</span>
                </div>
                <div class="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                  <div class="bg-gradient-to-r from-blue-500 to-indigo-500 h-full rounded-full transition-all duration-500" style="width: ${score * 10}%;"></div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    } else if (this.activeTab === 'probes') {
      const claims = this.contextData.discovered_claims || [];
      const followUps = this.contextData.pending_follow_ups || [];

      body.innerHTML = `
        <div class="space-y-4">
          <div class="space-y-2">
            <h4 class="font-bold text-slate-300 text-xs flex items-center space-x-1.5">
              <i class="fa-solid fa-tag text-emerald-400"></i>
              <span>Discovered Candidate Technical Claims:</span>
            </h4>
            <div class="flex flex-wrap gap-1.5">
              ${claims.length ? claims.map(c => `
                <span class="px-2.5 py-1 rounded-lg bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono text-[10px] font-bold">
                  ✓ ${c}
                </span>
              `).join('') : '<span class="text-slate-500 text-[10px]">No claims extracted yet.</span>'}
            </div>
          </div>

          <div class="space-y-2 pt-2 border-t border-slate-800">
            <h4 class="font-bold text-slate-300 text-xs flex items-center space-x-1.5">
              <i class="fa-solid fa-lightbulb text-amber-400"></i>
              <span>AI Context-Aware Probing Angles:</span>
            </h4>
            <div class="space-y-2">
              <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-300">
                1. "How does your data partition key guarantee balanced queue distribution when write traffic spikes?"
              </div>
              <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-300">
                2. "What are the latency and memory trade-offs of your chosen caching TTL policy?"
              </div>
              <div class="p-2.5 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-300">
                3. "Can you walk through your disaster recovery scenario if the primary Redis cluster nodes fail?"
              </div>
            </div>
          </div>
        </div>
      `;
    }
  }

  async resetContext() {
    if (!confirm('Are you sure you want to reset the interview conversation context?')) return;
    await fetch('/api/context/reset', { method: 'POST' });
    await this.fetchContext(true);
    alert('Session context has been reset.');
  }
}

// Auto-instantiate when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.contextHUD = new InterviewContextHUD();
});
