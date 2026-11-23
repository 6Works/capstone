/**
 * Central LLM Model Gateway & Orchestrator UI Component.
 * Provides a global header pill and a full-featured modal for hot-swapping
 * between Gemini, OpenAI, Ollama (Local), LM Studio (Local), and Rule-Based Fallback.
 */

class LLMGatewayUI {
  constructor(mountContainerId = 'llmPillMount') {
    this.mountContainer = document.getElementById(mountContainerId);
    this.status = null;
    this.init();
  }

  async init() {
    this.injectModal();
    await this.fetchStatus();
    this.renderPill();
  }

  async fetchStatus(forceRefresh = false) {
    try {
      const resp = await fetch(`/api/llm/status${forceRefresh ? '?force_refresh=true' : ''}`);
      this.status = await resp.json();
      return this.status;
    } catch (e) {
      console.warn('[LLMGatewayUI] Failed to fetch LLM status:', e);
      return null;
    }
  }

  renderPill() {
    if (!this.mountContainer) return;

    const activeProvider = this.status ? this.status.active_provider : 'rule_based';
    const activeModel = this.status ? this.status.active_model : 'deterministic';
    
    let badgeColor = 'bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100';
    let icon = 'fa-solid fa-bolt text-indigo-600';
    let label = 'LLM: Gemini';

    if (activeProvider === 'gemini') {
      label = `LLM: Gemini (${activeModel})`;
      icon = 'fa-brands fa-google text-blue-600';
      badgeColor = 'bg-blue-50 border-blue-200 text-blue-800 hover:bg-blue-100';
    } else if (activeProvider === 'openai') {
      label = `LLM: OpenAI (${activeModel})`;
      icon = 'fa-solid fa-brain text-emerald-600';
      badgeColor = 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100';
    } else if (activeProvider === 'ollama') {
      label = `LLM: Ollama (${activeModel})`;
      icon = 'fa-solid fa-server text-purple-600';
      badgeColor = 'bg-purple-50 border-purple-200 text-purple-800 hover:bg-purple-100';
    } else if (activeProvider === 'local_openai') {
      label = `LLM: Local LM Studio (${activeModel})`;
      icon = 'fa-solid fa-desktop text-amber-600';
      badgeColor = 'bg-amber-50 border-amber-200 text-amber-800 hover:bg-amber-100';
    } else {
      label = 'LLM: Offline Safe (Rule-Based)';
      icon = 'fa-solid fa-shield-halved text-slate-600';
      badgeColor = 'bg-slate-100 border-slate-300 text-slate-800 hover:bg-slate-200';
    }

    this.mountContainer.innerHTML = `
      <button onclick="window.llmGatewayUI.openModal()" 
              class="flex items-center space-x-2 px-3 py-1.5 rounded-full border text-xs font-semibold shadow-2xs transition cursor-pointer ${badgeColor}">
        <i class="${icon}"></i>
        <span class="max-w-[170px] truncate">${label}</span>
        <i class="fa-solid fa-sliders text-[10px] opacity-70"></i>
      </button>
    `;
  }

  injectModal() {
    if (document.getElementById('llmGatewayModal')) return;

    const modalHTML = `
      <div id="llmGatewayModal" class="hidden fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
        <div class="bg-white rounded-3xl max-w-3xl w-full border border-slate-200 shadow-2xl flex flex-col max-h-[90vh] overflow-hidden animate-in fade-in zoom-in-95 duration-200">
          
          <!-- Modal Header -->
          <div class="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/80">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 rounded-2xl bg-gradient-to-tr from-indigo-600 to-blue-500 text-white flex items-center justify-center text-lg shadow-sm">
                <i class="fa-solid fa-microchip"></i>
              </div>
              <div>
                <h3 class="text-base font-extrabold text-slate-900 flex items-center space-x-2">
                  <span>Central LLM Gateway & Orchestrator</span>
                  <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800 border border-indigo-200">SOLID LSP</span>
                </h3>
                <p class="text-xs text-slate-500">Hot-swap between Cloud APIs, Local Ollama/LM Studio models, and Offline fallback gracefully</p>
              </div>
            </div>
            <button onclick="window.llmGatewayUI.closeModal()" class="w-8 h-8 rounded-full bg-slate-200 hover:bg-slate-300 text-slate-600 flex items-center justify-center transition">
              <i class="fa-solid fa-xmark text-sm"></i>
            </button>
          </div>

          <!-- Modal Body (Scrollable) -->
          <div class="p-6 overflow-y-auto space-y-6 flex-1 text-slate-800 text-xs">
            
            <!-- Provider Selection Grid -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <span class="font-bold text-slate-700 uppercase tracking-wider text-[11px]">Select Active Provider</span>
                <button onclick="window.llmGatewayUI.refreshHealth()" class="text-blue-600 hover:text-blue-800 font-semibold flex items-center space-x-1">
                  <i class="fa-solid fa-arrows-rotate text-[10px]"></i>
                  <span>Refresh Status</span>
                </button>
              </div>

              <div id="llmProviderCardsGrid" class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <!-- Dynamically rendered provider cards -->
              </div>
            </div>

            <!-- Provider Configuration Box -->
            <div id="llmConfigBox" class="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-4">
              <div class="font-bold text-slate-900 text-sm flex items-center justify-between">
                <span id="configBoxTitle">Configure Provider</span>
                <span id="configBoxStatusBadge" class="text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-slate-200 text-slate-700">Status</span>
              </div>

              <!-- Model Selector -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label class="block font-semibold text-slate-600 mb-1">Select Model</label>
                  <select id="configModelSelect" class="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-mono focus:outline-hidden focus:border-indigo-500">
                  </select>
                </div>
                <div>
                  <label class="block font-semibold text-slate-600 mb-1">Temperature (Creativity)</label>
                  <input type="range" id="configTempRange" min="0" max="1.2" step="0.1" value="0.7" oninput="document.getElementById('tempDisplay').innerText = this.value" class="w-full">
                  <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Deterministic (0.0)</span>
                    <span id="tempDisplay" class="font-bold text-indigo-600">0.7</span>
                    <span>Creative (1.2)</span>
                  </div>
                </div>
              </div>

              <!-- API Key Field (for Gemini / OpenAI) -->
              <div id="apiKeyFieldContainer" class="hidden">
                <label id="apiKeyLabel" class="block font-semibold text-slate-600 mb-1">API Key</label>
                <div class="relative">
                  <input type="password" id="configApiKeyInput" placeholder="Enter API key or leave blank to use existing env var..." class="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-mono pr-10 focus:outline-hidden focus:border-indigo-500">
                  <button type="button" onclick="window.llmGatewayUI.togglePasswordVisibility('configApiKeyInput')" class="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600">
                    <i class="fa-regular fa-eye text-xs"></i>
                  </button>
                </div>
                <p class="text-[10px] text-slate-500 mt-1">Key is cached in-memory and never logged or written to git files.</p>
              </div>

              <!-- Base URL Field (for Ollama / Local OpenAI) -->
              <div id="baseUrlFieldContainer" class="hidden">
                <label id="baseUrlLabel" class="block font-semibold text-slate-600 mb-1">Base Endpoint URL</label>
                <input type="text" id="configBaseUrlInput" placeholder="http://127.0.0.1:11434" class="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-mono focus:outline-hidden focus:border-indigo-500">
                <p class="text-[10px] text-slate-500 mt-1">Make sure the local daemon or LM Studio server is running on this port.</p>
              </div>

              <!-- Fallback Toggle -->
              <div class="flex items-center justify-between pt-2 border-t border-slate-200">
                <div class="flex items-center space-x-2">
                  <i class="fa-solid fa-shield-cat text-emerald-600 text-sm"></i>
                  <div>
                    <div class="font-bold text-slate-800">Auto-Fallback Protection</div>
                    <div class="text-[10px] text-slate-500">Gracefully cascades to offline deterministic engine if cloud API or local server disconnects</div>
                  </div>
                </div>
                <input type="checkbox" id="configFallbackToggle" checked class="w-4 h-4 text-indigo-600 rounded">
              </div>

              <!-- Save / Activate Button -->
              <div class="flex justify-end pt-2">
                <button onclick="window.llmGatewayUI.applyConfiguration()" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold shadow-sm transition flex items-center space-x-2">
                  <i class="fa-solid fa-check"></i>
                  <span>Activate & Save Provider</span>
                </button>
              </div>
            </div>

            <!-- Live Testbench Playground -->
            <div class="p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100 space-y-3">
              <div class="flex items-center justify-between">
                <span class="font-bold text-indigo-900 flex items-center space-x-2">
                  <i class="fa-solid fa-vial"></i>
                  <span>LLM Gateway Live Testbench</span>
                </span>
                <span id="testbenchTiming" class="text-[10px] font-mono text-indigo-600">Latency: -- ms</span>
              </div>
              <div class="flex space-x-2">
                <input type="text" id="testPromptInput" value="Explain how consistent hashing avoids re-sharding in distributed storage in 2 sentences." class="flex-1 bg-white border border-indigo-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-hidden focus:border-indigo-500">
                <button onclick="window.llmGatewayUI.runTestPrompt()" id="btnRunTestPrompt" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold transition flex items-center space-x-1.5 shrink-0">
                  <i class="fa-solid fa-paper-plane text-xs"></i>
                  <span>Test</span>
                </button>
              </div>
              <div id="testOutputArea" class="p-3 bg-white border border-indigo-100 rounded-xl font-mono text-xs text-slate-700 min-h-[50px] whitespace-pre-wrap">
                Click "Test" to verify the active LLM response and inspect latency.
              </div>
            </div>

          </div>

          <!-- Modal Footer -->
          <div class="px-6 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-slate-500 text-xs">
            <div class="flex items-center space-x-1">
              <i class="fa-solid fa-circle-check text-emerald-500"></i>
              <span>Zero-downtime hot-swapping enabled</span>
            </div>
            <button onclick="window.llmGatewayUI.closeModal()" class="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-700 rounded-xl font-bold transition">
              Done
            </button>
          </div>

        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);
  }

  openModal() {
    this.fetchStatus().then(() => {
      this.populateModal();
      document.getElementById('llmGatewayModal').classList.remove('hidden');
    });
  }

  closeModal() {
    document.getElementById('llmGatewayModal').classList.add('hidden');
    this.renderPill();
  }

  populateModal() {
    if (!this.status) return;

    const grid = document.getElementById('llmProviderCardsGrid');
    const activePid = this.status.active_provider;
    
    const providerMeta = {
      gemini: { icon: 'fa-brands fa-google', color: 'text-blue-600', desc: 'Fast, high-context Google cloud model family' },
      openai: { icon: 'fa-solid fa-brain', color: 'text-emerald-600', desc: 'Standard GPT-4o / GPT-4o-mini reasoning API' },
      ollama: { icon: 'fa-solid fa-server', color: 'text-purple-600', desc: 'Runs locally on your GPU/CPU (http://127.0.0.1:11434)' },
      local_openai: { icon: 'fa-solid fa-desktop', color: 'text-amber-600', desc: 'LM Studio / vLLM local compatible endpoints' },
      rule_based: { icon: 'fa-solid fa-shield-halved', color: 'text-slate-600', desc: '100% offline deterministic fallback engine' }
    };

    grid.innerHTML = (this.status.providers || []).map(p => {
      const isSelected = p.provider_id === activePid;
      const meta = providerMeta[p.provider_id] || { icon: 'fa-solid fa-robot', color: 'text-slate-600', desc: '' };
      const statusBadge = p.is_available 
        ? `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">Online</span>`
        : `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-amber-100 text-amber-800 border border-amber-200">Offline / Key needed</span>`;

      return `
        <div onclick="window.llmGatewayUI.selectProvider('${p.provider_id}')" 
             class="p-3.5 rounded-2xl border transition cursor-pointer flex flex-col justify-between ${isSelected ? 'border-indigo-600 bg-indigo-50/70 ring-2 ring-indigo-500/20' : 'border-slate-200 bg-white hover:border-slate-300'}">
          <div class="flex items-start justify-between">
            <div class="flex items-center space-x-2.5">
              <div class="w-8 h-8 rounded-xl bg-white border border-slate-200 flex items-center justify-center text-sm shadow-2xs ${meta.color}">
                <i class="${meta.icon}"></i>
              </div>
              <div>
                <div class="font-extrabold text-slate-900">${p.provider_name}</div>
                <div class="text-[10px] text-slate-500 line-clamp-1">${meta.desc}</div>
              </div>
            </div>
            ${statusBadge}
          </div>
          <div class="mt-2.5 flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span>${(p.available_models || []).length} models</span>
            ${isSelected ? '<span class="text-indigo-600 font-bold flex items-center space-x-1"><i class="fa-solid fa-circle-check"></i><span>Active</span></span>' : '<span class="text-slate-400">Click to configure</span>'}
          </div>
        </div>
      `;
    }).join('');

    this.selectProvider(activePid);
  }

  selectProvider(providerId) {
    this.selectedProviderId = providerId;
    const provider = (this.status.providers || []).find(p => p.provider_id === providerId);
    if (!provider) return;

    // Update config box title & badge
    document.getElementById('configBoxTitle').innerText = `Configure ${provider.provider_name}`;
    const badge = document.getElementById('configBoxStatusBadge');
    if (provider.is_available) {
      badge.className = 'text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800';
      badge.innerText = 'Connected & Operational';
    } else {
      badge.className = 'text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800';
      badge.innerText = provider.status_message || 'Setup required';
    }

    // Populate model dropdown
    const modelSelect = document.getElementById('configModelSelect');
    const models = provider.available_models || [];
    modelSelect.innerHTML = models.map(m => `<option value="${m}" ${m === this.status.active_model ? 'selected' : ''}>${m}</option>`).join('');

    // Toggle fields based on provider type
    const keyContainer = document.getElementById('apiKeyFieldContainer');
    const urlContainer = document.getElementById('baseUrlFieldContainer');

    if (providerId === 'gemini') {
      keyContainer.classList.remove('hidden');
      document.getElementById('apiKeyLabel').innerText = 'Google Gemini API Key (GEMINI_API_KEY)';
      urlContainer.classList.add('hidden');
    } else if (providerId === 'openai') {
      keyContainer.classList.remove('hidden');
      document.getElementById('apiKeyLabel').innerText = 'OpenAI API Key (OPENAI_API_KEY)';
      urlContainer.classList.add('hidden');
    } else if (providerId === 'ollama') {
      keyContainer.classList.add('hidden');
      urlContainer.classList.remove('hidden');
      document.getElementById('baseUrlLabel').innerText = 'Ollama Host URL (IPv4)';
      document.getElementById('configBaseUrlInput').value = provider.endpoint_url || 'http://127.0.0.1:11434';
    } else if (providerId === 'local_openai') {
      keyContainer.classList.add('hidden');
      urlContainer.classList.remove('hidden');
      document.getElementById('baseUrlLabel').innerText = 'Local Server URL (LM Studio / vLLM)';
      document.getElementById('configBaseUrlInput').value = provider.endpoint_url || 'http://localhost:1234/v1';
    } else {
      keyContainer.classList.add('hidden');
      urlContainer.classList.add('hidden');
    }

    document.getElementById('configFallbackToggle').checked = this.status.fallback_enabled !== false;
    document.getElementById('configTempRange').value = this.status.temperature || 0.7;
    document.getElementById('tempDisplay').innerText = this.status.temperature || 0.7;
  }

  async applyConfiguration() {
    const providerId = this.selectedProviderId;
    const model = document.getElementById('configModelSelect').value;
    const apiKey = document.getElementById('configApiKeyInput').value.trim() || undefined;
    const baseUrl = document.getElementById('configBaseUrlInput').value.trim() || undefined;
    const temperature = parseFloat(document.getElementById('configTempRange').value);
    const fallbackEnabled = document.getElementById('configFallbackToggle').checked;

    try {
      const resp = await fetch('/api/llm/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_id: providerId,
          model: model,
          api_key: apiKey,
          base_url: baseUrl,
          temperature: temperature,
          fallback_enabled: fallbackEnabled
        })
      });

      this.status = await resp.json();
      this.populateModal();
      this.renderPill();
      alert(`Provider "${providerId}" activated with model "${model}".`);
    } catch (e) {
      alert('Failed to configure provider: ' + e.message);
    }
  }

  async refreshHealth() {
    await this.fetchStatus(true);
    this.populateModal();
    this.renderPill();
  }

  async runTestPrompt() {
    const prompt = document.getElementById('testPromptInput').value.trim();
    if (!prompt) return;

    const outArea = document.getElementById('testOutputArea');
    const timing = document.getElementById('testbenchTiming');
    const btn = document.getElementById('btnRunTestPrompt');

    btn.disabled = true;
    outArea.innerText = 'Generating response through active LLM provider...';
    timing.innerText = 'Testing...';

    const t0 = performance.now();
    try {
      const resp = await fetch('/api/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
      });
      const data = await resp.json();
      const latencyMs = Math.round(performance.now() - t0);

      timing.innerHTML = `<span class="text-emerald-700 font-bold">${latencyMs} ms</span> | Provider: <b>${data.provider}</b> | Model: <b>${data.model}</b>`;
      
      let badge = '';
      if (data.fallback_used) {
        badge = `<div class="mb-2 p-2 rounded bg-amber-50 border border-amber-200 text-amber-800 text-[10px]">
          <b>[Fallback Notice]</b> ${data.fallback_reason || 'Switched to fallback engine'}
        </div>`;
      }

      outArea.innerHTML = badge + `<div class="text-slate-800">${data.content}</div>`;
    } catch (e) {
      outArea.innerHTML = `<span class="text-rose-600 font-bold">[Error] ${e.message}</span>`;
      timing.innerText = 'Failed';
    } finally {
      btn.disabled = false;
    }
  }

  togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    input.type = input.type === 'password' ? 'text' : 'password';
  }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.llmGatewayUI = new LLMGatewayUI();
});
