/**
 * Atomic Capability Plugins Registry & Unified Event Bus.
 * Implements the "Plugs" layer of the 3-Layer Plug-and-Play Architecture.
 * Standardizes WorkspacePlugin interface, event emissions, and artifact exports.
 */

class EventBus {
  constructor() {
    this.listeners = {};
    this.eventHistory = [];
  }

  subscribe(eventType, callback) {
    if (!this.listeners[eventType]) {
      this.listeners[eventType] = [];
    }
    this.listeners[eventType].push(callback);
    return () => {
      this.listeners[eventType] = this.listeners[eventType].filter(cb => cb !== callback);
    };
  }

  emit(pluginId, eventType, payload = {}) {
    const event = {
      pluginId,
      eventType,
      payload,
      timestamp: Date.now()
    };
    this.eventHistory.push(event);

    // Call wildcard and specific listeners
    const targets = (this.listeners[eventType] || []).concat(this.listeners['*'] || []);
    targets.forEach(cb => {
      try {
        cb(event);
      } catch (err) {
        console.error(`[EventBus] Error in listener for ${eventType}:`, err);
      }
    });

    // Also forward to server telemetry stream if desired
    fetch('/api/blueprints/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event)
    }).catch(() => {});
  }

  getEventHistory() {
    return this.eventHistory;
  }
}

// Global Event Bus Instance
window.workspaceEventBus = new EventBus();


// ============================================================================
// Base Workspace Plugin Contract
// ============================================================================

class WorkspacePlugin {
  constructor(id, title, icon = 'fa-cube') {
    this.id = id;
    this.title = title;
    this.icon = icon;
    this.container = null;
    this.config = {};
    this.bus = null;
  }

  mount(container, config = {}, bus = window.workspaceEventBus) {
    this.container = container;
    this.config = config;
    this.bus = bus;
  }

  getStateSnapshot() {
    return {};
  }

  async exportArtifact() {
    return {
      pluginId: this.id,
      snapshot: this.getStateSnapshot(),
      exportedAt: Date.now()
    };
  }

  unmount() {
    if (this.container) {
      this.container.innerHTML = '';
      this.container = null;
    }
  }
}


// ============================================================================
// 1. Monaco Code Editor Plugin (Core SDE, Fullstack)
// ============================================================================

class CodeEditorPlugin extends WorkspacePlugin {
  constructor() {
    super('CodeEditor', 'Monaco Code Editor', 'fa-code');
    this.editorInstance = null;
    this.language = 'python';
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    const initialCode = config.starter_code || '# Write your solution here\nclass Solution:\n    def solve(self):\n        pass\n';
    this.language = config.language || 'python';

    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-slate-800">
            <i class="fa-solid fa-code text-blue-600"></i>
            <span>${this.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 font-mono">Python / JS / TS</span>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="window.activePlugins['CodeEditor'].runCode()" class="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold transition flex items-center space-x-1.5 shadow-2xs">
              <i class="fa-solid fa-play text-[10px]"></i>
              <span>Execute Sandbox</span>
            </button>
          </div>
        </div>
        <div id="monacoMount_${this.id}" class="flex-1 w-full min-h-0 bg-white"></div>
        <div id="codeEditorOutput_${this.id}" class="h-28 bg-slate-900 text-slate-200 p-3 font-mono text-xs overflow-y-auto border-t border-slate-800 shrink-0">
          <div class="text-slate-500 italic">Output terminal ready. Click "Execute Sandbox" to run.</div>
        </div>
      </div>
    `;

    setTimeout(() => {
      const mountEl = document.getElementById(`monacoMount_${this.id}`);
      if (mountEl && window.monaco) {
        this.editorInstance = monaco.editor.create(mountEl, {
          value: initialCode,
          language: this.language,
          theme: 'vs',
          automaticLayout: true,
          fontSize: 13,
          minimap: { enabled: false },
          lineNumbers: 'on'
        });

        this.editorInstance.onDidChangeModelContent(() => {
          this.bus.emit(this.id, 'code_diff', {
            length: this.editorInstance.getValue().length,
            language: this.language
          });
        });
      }
    }, 150);
  }

  async runCode() {
    const code = this.editorInstance ? this.editorInstance.getValue() : '';
    const outEl = document.getElementById(`codeEditorOutput_${this.id}`);
    outEl.innerHTML = '<span class="text-blue-400">[Executing] Running in Python 3.10 sandbox...</span>';

    try {
      const resp = await fetch('/api/interviewer/run-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code, language: this.language })
      });
      const data = await resp.json();
      outEl.innerHTML = `<span class="text-emerald-400">[Status 0 Exit]:</span>\n<div class="text-slate-100 whitespace-pre">${data.output}</div>`;
      this.bus.emit(this.id, 'code_run', { status: 'success', output: data.output });
    } catch (err) {
      outEl.innerHTML = `<span class="text-rose-400">[Execution Error]:</span> ${err.message}`;
      this.bus.emit(this.id, 'code_run', { status: 'error', error: err.message });
    }
  }

  getStateSnapshot() {
    return {
      code: this.editorInstance ? this.editorInstance.getValue() : '',
      language: this.language
    };
  }
}


// ============================================================================
// 2. Interactive Notebook Kernel Plugin (ML / Data Science)
// ============================================================================

class NotebookKernelPlugin extends WorkspacePlugin {
  constructor() {
    super('NotebookKernel', 'Interactive Jupyter Notebook', 'fa-book-bookmark');
    this.cells = [
      {
        id: 'c1',
        type: 'code',
        content: '# Cell 1: Exploratory Data Analysis & Class Balance\nimport numpy as np\n\ny_train = np.array([1, 0, 1, 1, 0, 0, 1, 0])\npos_rate = np.mean(y_train)\nprint(f"[Dataset] Cohort size: {len(y_train)} | Positive Rate: {pos_rate:.2%}")',
        output: '[Dataset] Cohort size: 8 | Positive Rate: 50.00%'
      },
      {
        id: 'c2',
        type: 'code',
        content: '# Cell 2: Training Loss Convergence & AUC Calibration\nloss_history = [0.82, 0.65, 0.44, 0.31, 0.22]\nprint(f"[Epoch 5] Final Loss: {loss_history[-1]} | ROC AUC: 0.942")',
        output: '[Epoch 5] Final Loss: 0.22 | ROC AUC: 0.942'
      }
    ];
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-purple-900">
            <i class="fa-solid fa-brain text-purple-600"></i>
            <span>${this.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 font-mono">Python 3.10 · PyTorch/NumPy</span>
          </div>
          <button onclick="window.activePlugins['NotebookKernel'].addCell()" class="px-2.5 py-1 bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200 rounded-lg text-xs font-bold transition flex items-center space-x-1">
            <i class="fa-solid fa-plus text-[10px]"></i>
            <span>Add Cell</span>
          </button>
        </div>
        <div class="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-50/50">
          ${this.cells.map((cell, idx) => `
            <div class="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs space-y-2">
              <div class="flex items-center justify-between text-[11px] font-mono text-slate-500">
                <span class="font-bold text-purple-700">In [${idx + 1}]:</span>
                <button onclick="window.activePlugins['NotebookKernel'].runCell(${idx})" class="px-2.5 py-0.5 bg-purple-600 hover:bg-purple-700 text-white rounded text-[10px] font-bold transition flex items-center space-x-1">
                  <i class="fa-solid fa-play text-[8px]"></i>
                  <span>Run</span>
                </button>
              </div>
              <textarea id="notebook_cell_${idx}" rows="4" class="w-full font-mono text-xs p-2.5 bg-slate-900 text-purple-200 rounded-lg focus:outline-hidden leading-relaxed">${cell.content}</textarea>
              ${cell.output ? `
                <div class="pt-2 border-t border-slate-100 text-[11px] font-mono text-slate-700">
                  <span class="text-slate-400 font-bold">Out [${idx + 1}]:</span>
                  <div class="bg-slate-50 border border-slate-200 p-2 rounded mt-1 text-slate-800">${cell.output}</div>
                </div>
              ` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  addCell() {
    this.cells.push({
      id: 'c' + (this.cells.length + 1),
      type: 'code',
      content: '# New computation cell\nimport numpy as np\n',
      output: ''
    });
    this.render();
  }

  async runCell(idx) {
    const textarea = document.getElementById(`notebook_cell_${idx}`);
    if (textarea) this.cells[idx].content = textarea.value;

    const code = this.cells[idx].content;
    try {
      const resp = await fetch('/api/interviewer/run-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code, language: 'python' })
      });
      const data = await resp.json();
      this.cells[idx].output = data.output;
      this.render();
      this.bus.emit(this.id, 'cell_run', { cell_index: idx + 1, status: 'completed' });
    } catch (e) {
      this.cells[idx].output = 'Execution failed: ' + e.message;
      this.render();
    }
  }

  getStateSnapshot() {
    return { cells: this.cells };
  }
}


// ============================================================================
// 3. Linux Terminal Plugin (Cybersecurity, Fullstack, SDE)
// ============================================================================

class LinuxTerminalPlugin extends WorkspacePlugin {
  constructor() {
    super('LinuxTerminal', 'Linux Bash Console', 'fa-terminal');
    this.history = [
      'Welcome to SecOps Sandbox v2.4 (GNU/Linux 6.1.0-x86_64)',
      'Type "help", "nmap -sV", "ps aux", "iptables -L", or "cat /var/log/auth.log" to inspect.'
    ];
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-slate-950 text-slate-200 border border-slate-800 rounded-2xl overflow-hidden shadow-xs font-mono">
        <div class="h-9 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between text-xs text-slate-400 shrink-0">
          <div class="flex items-center space-x-2">
            <span class="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>
            <span class="text-slate-300 font-bold ml-2">candidate@sandbox:~</span>
          </div>
          <span class="text-[10px] text-slate-500">Tier 2 Namespace Active</span>
        </div>
        <div id="terminalBody_${this.id}" class="flex-1 p-4 overflow-y-auto space-y-1.5 text-xs text-emerald-400 whitespace-pre-wrap leading-relaxed">
          ${this.history.map(line => `<div>${line}</div>`).join('')}
        </div>
        <div class="p-2.5 bg-slate-900/90 border-t border-slate-800 flex items-center space-x-2 text-xs">
          <span class="text-emerald-500 font-bold">candidate@sandbox:~$</span>
          <input type="text" id="terminalInput_${this.id}" onkeydown="if(event.key==='Enter') window.activePlugins['LinuxTerminal'].executeCommand(this.value)" placeholder="Enter command..." class="flex-1 bg-transparent text-slate-100 focus:outline-hidden font-mono">
        </div>
      </div>
    `;
  }

  executeCommand(cmd) {
    const input = document.getElementById(`terminalInput_${this.id}`);
    if (input) input.value = '';
    const clean = cmd.trim();
    if (!clean) return;

    this.history.push(`candidate@sandbox:~$ ${clean}`);

    // Simulated responsive linux execution
    let output = '';
    const lower = clean.lower ? clean.lower() : clean.toLowerCase();
    if (lower === 'clear') {
      this.history = [];
    } else if (lower.startsWith('nmap')) {
      output = "Starting Nmap 7.93 ( https://nmap.org )\nNmap scan report for host (10.0.12.84)\nPORT     STATE SERVICE VERSION\n22/tcp   open  ssh     OpenSSH 8.9p1\n80/tcp   open  http    nginx 1.18.0\n5432/tcp open  pgsql   PostgreSQL 15.2";
    } else if (lower.startsWith('cat /var/log/auth.log')) {
      output = "Nov 23 09:12:04 host sshd[1092]: Failed password for invalid user admin from 198.51.100.42 port 50221\nNov 23 09:12:08 host sshd[1098]: Accepted publickey for candidate from 10.0.1.5 port 50228";
    } else if (lower.startsWith('iptables')) {
      output = "Chain INPUT (policy ACCEPT)\ntarget     prot opt source               destination         \nACCEPT     tcp  --  anywhere             anywhere             tcp dpt:https\nDROP       tcp  --  198.51.100.42        anywhere";
    } else if (lower === 'ps aux') {
      output = "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nroot         1  0.0  0.1   1284   400 ?        Ss   09:00   0:00 /init\ncandidate   42  0.2  0.8 104200 16400 pts/0    S+   09:10   0:01 node server.js";
    } else {
      output = `[Executed in sandbox]: ${clean}`;
    }

    if (output) this.history.push(output);

    const body = document.getElementById(`terminalBody_${this.id}`);
    if (body) {
      body.innerHTML = this.history.map(line => `<div>${line}</div>`).join('');
      body.scrollTop = body.scrollHeight;
    }

    this.bus.emit(this.id, 'terminal_exec', { command: clean, output: output });
  }

  getStateSnapshot() {
    return { command_history: this.history };
  }
}


// ============================================================================
// 4. Live Web Preview Plugin (Fullstack Web Development)
// ============================================================================

class LiveWebPreviewPlugin extends WorkspacePlugin {
  constructor() {
    super('LiveWebPreview', 'Live Web App Preview', 'fa-globe');
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between text-xs shrink-0">
          <div class="flex items-center space-x-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            <span class="font-bold text-slate-800">Live Preview</span>
          </div>
          <div class="flex items-center space-x-2 flex-1 max-w-sm mx-4 bg-white border border-slate-300 rounded-lg px-2.5 py-1 text-[11px] font-mono text-slate-600">
            <i class="fa-solid fa-lock text-[10px] text-emerald-600"></i>
            <span>http://localhost:3000/dashboard</span>
          </div>
          <div class="flex items-center space-x-1.5 text-slate-400">
            <button onclick="window.activePlugins['LiveWebPreview'].refresh()" title="Reload Preview" class="p-1 hover:text-slate-600"><i class="fa-solid fa-rotate-right"></i></button>
            <i class="fa-solid fa-mobile-screen hover:text-slate-600 cursor-pointer p-1"></i>
          </div>
        </div>
        <div id="previewFrameContainer" class="flex-1 p-6 bg-slate-100 flex items-center justify-center overflow-auto">
          <div class="w-full max-w-md bg-white rounded-2xl border border-slate-200 shadow-md p-6 space-y-4">
            <div class="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h4 class="font-bold text-slate-900 text-sm">Cluster Metrics Feed</h4>
                <p class="text-[11px] text-slate-400">Hot-reloaded from local port 3000</p>
              </div>
              <span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-bold">200 OK</span>
            </div>
            <div class="grid grid-cols-2 gap-3">
              <div class="p-3 bg-blue-50 rounded-xl border border-blue-100">
                <span class="text-[10px] font-bold text-blue-800 uppercase">Requests / sec</span>
                <div class="text-xl font-black text-blue-950 mt-0.5">24,510</div>
                <span class="text-[10px] text-emerald-600 font-bold">+12.4%</span>
              </div>
              <div class="p-3 bg-indigo-50 rounded-xl border border-indigo-100">
                <span class="text-[10px] font-bold text-indigo-800 uppercase">p99 Latency</span>
                <div class="text-xl font-black text-indigo-950 mt-0.5">8.4 ms</div>
                <span class="text-[10px] text-indigo-600 font-bold">Target &lt; 15ms</span>
              </div>
            </div>
            <button onclick="alert('Action dispatched in preview');" class="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-bold transition">
              Trigger Real-Time Webhook
            </button>
          </div>
        </div>
      </div>
    `;
  }

  refresh() {
    this.bus.emit(this.id, 'dom_interaction', { action: 'refresh_preview' });
  }

  getStateSnapshot() {
    return { url: 'http://localhost:3000/dashboard', status: 'rendered' };
  }
}


// ============================================================================
// 5. Interactive Spreadsheet Grid Plugin (MBA / Consulting Strategy)
// ============================================================================

class SpreadsheetGridPlugin extends WorkspacePlugin {
  constructor() {
    super('SpreadsheetGrid', 'Strategic Financial Spreadsheet', 'fa-table-cells');
    this.grid = [
      ['Metric', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
      ['EV Fleet Units', '3000', '6000', '9000', '12000', '15000'],
      ['Fuel Savings ($M)', '12.4', '25.8', '40.2', '55.6', '71.0'],
      ['Maintenance Savings ($M)', '4.2', '8.9', '14.1', '19.8', '25.5'],
      ['Total Cost Benefit ($M)', '=SUM(B3:B4)', '=SUM(C3:C4)', '=SUM(D3:D4)', '=SUM(E3:E4)', '=SUM(F3:F4)'],
      ['EBITDA Margin', '14.5%', '16.2%', '18.8%', '20.5%', '22.1%']
    ];
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-amber-900">
            <i class="fa-solid fa-table-cells text-amber-600"></i>
            <span>${this.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-mono">Dynamic Formulas Active</span>
          </div>
          <span class="text-[11px] font-mono text-slate-500">Formulas: SUM, EBITDA, %, AVG</span>
        </div>
        <div class="flex-1 p-4 overflow-auto">
          <table class="w-full border-collapse border border-slate-300 text-xs">
            <tbody>
              ${this.grid.map((row, rIdx) => `
                <tr class="${rIdx === 0 ? 'bg-slate-100 font-bold text-slate-800' : 'hover:bg-slate-50'}">
                  ${row.map((cell, cIdx) => `
                    <td class="border border-slate-300 p-2 min-w-[100px] text-center">
                      ${rIdx === 0 || cIdx === 0 ? `
                        <span class="font-semibold text-slate-900">${cell}</span>
                      ` : `
                        <input type="text" value="${cell}" 
                               onchange="window.activePlugins['SpreadsheetGrid'].updateCell(${rIdx}, ${cIdx}, this.value)"
                               class="w-full text-center bg-transparent focus:bg-amber-50 focus:outline-hidden font-mono text-slate-800">
                      `}
                    </td>
                  `).join('')}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  updateCell(r, c, val) {
    this.grid[r][c] = val;
    this.bus.emit(this.id, 'formula_update', {
      row: r,
      col: c,
      value: val,
      cell: `${String.fromCharCode(65 + c)}${r + 1}`
    });
  }

  getStateSnapshot() {
    return { grid_data: this.grid };
  }
}


// ============================================================================
// 6. Infinite Whiteboard Architecture Canvas Plugin (System Design / PM)
// ============================================================================

class InfiniteWhiteboardPlugin extends WorkspacePlugin {
  constructor() {
    super('InfiniteWhiteboard', 'System Design Whiteboard', 'fa-object-group');
    this.nodes = [
      { id: 'n1', label: 'Route53 DNS / Multi-CDN', x: 40, y: 30, color: 'bg-blue-100 border-blue-400 text-blue-900' },
      { id: 'n2', label: 'Envoy API Gateway (Auth & Rate Limit)', x: 260, y: 30, color: 'bg-emerald-100 border-emerald-400 text-emerald-900' },
      { id: 'n3', label: 'Kafka Partitioned Commit Log', x: 120, y: 150, color: 'bg-purple-100 border-purple-400 text-purple-900' },
      { id: 'n4', label: 'Redis Cluster (Consistent Hash Ring)', x: 380, y: 150, color: 'bg-rose-100 border-rose-400 text-rose-900' },
      { id: 'n5', label: 'Cassandra / ScyllaDB Shards', x: 250, y: 260, color: 'bg-amber-100 border-amber-400 text-amber-900' }
    ];
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-cyan-900">
            <i class="fa-solid fa-network-wired text-cyan-600"></i>
            <span>${this.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-cyan-100 text-cyan-800 font-mono">Live Architecture Graph</span>
          </div>
          <div class="flex items-center space-x-1.5">
            <button onclick="window.activePlugins['InfiniteWhiteboard'].addNode('Microservice')" class="px-2 py-1 bg-white hover:bg-slate-100 border border-slate-300 rounded text-[11px] font-bold text-slate-700 transition">+ Service</button>
            <button onclick="window.activePlugins['InfiniteWhiteboard'].addNode('Cache')" class="px-2 py-1 bg-white hover:bg-slate-100 border border-slate-300 rounded text-[11px] font-bold text-slate-700 transition">+ Cache</button>
            <button onclick="window.activePlugins['InfiniteWhiteboard'].addNode('Queue')" class="px-2 py-1 bg-white hover:bg-slate-100 border border-slate-300 rounded text-[11px] font-bold text-slate-700 transition">+ Queue</button>
          </div>
        </div>
        <div id="whiteboardCanvas" class="flex-1 relative bg-[radial-gradient(#cbd5e1_1px,transparent_1px)] [background-size:16px_16px] p-6 overflow-auto">
          <!-- Flow arrows representation -->
          <svg class="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
              </marker>
            </defs>
            <line x1="160" y1="50" x2="260" y2="50" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="320" y1="80" x2="200" y2="150" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="360" y1="80" x2="420" y2="150" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="220" y1="200" x2="280" y2="260" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
          </svg>

          <!-- Rendered Architecture Nodes -->
          ${this.nodes.map(n => `
            <div style="position: absolute; left: ${n.x}px; top: ${n.y}px;" 
                 class="px-3.5 py-2.5 rounded-xl border ${n.color} shadow-sm text-xs font-bold cursor-move flex items-center space-x-2 select-none hover:shadow-md transition">
              <i class="fa-solid fa-server opacity-70"></i>
              <span>${n.label}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  addNode(type) {
    const labels = {
      'Microservice': 'Auth / Presence Service',
      'Cache': 'Distributed L1 Memcached',
      'Queue': 'RabbitMQ Dead-Letter Queue'
    };
    const node = {
      id: 'n' + (this.nodes.length + 1),
      label: labels[type] || type,
      x: 100 + Math.random() * 250,
      y: 100 + Math.random() * 200,
      color: 'bg-emerald-50 border-emerald-300 text-emerald-900'
    };
    this.nodes.push(node);
    this.render();
    this.bus.emit(this.id, 'whiteboard_node_add', { label: node.label, id: node.id });
  }

  getStateSnapshot() {
    return { nodes: this.nodes };
  }
}


// ============================================================================
// 7. SQL Workbench & Schema Browser Plugin (Fullstack, ML, SDE)
// ============================================================================

class SQLWorkbenchPlugin extends WorkspacePlugin {
  constructor() {
    super('SQLWorkbench', 'SQL Workbench & Query Engine', 'fa-database');
    this.query = "SELECT user_id, COUNT(event_id) AS total_events, AVG(latency_ms) AS avg_lat\nFROM click_stream_logs\nWHERE timestamp >= NOW() - INTERVAL '24 hours'\nGROUP BY user_id\nHAVING COUNT(event_id) > 10\nORDER BY total_events DESC\nLIMIT 5;";
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-indigo-900">
            <i class="fa-solid fa-database text-indigo-600"></i>
            <span>${this.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800 font-mono">PostgreSQL 15</span>
          </div>
          <button onclick="window.activePlugins['SQLWorkbench'].runQuery()" class="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-bold transition flex items-center space-x-1">
            <i class="fa-solid fa-play text-[9px]"></i>
            <span>Run SQL</span>
          </button>
        </div>
        <div class="h-32 border-b border-slate-200 p-2 bg-slate-900">
          <textarea id="sqlInput_${this.id}" class="w-full h-full bg-transparent text-indigo-200 font-mono text-xs focus:outline-hidden resize-none">${this.query}</textarea>
        </div>
        <div id="sqlResults_${this.id}" class="flex-1 p-3 overflow-auto bg-slate-50 text-xs font-mono text-slate-700">
          <div class="text-[11px] font-bold text-slate-500 mb-1">Query Results (5 rows returned in 4.2ms):</div>
          <table class="w-full border-collapse border border-slate-300 bg-white">
            <thead class="bg-slate-100 font-bold">
              <tr>
                <th class="border border-slate-300 p-1.5 text-left">user_id</th>
                <th class="border border-slate-300 p-1.5 text-right">total_events</th>
                <th class="border border-slate-300 p-1.5 text-right">avg_lat</th>
              </tr>
            </thead>
            <tbody>
              <tr><td class="border border-slate-300 p-1.5 text-blue-600 font-bold">u_8902</td><td class="border border-slate-300 p-1.5 text-right">412</td><td class="border border-slate-300 p-1.5 text-right">6.4 ms</td></tr>
              <tr><td class="border border-slate-300 p-1.5 text-blue-600 font-bold">u_1092</td><td class="border border-slate-300 p-1.5 text-right">389</td><td class="border border-slate-300 p-1.5 text-right">7.1 ms</td></tr>
              <tr><td class="border border-slate-300 p-1.5 text-blue-600 font-bold">u_3321</td><td class="border border-slate-300 p-1.5 text-right">294</td><td class="border border-slate-300 p-1.5 text-right">5.8 ms</td></tr>
              <tr><td class="border border-slate-300 p-1.5 text-blue-600 font-bold">u_5541</td><td class="border border-slate-300 p-1.5 text-right">251</td><td class="border border-slate-300 p-1.5 text-right">8.2 ms</td></tr>
              <tr><td class="border border-slate-300 p-1.5 text-blue-600 font-bold">u_7719</td><td class="border border-slate-300 p-1.5 text-right">190</td><td class="border border-slate-300 p-1.5 text-right">6.0 ms</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  runQuery() {
    const input = document.getElementById(`sqlInput_${this.id}`);
    if (input) this.query = input.value;
    this.bus.emit(this.id, 'sql_query_exec', { query: this.query, rows_returned: 5 });
  }

  getStateSnapshot() {
    return { last_query: this.query };
  }
}


// ============================================================================
// 8. Case Brief & Problem Panel Plugin (Universal across all roles)
// ============================================================================

class CaseBriefReaderPlugin extends WorkspacePlugin {
  constructor() {
    super('CaseBriefReader', 'Problem Statement & Case Exhibits', 'fa-file-lines');
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    const title = config.title || 'Technical Assessment Scenario';
    const desc = config.description || 'Deconstruct the scenario, outline your approach, and formulate your solution.';

    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-slate-800">
            <i class="fa-regular fa-file-lines text-blue-600"></i>
            <span>${this.title}</span>
          </div>
          <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-200 text-slate-700">Official Problem Doc</span>
        </div>
        <div class="flex-1 p-5 overflow-y-auto space-y-4 text-xs leading-relaxed text-slate-700">
          <div>
            <h3 class="font-extrabold text-slate-900 text-sm mb-1">${title}</h3>
            <p class="text-slate-600">${desc}</p>
          </div>
          <div class="p-3 bg-amber-50 rounded-xl border border-amber-200 space-y-1">
            <div class="font-bold text-amber-900 uppercase text-[10px] flex items-center space-x-1">
              <i class="fa-solid fa-bullseye"></i>
              <span>Key Objectives</span>
            </div>
            <ul class="list-disc pl-4 space-y-1 text-amber-800 text-[11px]">
              <li>State assumptions explicitly before writing code or formulas.</li>
              <li>Address both average and worst-case scaling behavior.</li>
              <li>Explain architectural trade-offs to the AI interviewer.</li>
            </ul>
          </div>
        </div>
      </div>
    `;
  }

  getStateSnapshot() {
    return { doc_read: true };
  }
}


// PromptPanel alias pointing to CaseBriefReader
class PromptPanelPlugin extends CaseBriefReaderPlugin {
  constructor() {
    super();
    this.id = 'PromptPanel';
    this.title = 'Problem Prompt & Specifications';
  }
}


// ============================================================================
// 9. Structured Notes & Scratchpad Plugin
// ============================================================================

class StructuredNotesPlugin extends WorkspacePlugin {
  constructor() {
    super('StructuredNotes', 'Candidate Notes & MECE Scratchpad', 'fa-pen-to-square');
    this.notes = "## Problem Notes & Framework\n- Target QPS: 50,000 requests/sec\n- Storage growth: ~2 TB / month\n- Key Partitioning: Consistent hashing with virtual nodes to prevent hot spots\n\n### Open Questions / Clarifications:\n1. Is strong consistency required for metadata, or is eventual consistency acceptable?\n2. What is the allowable failover threshold between availability zones?";
  }

  mount(container, config, bus) {
    super.mount(container, config, bus);
    this.container.innerHTML = `
      <div class="flex flex-col h-full bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
        <div class="h-10 bg-slate-50 border-b border-slate-200 px-4 flex items-center justify-between shrink-0">
          <div class="flex items-center space-x-2 font-bold text-xs text-slate-800">
            <i class="fa-regular fa-pen-to-square text-indigo-600"></i>
            <span>${this.title}</span>
          </div>
          <span class="text-[10px] text-slate-400">Live Synchronized</span>
        </div>
        <div class="flex-1 p-2 bg-slate-50/50">
          <textarea id="notesArea_${this.id}" oninput="window.activePlugins['StructuredNotes'].updateNotes(this.value)" class="w-full h-full p-3 bg-white border border-slate-200 rounded-xl font-mono text-xs text-slate-800 focus:outline-hidden leading-relaxed resize-none">${this.notes}</textarea>
        </div>
      </div>
    `;
  }

  updateNotes(val) {
    this.notes = val;
    this.bus.emit(this.id, 'notes_updated', { length: val.length });
  }

  getStateSnapshot() {
    return { notes: this.notes };
  }
}


// ============================================================================
// Plugin Master Registry & Factory
// ============================================================================

window.PLUGINS_REGISTRY = {
  CodeEditor: CodeEditorPlugin,
  NotebookKernel: NotebookKernelPlugin,
  LinuxTerminal: LinuxTerminalPlugin,
  LiveWebPreview: LiveWebPreviewPlugin,
  SpreadsheetGrid: SpreadsheetGridPlugin,
  InfiniteWhiteboard: InfiniteWhiteboardPlugin,
  SQLWorkbench: SQLWorkbenchPlugin,
  CaseBriefReader: CaseBriefReaderPlugin,
  PromptPanel: PromptPanelPlugin,
  StructuredNotes: StructuredNotesPlugin
};

window.activePlugins = {};
