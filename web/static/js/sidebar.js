/**
 * Modular Navigation Sidebar Component.
 * Automatically highlights the active page and renders links to all 8 separate pages.
 */
function renderSidebar(activeKey) {
  const sidebarContainer = document.getElementById('sidebar-container');
  if (!sidebarContainer) return;

  const currentPath = window.location.pathname;

  const menuItems = [
    { key: 'dashboard', href: '/', icon: 'fa-gauge-high', iconColor: 'text-indigo-600', title: 'Live Dashboard', subtitle: 'Main interactive hub', section: 'Main Platform' },
    { key: 'roles', href: '/roles', icon: 'fa-masks-theater', iconColor: 'text-indigo-600', title: 'Tier 1: Personas', subtitle: 'Choose AI Role Profile', section: 'Agent Intelligence' },
    { key: 'interviewer-mode', href: '/interviewer-mode', icon: 'fa-user-tie', iconColor: 'text-purple-600', title: 'Tier 2: Interviewer Mode', subtitle: 'Solo vs Co-Pilot Setup', section: 'Agent Intelligence' },
    { key: 'interview-profiling', href: '/interview/profiling', icon: 'fa-wand-magic-sparkles', iconColor: 'text-blue-600', title: 'Middle Layer: Profiling', subtitle: 'Intelligent Role Blueprints', section: 'Agent Intelligence' },
    { key: 'interview-workspace', href: '/interview/workspace', icon: 'fa-cubes', iconColor: 'text-emerald-600', title: 'Universal Workspace', subtitle: '3-Layer Plug-and-Play Shell', section: 'Agent Intelligence' },
    { key: 'interview-call', href: '/interview/call', icon: 'fa-video', iconColor: 'text-blue-500', title: 'Universal Zoom Call', subtitle: 'Video Call + On-Demand Tools', section: 'Agent Intelligence' },
    { key: 'audio-input', href: '/audio-input', icon: 'fa-microphone', iconColor: 'text-blue-500', title: '1. Audio Input', subtitle: 'PCM & Waveforms', section: 'Pipeline Stages' },
    { key: 'asr', href: '/asr', icon: 'fa-brain', iconColor: 'text-emerald-500', title: '2. Speech Recognition', subtitle: 'ASR & Whisper', section: 'Pipeline Stages' },
    { key: 'nlp', href: '/nlp', icon: 'fa-gears', iconColor: 'text-amber-500', title: '3. NLP & Intent', subtitle: 'Entities & Tools', section: 'Pipeline Stages' },
    { key: 'dialogue', href: '/dialogue', icon: 'fa-comments', iconColor: 'text-purple-500', title: '4. Dialogue & Memory', subtitle: 'FSM & Context Store', section: 'Pipeline Stages' },
    { key: 'tts', href: '/tts', icon: 'fa-file-audio', iconColor: 'text-pink-500', title: '5. Text-to-Speech', subtitle: 'ElevenLabs & Pauses', section: 'Pipeline Stages' },
    { key: 'audio-output', href: '/audio-output', icon: 'fa-volume-high', iconColor: 'text-orange-500', title: '6. Audio Output', subtitle: 'Hardware & Web Sinks', section: 'Pipeline Stages' },
    { key: 'architecture', href: '/architecture', icon: 'fa-sitemap', iconColor: 'text-indigo-600', title: '7. SOLID Architecture', subtitle: 'Orchestrator & DIP', section: 'Architecture' }
  ];

  let currentSection = '';
  let linksHtml = '';

  menuItems.forEach(item => {
    if (item.section !== currentSection) {
      currentSection = item.section;
      linksHtml += `<div class="pt-3 px-3 pb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">${currentSection}</div>`;
    }

    const isActive = (item.key === activeKey) || (currentPath === item.href) || (currentPath === item.href + '/');
    const activeClass = isActive ? 'nav-link active' : 'nav-link text-slate-700 hover:bg-slate-100';

    linksHtml += `
      <a href="${item.href}" class="${activeClass} flex items-center space-x-3 px-3 py-2.5 rounded-lg transition text-xs">
        <i class="fa-solid ${item.icon} ${item.iconColor} text-sm w-4"></i>
        <div class="flex-1 truncate">
          <div class="truncate">${item.title}</div>
          <div class="text-[10px] text-slate-400 font-normal">${item.subtitle}</div>
        </div>
      </a>
    `;
  });

  sidebarContainer.innerHTML = `
    <aside class="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 h-screen sticky top-0 z-40 select-none">
      <div class="px-5 py-4 border-b border-slate-100 flex items-center space-x-3">
        <div class="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-indigo-100">
          <i class="fa-solid fa-waveform-lines"></i>
        </div>
        <div>
          <h1 class="text-sm font-extrabold text-slate-900 leading-tight">Conversational AI</h1>
          <p class="text-[10px] text-indigo-600 font-semibold uppercase tracking-wider">SOLID Microservices</p>
        </div>
      </div>

      <nav class="flex-1 overflow-y-auto py-3 px-2 space-y-1">
        ${linksHtml}
      </nav>

      <div class="p-3 border-t border-slate-100 bg-slate-50 text-[11px] space-y-1">
        <div class="flex items-center justify-between text-slate-500 font-medium">
          <span>Environment</span>
          <span class="font-mono text-indigo-600 font-bold">capstone_voice</span>
        </div>
        <div class="flex items-center space-x-1.5 text-emerald-600 font-semibold text-[10px]">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Gateway Live (127.0.0.1:8000)</span>
        </div>
      </div>
    </aside>
  `;
}
