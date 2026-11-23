/**
 * Enhanced 2D Reactive Human Avatar Component (Duolingo Style).
 * Features rich cartoon vector rigging, natural breathing squash-and-stretch,
 * eye glancing & blinking, phoneme-synced speech animation, and real-time streaming sentiment expressions.
 */

class ReactiveAvatar {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.name = options.name || 'Mira';
    this.roleTitle = options.roleTitle || 'AI Technical Interviewer';
    this.theme = options.theme || 'light'; // light or dark
    this.state = 'idle'; // idle, listening, speaking, thinking, impressed, skeptical
    this.sentiment = 0.0;
    this.isSpeaking = false;
    this.mouthInterval = null;
    this.blinkTimeout = null;
    this.glanceTimeout = null;

    this.render();
    this.startBlinkingLoop();
    this.startGlancingLoop();
  }

  render() {
    const isLight = this.theme === 'light';
    const bgClasses = isLight 
      ? 'bg-gradient-to-b from-slate-100 to-indigo-50/60 border border-slate-200' 
      : 'bg-gradient-to-b from-slate-900 to-indigo-950 border border-slate-800';
    const textPrimary = isLight ? 'text-slate-800' : 'text-white';
    const textSecondary = isLight ? 'text-indigo-600' : 'text-indigo-300';
    const badgeBg = isLight ? 'bg-white/90 text-slate-700 border-slate-200' : 'bg-slate-900/80 text-slate-300 border-slate-700/80';

    this.container.innerHTML = `
      <style>
        @keyframes avatar-breathe {
          0%, 100% { transform: translateY(0px) scale(1, 1); }
          50% { transform: translateY(-4px) scale(1.015, 0.985); }
        }
        @keyframes avatar-bounce-speak {
          0%, 100% { transform: translateY(0px) scale(1); }
          50% { transform: translateY(-3px) scale(1.02); }
        }
        .avatar-breathing {
          animation: avatar-breathe 4s ease-in-out infinite;
          transform-origin: center bottom;
        }
        .avatar-speaking {
          animation: avatar-bounce-speak 0.35s ease-in-out infinite;
          transform-origin: center bottom;
        }
      </style>

      <div class="relative w-full h-full flex flex-col items-center justify-between select-none overflow-hidden ${bgClasses} p-3 rounded-2xl shadow-xs transition-colors">
        
        <!-- Top Status Indicators Bar -->
        <div class="w-full flex items-center justify-between z-10">
          <!-- State Badge -->
          <div class="flex items-center space-x-1.5 ${badgeBg} backdrop-blur-xs px-2.5 py-1 rounded-full border text-[10px] font-mono shadow-xs">
            <span id="avatarPulseDot" class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span id="avatarStateLabel" class="font-bold">Listening</span>
          </div>

          <!-- Real-Time Sentiment Mood -->
          <div class="flex items-center space-x-1 ${badgeBg} backdrop-blur-xs px-2.5 py-1 rounded-full border text-[10px] font-bold shadow-xs">
            <span id="avatarSentimentIcon">😊</span>
            <span id="avatarSentimentText" class="font-mono text-emerald-600 font-bold">Attentive</span>
          </div>
        </div>

        <!-- 2D Animated Duolingo-Style Character Rig -->
        <div id="avatarRigWrapper" class="avatar-breathing relative w-44 h-44 md:w-48 md:h-48 my-auto flex items-center justify-center">
          <svg viewBox="0 0 240 240" class="w-full h-full filter drop-shadow-md" xmlns="http://www.w3.org/2000/svg">
            
            <!-- Audio Waves Halo -->
            <circle id="haloCircle" cx="120" cy="120" r="102" fill="none" stroke="rgba(99, 102, 241, 0.2)" stroke-width="4" stroke-dasharray="8 6" />

            <!-- Torso & Business Suit -->
            <path d="M 40 235 C 50 182, 82 172, 120 172 C 158 172, 190 182, 200 235 Z" fill="#312e81" />
            <!-- Shirt V-Neck Collar -->
            <path d="M 94 172 L 120 210 L 146 172 Z" fill="#ffffff" />
            <!-- Necktie / Accent Ribbon -->
            <path d="M 115 186 L 125 186 L 123 216 L 117 216 Z" fill="#4f46e5" />
            <!-- Lapels -->
            <path d="M 84 172 L 105 220 L 120 220 L 98 172 Z" fill="#4338ca" />
            <path d="M 156 172 L 135 220 L 120 220 L 142 172 Z" fill="#4338ca" />

            <!-- Neck -->
            <rect x="106" y="146" width="28" height="30" rx="6" fill="#fde68a" />
            <path d="M 106 148 C 114 158, 126 158, 134 148 L 134 158 C 126 165, 114 165, 106 158 Z" fill="#f59e0b" opacity="0.3" />

            <!-- Head & Face (Duolingo Round Style) -->
            <g id="headGroup" class="transition-transform duration-300" transform-origin="120 120">
              
              <!-- Ears -->
              <circle cx="56" cy="115" r="15" fill="#fde68a" stroke="#d97706" stroke-width="2" />
              <circle cx="184" cy="115" r="15" fill="#fde68a" stroke="#d97706" stroke-width="2" />
              <!-- Earring accents -->
              <circle cx="56" cy="124" r="3.5" fill="#6366f1" />
              <circle cx="184" cy="124" r="3.5" fill="#6366f1" />

              <!-- Main Face Sphere -->
              <ellipse cx="120" cy="112" rx="60" ry="65" fill="#fef3c7" stroke="#d97706" stroke-width="2.5" />

              <!-- Stylized Modern Hair (Duolingo Indigo Bob) -->
              <path d="M 56 108 C 46 50, 90 32, 120 32 C 162 32, 194 50, 184 108 C 176 62, 156 48, 120 48 C 84 48, 64 62, 56 108 Z" fill="#312e81" />
              <!-- Bangs & Top Tuft -->
              <path d="M 60 84 C 80 62, 110 62, 126 76 C 142 62, 172 68, 180 86 C 166 70, 136 68, 120 82 C 104 68, 74 70, 60 84 Z" fill="#4338ca" />

              <!-- Cheeks Blush -->
              <ellipse id="blushLeft" cx="78" cy="126" rx="9" ry="6" fill="#f43f5e" opacity="0.18" class="transition-opacity duration-300" />
              <ellipse id="blushRight" cx="162" cy="126" rx="9" ry="6" fill="#f43f5e" opacity="0.18" class="transition-opacity duration-300" />

              <!-- Eyebrows -->
              <g id="eyebrowsGroup">
                <path id="browLeft" d="M 74 91 Q 89 84 102 90" stroke="#1e1b4b" stroke-width="4" stroke-linecap="round" fill="none" class="transition-all duration-300" />
                <path id="browRight" d="M 138 90 Q 151 84 166 91" stroke="#1e1b4b" stroke-width="4" stroke-linecap="round" fill="none" class="transition-all duration-300" />
              </g>

              <!-- Left Eye Group -->
              <g id="eyeLeftGroup">
                <!-- Sclera (White) -->
                <ellipse cx="88" cy="107" rx="14" ry="15" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" />
                <!-- Iris -->
                <g id="pupilLeftGroup" class="transition-transform duration-200" transform-origin="88 107">
                  <ellipse cx="88" cy="107" rx="9" ry="9" fill="#4f46e5" />
                  <ellipse cx="88" cy="107" rx="5" ry="5" fill="#0f172a" />
                  <!-- Specular Catchlights (Duolingo Sparkle) -->
                  <circle cx="85" cy="103" r="3" fill="#ffffff" />
                  <circle cx="91" cy="109" r="1.5" fill="#ffffff" />
                </g>
                <!-- Happy Squint Eye Shape (Appears on positive sentiment) -->
                <path id="squintLeft" d="M 74 107 Q 88 96 102 107" stroke="#1e1b4b" stroke-width="4" stroke-linecap="round" fill="none" opacity="0" class="transition-opacity duration-200" />
                <!-- Eyelid for Blinking -->
                <path id="eyelidLeft" d="M 72 107 Q 88 88 104 107 Z" fill="#fef3c7" class="transition-all duration-75" style="transform: scaleY(0); transform-origin: 88px 97px;" />
              </g>

              <!-- Right Eye Group -->
              <g id="eyeRightGroup">
                <!-- Sclera (White) -->
                <ellipse cx="152" cy="107" rx="14" ry="15" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" />
                <!-- Iris -->
                <g id="pupilRightGroup" class="transition-transform duration-200" transform-origin="152 107">
                  <ellipse cx="152" cy="107" rx="9" ry="9" fill="#4f46e5" />
                  <ellipse cx="152" cy="107" rx="5" ry="5" fill="#0f172a" />
                  <!-- Specular Catchlights -->
                  <circle cx="149" cy="103" r="3" fill="#ffffff" />
                  <circle cx="155" cy="109" r="1.5" fill="#ffffff" />
                </g>
                <!-- Happy Squint Eye Shape -->
                <path id="squintRight" d="M 138 107 Q 152 96 166 107" stroke="#1e1b4b" stroke-width="4" stroke-linecap="round" fill="none" opacity="0" class="transition-opacity duration-200" />
                <!-- Eyelid for Blinking -->
                <path id="eyelidRight" d="M 136 107 Q 152 88 168 107 Z" fill="#fef3c7" class="transition-all duration-75" style="transform: scaleY(0); transform-origin: 152px 97px;" />
              </g>

              <!-- Cute Button Nose -->
              <path d="M 117 116 Q 120 123 124 121" stroke="#b45309" stroke-width="2.5" stroke-linecap="round" fill="none" />

              <!-- Expressive Cartoon Mouth Rig -->
              <g id="mouthRig" transform-origin="120 140">
                <!-- Resting Smile Curve -->
                <path id="mouthResting" d="M 105 138 Q 120 147 135 138" stroke="#be123c" stroke-width="3.5" stroke-linecap="round" fill="none" class="transition-all duration-150" />
                
                <!-- Open Talking Mouth / Big Grin Shape -->
                <g id="mouthOpenGroup" opacity="0" class="transition-opacity duration-150">
                  <path id="mouthOpenCavity" d="M 104 137 Q 120 156 136 137 Z" fill="#881337" stroke="#be123c" stroke-width="2.5" />
                  <!-- Upper White Teeth Line -->
                  <path d="M 107 138 Q 120 142 133 138 Z" fill="#ffffff" />
                  <!-- Tongue -->
                  <ellipse cx="120" cy="148" rx="8" ry="4" fill="#fb7185" />
                </g>
              </g>

              <!-- Geek Chic Smart Glasses -->
              <g id="glassesGroup">
                <rect x="70" y="94" width="36" height="26" rx="8" fill="rgba(255,255,255,0.12)" stroke="#1e1b4b" stroke-width="2.5" />
                <rect x="134" y="94" width="36" height="26" rx="8" fill="rgba(255,255,255,0.12)" stroke="#1e1b4b" stroke-width="2.5" />
                <path d="M 106 104 Q 120 101 134 104" stroke="#1e1b4b" stroke-width="2.5" fill="none" />
                <!-- Reflection stripes -->
                <line x1="75" y1="99" x2="88" y2="114" stroke="#ffffff" stroke-width="2" opacity="0.5" stroke-linecap="round" />
                <line x1="139" y1="99" x2="152" y2="114" stroke="#ffffff" stroke-width="2" opacity="0.5" stroke-linecap="round" />
              </g>

            </g>
          </svg>
        </div>

        <!-- Nameplate -->
        <div class="w-full text-center pb-0.5">
          <div class="text-xs font-black ${textPrimary} flex items-center justify-center space-x-1">
            <span>${this.name}</span>
            <i class="fa-solid fa-circle-check text-indigo-500 text-[10px]"></i>
          </div>
          <div class="text-[9px] ${textSecondary} font-mono tracking-wider">${this.roleTitle}</div>
        </div>

      </div>
    `;
  }

  startBlinkingLoop() {
    const doBlink = () => {
      const eyelidLeft = document.getElementById('eyelidLeft');
      const eyelidRight = document.getElementById('eyelidRight');
      if (eyelidLeft && eyelidRight) {
        eyelidLeft.style.transform = 'scaleY(2.2)';
        eyelidRight.style.transform = 'scaleY(2.2)';
        setTimeout(() => {
          eyelidLeft.style.transform = 'scaleY(0)';
          eyelidRight.style.transform = 'scaleY(0)';
        }, 130);
      }
      const nextBlink = 3000 + Math.random() * 4000;
      this.blinkTimeout = setTimeout(doBlink, nextBlink);
    };
    this.blinkTimeout = setTimeout(doBlink, 2200);
  }

  startGlancingLoop() {
    const doGlance = () => {
      const pupilL = document.getElementById('pupilLeftGroup');
      const pupilR = document.getElementById('pupilRightGroup');
      if (pupilL && pupilR && this.state !== 'speaking') {
        const directions = [
          'translate(0px, 0px)',
          'translate(2px, -2px)',
          'translate(-2px, 0px)',
          'translate(0px, 0px)'
        ];
        const dir = directions[Math.floor(Math.random() * directions.length)];
        pupilL.style.transform = dir;
        pupilR.style.transform = dir;
      }
      const nextGlance = 4000 + Math.random() * 3000;
      this.glanceTimeout = setTimeout(doGlance, nextGlance);
    };
    this.glanceTimeout = setTimeout(doGlance, 3500);
  }

  setState(newState) {
    this.state = newState;
    const dot = document.getElementById('avatarPulseDot');
    const label = document.getElementById('avatarStateLabel');
    const head = document.getElementById('headGroup');
    const wrapper = document.getElementById('avatarRigWrapper');

    if (newState === 'speaking') {
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-indigo-500 animate-ping';
      if (label) label.innerText = 'Speaking';
      if (wrapper) wrapper.className = 'avatar-speaking relative w-44 h-44 md:w-48 md:h-48 my-auto flex items-center justify-center';
      this.startSpeaking();
    } else if (newState === 'listening') {
      this.stopSpeaking();
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      if (label) label.innerText = 'Listening';
      if (head) head.style.transform = 'rotate(0deg)';
      if (wrapper) wrapper.className = 'avatar-breathing relative w-44 h-44 md:w-48 md:h-48 my-auto flex items-center justify-center';
    } else if (newState === 'thinking') {
      this.stopSpeaking();
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-amber-500 animate-bounce';
      if (label) label.innerText = 'Thinking...';
      if (head) head.style.transform = 'rotate(-5deg) translateY(-2px)';
      const pupilL = document.getElementById('pupilLeftGroup');
      const pupilR = document.getElementById('pupilRightGroup');
      if (pupilL && pupilR) {
        pupilL.style.transform = 'translate(1px, -3px)';
        pupilR.style.transform = 'translate(1px, -3px)';
      }
    } else if (newState === 'impressed') {
      this.stopSpeaking();
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      if (label) label.innerText = 'Impressed';
      if (head) head.style.transform = 'rotate(3deg) translateY(-3px)';
    } else if (newState === 'skeptical') {
      this.stopSpeaking();
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-rose-500';
      if (label) label.innerText = 'Evaluating';
      if (head) head.style.transform = 'rotate(6deg) translateY(2px)';
    } else {
      this.stopSpeaking();
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-slate-400';
      if (label) label.innerText = 'Idle';
      if (head) head.style.transform = 'rotate(0deg)';
      if (wrapper) wrapper.className = 'avatar-breathing relative w-44 h-44 md:w-48 md:h-48 my-auto flex items-center justify-center';
    }
  }

  setSentiment(sentimentScore, label = null) {
    this.sentiment = sentimentScore;
    const icon = document.getElementById('avatarSentimentIcon');
    const text = document.getElementById('avatarSentimentText');
    const browLeft = document.getElementById('browLeft');
    const browRight = document.getElementById('browRight');
    const blushLeft = document.getElementById('blushLeft');
    const blushRight = document.getElementById('blushRight');
    const squintL = document.getElementById('squintLeft');
    const squintR = document.getElementById('squintRight');

    if (sentimentScore > 0.25) {
      // Impressed / Positive
      if (icon) icon.innerText = '🌟';
      if (text) {
        text.innerText = label || 'Impressed';
        text.className = 'font-mono text-emerald-600 font-bold';
      }
      if (browLeft) browLeft.setAttribute('d', 'M 74 88 Q 89 80 102 88');
      if (browRight) browRight.setAttribute('d', 'M 138 88 Q 151 80 166 88');
      if (blushLeft) blushLeft.setAttribute('opacity', '0.5');
      if (blushRight) blushRight.setAttribute('opacity', '0.5');
      if (squintL) squintL.setAttribute('opacity', '0.9');
      if (squintR) squintR.setAttribute('opacity', '0.9');
      this.setState('impressed');
    } else if (sentimentScore < -0.2) {
      // Critical / Dubious / Inquisitive
      if (icon) icon.innerText = '🧐';
      if (text) {
        text.innerText = label || 'Evaluating';
        text.className = 'font-mono text-amber-600 font-bold';
      }
      if (browLeft) browLeft.setAttribute('d', 'M 74 94 Q 89 92 102 96');
      if (browRight) browRight.setAttribute('d', 'M 138 84 Q 151 78 166 86'); // Raised right eyebrow
      if (blushLeft) blushLeft.setAttribute('opacity', '0.1');
      if (blushRight) blushRight.setAttribute('opacity', '0.1');
      if (squintL) squintL.setAttribute('opacity', '0');
      if (squintR) squintR.setAttribute('opacity', '0');
      this.setState('skeptical');
    } else {
      // Attentive Neutral
      if (icon) icon.innerText = '😊';
      if (text) {
        text.innerText = label || 'Attentive';
        text.className = 'font-mono text-indigo-600 font-bold';
      }
      if (browLeft) browLeft.setAttribute('d', 'M 74 91 Q 89 84 102 90');
      if (browRight) browRight.setAttribute('d', 'M 138 90 Q 151 84 166 91');
      if (blushLeft) blushLeft.setAttribute('opacity', '0.18');
      if (blushRight) blushRight.setAttribute('opacity', '0.18');
      if (squintL) squintL.setAttribute('opacity', '0');
      if (squintR) squintR.setAttribute('opacity', '0');
      this.setState('listening');
    }
  }

  startSpeaking() {
    if (this.isSpeaking) return;
    this.isSpeaking = true;

    const resting = document.getElementById('mouthResting');
    const openGroup = document.getElementById('mouthOpenGroup');
    const cavity = document.getElementById('mouthOpenCavity');
    const halo = document.getElementById('haloCircle');

    if (resting) resting.setAttribute('opacity', '0');
    if (openGroup) openGroup.setAttribute('opacity', '1');
    if (halo) halo.setAttribute('stroke', '#6366f1');

    let toggle = false;
    this.mouthInterval = setInterval(() => {
      toggle = !toggle;
      if (cavity) {
        if (toggle) {
          cavity.setAttribute('d', 'M 104 137 Q 120 157 136 137 Z');
        } else {
          cavity.setAttribute('d', 'M 106 138 Q 120 148 134 138 Z');
        }
      }
    }, 150);
  }

  stopSpeaking() {
    this.isSpeaking = false;
    if (this.mouthInterval) {
      clearInterval(this.mouthInterval);
      this.mouthInterval = null;
    }

    const resting = document.getElementById('mouthResting');
    const openGroup = document.getElementById('mouthOpenGroup');
    const halo = document.getElementById('haloCircle');

    if (resting) resting.setAttribute('opacity', '1');
    if (openGroup) openGroup.setAttribute('opacity', '0');
    if (halo) halo.setAttribute('stroke', 'rgba(99, 102, 241, 0.2)');
  }
}
