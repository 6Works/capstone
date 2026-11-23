/**
 * In-Browser Web Audio API 16-bit Mono PCM WAV Recorder.
 * Captures microphone audio frames and encodes them into valid 16000 Hz WAV.
 */
class PCMRecorder {
  constructor(targetSampleRate = 16000) {
    this.targetSampleRate = targetSampleRate;
    this.audioCtx = null;
    this.micStream = null;
    this.scriptProcessor = null;
    this.pcmBuffers = [];
    this.isRecording = false;
  }

  async start() {
    this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.audioCtx.createMediaStreamSource(this.micStream);

    this.pcmBuffers = [];
    this.scriptProcessor = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.scriptProcessor.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.pcmBuffers.push(new Float32Array(input));
    };

    source.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioCtx.destination);
    this.isRecording = true;
  }

  async stop() {
    this.isRecording = false;
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }
    const currentRate = this.audioCtx ? this.audioCtx.sampleRate : 44100;
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      await this.audioCtx.close();
    }

    let totalLength = 0;
    for (const b of this.pcmBuffers) totalLength += b.length;
    if (totalLength === 0) return null;

    const flattened = new Float32Array(totalLength);
    let offset = 0;
    for (const b of this.pcmBuffers) {
      flattened.set(b, offset);
      offset += b.length;
    }

    return this.encodeWAV(flattened, currentRate);
  }

  encodeWAV(samples, sampleRate) {
    let targetRate = this.targetSampleRate;
    let targetSamples = samples;
    if (sampleRate !== targetRate) {
      const ratio = sampleRate / targetRate;
      const newLen = Math.round(samples.length / ratio);
      targetSamples = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        targetSamples[i] = samples[Math.round(i * ratio)];
      }
    } else {
      targetRate = sampleRate;
    }

    const buffer = new ArrayBuffer(44 + targetSamples.length * 2);
    const view = new DataView(buffer);

    function writeString(v, off, str) {
      for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i));
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + targetSamples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM = 1
    view.setUint16(22, 1, true); // Mono = 1
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, 'data');
    view.setUint32(40, targetSamples.length * 2, true);

    let pcmOff = 44;
    for (let i = 0; i < targetSamples.length; i++, pcmOff += 2) {
      let s = Math.max(-1, Math.min(1, targetSamples[i]));
      view.setInt16(pcmOff, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }
}
