export function createAudioPlayer(onVolume) {
  let audio = new Audio();
  audio.crossOrigin = 'anonymous';
  let ctx = null;
  let source = null;
  let analyser = null;
  let data = null;
  let raf = null;

  function ensureContext() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    source = ctx.createMediaElementSource(audio);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    data = new Uint8Array(analyser.frequencyBinCount);
    source.connect(analyser);
    analyser.connect(ctx.destination);
  }

  function startAnalyse() {
    if (!analyser) return;
    const loop = () => {
      analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const avg = sum / data.length;
      const mouth = Math.min(1, avg / 200);
      if (onVolume) onVolume(mouth);
      raf = requestAnimationFrame(loop);
    };
    cancelAnalyse();
    raf = requestAnimationFrame(loop);
  }

  function cancelAnalyse() {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
  }

  function play(url) {
    ensureContext();
    audio.src = url;
    audio.play().then(() => startAnalyse()).catch(() => {
      try { ctx.resume(); startAnalyse(); } catch (e) {}
    });
  }

  function stop() {
    cancelAnalyse();
    audio.pause();
    audio.currentTime = 0;
  }

  return {
    play,
    stop,
    audio
  };
}
