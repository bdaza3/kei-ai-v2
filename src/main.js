import loadAvatar from './avatar.js';
import { createAudioPlayer } from './audio.js';

document.addEventListener('DOMContentLoaded', async () => {
  const avatarContainer = document.getElementById('avatar-container');
  const modelUrl = './model/kei.vrm';
  let avatar = null;
  try {
    avatar = await loadAvatar(avatarContainer, modelUrl);
  } catch (e) {
    console.error('Failed to load avatar', e);
  }

  const player = createAudioPlayer((mouth) => {
    if (avatar && avatar.setMouth) avatar.setMouth(mouth);
  });

  const fileInput = document.getElementById('testAudioFile');
  const playBtn = document.getElementById('playTestAudioBtn');
  if (playBtn) playBtn.addEventListener('click', () => {
    if (fileInput && fileInput.files && fileInput.files[0]) {
      const url = URL.createObjectURL(fileInput.files[0]);
      player.play(url);
      if (avatar && avatar.setEmotion) avatar.setEmotion('happy');
    } else {
      alert('Please select an audio file to test lip sync.');
    }
  });

  const btnHappy = document.getElementById('btnHappy');
  const btnNeutral = document.getElementById('btnNeutral');
  const btnSad = document.getElementById('btnSad');
  if (btnHappy) btnHappy.addEventListener('click', () => avatar && avatar.setEmotion && avatar.setEmotion('happy'));
  if (btnNeutral) btnNeutral.addEventListener('click', () => avatar && avatar.setEmotion && avatar.setEmotion('neutral'));
  if (btnSad) btnSad.addEventListener('click', () => avatar && avatar.setEmotion && avatar.setEmotion('sad'));

  // Example: when chat response arrives, use setEmotion + play audio + show text
  window.keiUi = {
    playResponse: async (audioUrl, emotion, text) => {
      if (avatar && avatar.setEmotion) avatar.setEmotion('thinking');
      if (emotion && avatar && avatar.setEmotion) avatar.setEmotion(emotion);
      if (audioUrl) player.play(audioUrl);
      if (text) {
        const out = document.getElementById('llmOutput');
        if (out) out.textContent = text;
      }
    }
  };
});
