const recordBtn = document.getElementById("recordBtn");
const speechOutput = document.getElementById("speechOutput");
const llmOutput = document.getElementById("llmOutput");

const TRANSCRIBE_URL = "http://localhost:5000/transcribe";
const CHAT_URL = "http://localhost:5000/chat";
const BACKEND_BASE_URL = new URL(CHAT_URL).origin;
const AUDIO_JOB_URL = `${BACKEND_BASE_URL}/audio-jobs`;

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isTranscribing = false;
let currentAudio = null;
let currentAudioJobToken = 0;

const transcriptHandlers = [];
export function onTranscript(handler) {
  if (typeof handler === "function") transcriptHandlers.push(handler);
}

function setRecordingUI(recording) {
  if (!recordBtn) return;
  recordBtn.textContent = recording ? "Stop Recording" : "Start Recording";
}

async function transcribeWithWhisper(audioBlob) {
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.webm");
  try {
    const response = await fetch(TRANSCRIBE_URL, { method: "POST", body: formData });
    if (!response.ok) throw new Error("Transcription failed");
    const result = await response.json();
    return result.text;
  } catch (err) {
    return { error: "Could not transcribe audio. Make sure the Whisper server is running." };
  }
}

async function askGemma(transcriptText) {
  try {
    const response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: transcriptText }),
    });

    const data = await response.json();
    if (!response.ok) {
      const message = data && data.error ? data.error : "LLM request failed";
      return { error: message };
    }

    if (data && (typeof data.english === "string" || typeof data.japanese === "string")) {
      return {
        japanese: typeof data.japanese === "string" ? data.japanese.trim() : "",
        english: typeof data.english === "string" ? data.english.trim() : "",
        audioUrl: typeof data.audio_url === "string" ? data.audio_url.trim() : "",
        audioJobId: typeof data.audio_job_id === "string" ? data.audio_job_id.trim() : "",
        audioPending: Boolean(data.audio_pending),
        ttsError: typeof data.tts_error === "string" ? data.tts_error.trim() : "",
        model: data.model || "Gemma 3",
      };
    }
    return { error: "Invalid LLM response format" };
  } catch (err) {
    return { error: "Could not reach OpenRouter chat endpoint. Make sure whisper_server.py is running." };
  }
}

async function waitForAudioJob(audioJobId, token) {
  if (!audioJobId) return null;

  for (let attempt = 0; attempt < 30; attempt += 1) {
    if (token !== currentAudioJobToken) return null;
    try {
      const response = await fetch(`${AUDIO_JOB_URL}/${encodeURIComponent(audioJobId)}`);
      if (!response.ok) {
        return null;
      }
      const job = await response.json();
      if (token !== currentAudioJobToken) return null;
      if (job && typeof job.audio_url === "string" && job.audio_url.trim()) {
        return job.audio_url.trim();
      }
      if (job && job.status === "error") {
        if (llmOutput && typeof job.tts_error === "string" && job.tts_error.trim()) {
          llmOutput.dataset.ttsError = job.tts_error.trim();
          llmOutput.title = job.tts_error.trim();
        }
        return null;
      }
    } catch (err) {
      return null;
    }
    await new Promise(resolve => setTimeout(resolve, 350));
  }
  return null;
}

async function playBackendAudio(audioUrl, token) {
  if (!audioUrl || token !== currentAudioJobToken) return;
  const src = audioUrl.startsWith("http")
    ? audioUrl
    : `${BACKEND_BASE_URL}${audioUrl}`;
  try {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    currentAudio = new Audio(src);
    currentAudio.play().catch(() => {});
  } catch (e) {
    console.error("Audio playback failed", e);
  }
}

async function startRecording() {
  if (isRecording || isTranscribing) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (speechOutput) {
      speechOutput.textContent = "Audio recording is not supported in this browser.";
      speechOutput.style.color = "#dc3545";
    }
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    isRecording = true;
    setRecordingUI(true);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      isRecording = false;
      setRecordingUI(false);
      if (!audioChunks.length) {
        if (speechOutput) {
          speechOutput.textContent = "No audio captured. Try again.";
          speechOutput.style.color = "#dc3545";
        }
        stream.getTracks().forEach(t => t.stop());
        return;
      }

      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      if (speechOutput) {
        speechOutput.textContent = "Transcribing...";
        speechOutput.style.color = "#667eea";
      }
      isTranscribing = true;
      if (recordBtn) recordBtn.disabled = true;

      const transcript = await transcribeWithWhisper(audioBlob);
      if (transcript && typeof transcript === "string") {
        const cleaned = transcript.trim();
        if (speechOutput) {
          speechOutput.textContent = `You said: "${cleaned}"`;
          speechOutput.style.color = "#28a745";
        }

        if (llmOutput) {
          llmOutput.textContent = "Kei is thinking...";
          llmOutput.style.color = "#333";
        }

        const llmResult = await askGemma(cleaned);
        if (llmOutput) {
          if (llmResult.english) {
            llmOutput.textContent = `Kei: ${llmResult.english}`;
            llmOutput.dataset.japanese = llmResult.japanese || "";
            llmOutput.dataset.english = llmResult.english || "";
            llmOutput.style.color = "#1f3c88";
            delete llmOutput.dataset.ttsError;
            llmOutput.title = "";

            currentAudioJobToken += 1;
            const token = currentAudioJobToken;
            if (llmResult.audioUrl) {
              playBackendAudio(llmResult.audioUrl, token);
            } else if (llmResult.audioPending && llmResult.audioJobId) {
              waitForAudioJob(llmResult.audioJobId, token).then((readyAudioUrl) => {
                if (readyAudioUrl) playBackendAudio(readyAudioUrl, token);
              });
            } else if (llmResult.ttsError) {
              console.warn(llmResult.ttsError);
              llmOutput.dataset.ttsError = llmResult.ttsError;
              llmOutput.title = llmResult.ttsError;
            }
          } else {
            llmOutput.textContent = `Kei error: ${llmResult.error || "Unknown error"}`;
            llmOutput.style.color = "#dc3545";
          }
        }

        transcriptHandlers.forEach(h => {
          try { h(cleaned); } catch (e) { console.error(e); }
        });
      } else if (transcript && transcript.error) {
        if (speechOutput) {
          speechOutput.textContent = `Transcription failed: ${transcript.error}`;
          speechOutput.style.color = "#dc3545";
        }
        if (llmOutput) llmOutput.textContent = "";
      } else {
        if (speechOutput) {
          speechOutput.textContent = "Transcription failed. Try again.";
          speechOutput.style.color = "#dc3545";
        }
        if (llmOutput) llmOutput.textContent = "";
      }

      stream.getTracks().forEach(t => t.stop());
      isTranscribing = false;
      if (recordBtn) recordBtn.disabled = false;
    };

    mediaRecorder.start();
    if (speechOutput) {
      speechOutput.textContent = "Recording... Press Enter again to stop.";
      speechOutput.style.color = "#667eea";
    }
  } catch (err) {
    isRecording = false;
    setRecordingUI(false);
    if (speechOutput) {
      speechOutput.textContent = "Mic permission denied. Please allow microphone access.";
      speechOutput.style.color = "#dc3545";
    }
    if (recordBtn) recordBtn.disabled = false;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

async function toggleRecording() {
  if (isTranscribing) return;
  if (isRecording) stopRecording();
  else await startRecording();
}

if (recordBtn) recordBtn.onclick = () => { toggleRecording(); };

document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.repeat) return;
  const target = event.target;
  const tag = target && target.tagName;
  const isTypingField = target && (target.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT");
  if (isTypingField) return;
  event.preventDefault();
  toggleRecording();
});

setRecordingUI(false);

