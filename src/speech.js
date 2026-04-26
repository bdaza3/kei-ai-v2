const recordBtn = document.getElementById("recordBtn");
const speechOutput = document.getElementById("speechOutput");
const llmOutput = document.getElementById("llmOutput");

const TRANSCRIBE_URL = "http://localhost:5000/transcribe";
const CHAT_URL = "http://localhost:5000/chat";

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isTranscribing = false;

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

    if (data && typeof data.reply === "string") {
      return { reply: data.reply.trim(), model: data.model || "Gemma 3" };
    }
    return { error: "Invalid LLM response format" };
  } catch (err) {
    return { error: "Could not reach OpenRouter chat endpoint. Make sure whisper_server.py is running." };
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
          llmOutput.textContent = "Gemma 3 is thinking...";
          llmOutput.style.color = "#333";
        }

        const llmResult = await askGemma(cleaned);
        if (llmOutput) {
          if (llmResult.reply) {
            llmOutput.textContent = `Gemma 3: ${llmResult.reply}`;
            llmOutput.style.color = "#1f3c88";
          } else {
            llmOutput.textContent = `Gemma 3 error: ${llmResult.error || "Unknown error"}`;
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

