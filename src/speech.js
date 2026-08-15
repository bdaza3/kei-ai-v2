const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const speechOutput = document.getElementById("speechOutput");
const llmOutput = document.getElementById("llmOutput");

const CHAT_URL = "http://localhost:5000/chat";

async function askKei(text) {
  try {
    const response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
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
      };
    }

    return { error: "Invalid LLM response format" };
  } catch (err) {
    return { error: "Could not reach the chat endpoint. Make sure whisper_server.py is running." };
  }
}

async function submitChat() {
  const text = chatInput ? chatInput.value.trim() : "";
  if (!text) {
    if (speechOutput) {
      speechOutput.textContent = "Type a message before sending.";
      speechOutput.style.color = "#dc3545";
    }
    return;
  }

  if (speechOutput) {
    speechOutput.textContent = "Kei is thinking...";
    speechOutput.style.color = "#667eea";
  }
  if (llmOutput) {
    llmOutput.textContent = "";
  }
  if (sendBtn) sendBtn.disabled = true;

  try {
    const result = await askKei(text);

    if (result.error) {
      if (speechOutput) {
        speechOutput.textContent = result.error;
        speechOutput.style.color = "#dc3545";
      }
      if (llmOutput) llmOutput.textContent = "";
      return;
    }

    const displayText = result.english || result.japanese || "Kei did not return a reply.";
    if (llmOutput) {
      llmOutput.textContent = `Kei: ${displayText}`;
      llmOutput.style.color = "#1f3c88";
    }

    if (speechOutput) {
      speechOutput.textContent = `You: ${text}`;
      speechOutput.style.color = "#28a745";
    }
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    if (chatInput) {
      chatInput.value = "";
      chatInput.focus();
    }
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitChat();
  });
}

if (chatInput) {
  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submitChat();
    }
  });
}

