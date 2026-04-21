from voice_input import VoiceInput
import time

v = VoiceInput()
print("Waiting for model to load...")
loaded = v.wait_loaded(timeout=30)
print("Model loaded:" , loaded)
print("Recording 4s — speak now")
v.start_recording()
time.sleep(4)
v.stop_and_transcribe()
print("Waiting for transcription...")
# collect transcripts (partials then final) for up to 10 seconds
transcripts = []
end = time.time() + 10
while time.time() < end:
	t = v.get_nowait()
	if t:
		transcripts.append(t)
	else:
		time.sleep(0.3)

final = transcripts[-1] if transcripts else None
print("Transcripts (collected):", transcripts)
print("Final transcript:", final)