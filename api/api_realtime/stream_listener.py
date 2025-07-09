import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from threading import Thread, Event
from collections import deque
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import requests
import tempfile
import os
import time
import uuid
from datetime import datetime, timezone
import queue
import sys

# === Configuration ===
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SECONDS = 2
SEGMENT_SECONDS = 2.0
OVERLAP_SECONDS = 0.25
SEND_INTERVAL = 0.75
ENDPOINT = "http://localhost:8010/infer"
SOURCE_NAME = "mic_stream"
QUEUE_MAXSIZE = 10

# === Session UUID and Shared Structures ===
SESSION_ID = str(uuid.uuid4())
audio_buffer = deque(maxlen=int(SAMPLE_RATE * BUFFER_SECONDS))
segment_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

# Control flags
stop_event = Event()
stream = None

# === Redirect Print to GUI ===
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        self.widget.insert(tk.END, string)
        self.widget.see(tk.END)

    def flush(self):
        pass

# === Audio Stream Callback ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio stream status: {status}")
    audio_buffer.extend(indata[:, 0])

# === Producer Thread ===
def segment_producer():
    segment_len = int(SEGMENT_SECONDS * SAMPLE_RATE)
    segment_counter = 0
    while not stop_event.is_set():
        time.sleep(SEND_INTERVAL)
        if len(audio_buffer) >= segment_len:
            segment = list(audio_buffer)[-segment_len:]
            segment_np = np.array(segment, dtype='int16')

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav.write(tmp.name, SAMPLE_RATE, segment_np)
                segment_queue.put((tmp.name, segment_counter))
                segment_counter += 1

# === Consumer Thread ===
def segment_consumer():
    while not stop_event.is_set():
        try:
            path, segment_idx = segment_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            print(f"\nCaptured Clip {segment_idx} ({SEGMENT_SECONDS:.1f}s)")
            print(f"Saved to: {path}")
            print(f"Sending to: {ENDPOINT} ...")

            start_time = time.perf_counter()
            with open(path, 'rb') as f:
                response = requests.post(
                    ENDPOINT,
                    files={'file': (os.path.basename(path), f, 'audio/wav')},
                    data={
                        'source': SOURCE_NAME,
                        'session_id': SESSION_ID,
                        'segment_idx': segment_idx
                    },
                    timeout=10
                )
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)

            if response.status_code == 200:
                result = response.json()
                print(f"Inference Response Time: {elapsed} ms")

                print(f"Clip {segment_idx} | {result['segments_processed']} Segments")
                for pred in result.get("predictions", []):
                    label = pred.get('label')
                    confidence = pred.get('confidence')
                    smoothed_label = pred.get('smoothed_label')
                    smoothed_conf = pred.get('smoothed_confidence')
                    latency = pred.get('latency_ms')
                    print(f"→ [{pred.get('segment')}] {label} ({confidence}), "
                          f"Smoothed: {smoothed_label} ({smoothed_conf}), Latency: {latency} ms")

                if "influx_log_time_ms" in result:
                    print(f"InfluxDB write took {result['influx_log_time_ms']} ms")

                print(f"Clip {segment_idx} processed | Avg latency: {result.get('avg_latency_ms')} ms\n")
            else:
                print(f"Inference failed: {response.status_code} {response.text}")

        except Exception as e:
            print(f"Error sending clip {segment_idx}: {e}")

        finally:
            try:
                os.remove(path)
            except:
                pass

# === Start / Stop Handlers ===
def start_stream():
    if stop_event.is_set():
        stop_event.clear()

    global stream
    stream = sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    stream.start()
    print(f"\nListening started. Session ID: {SESSION_ID}\n")

    Thread(target=segment_producer, daemon=True).start()
    Thread(target=segment_consumer, daemon=True).start()

def stop_stream():
    stop_event.set()
    global stream
    if stream:
        try:
            stream.stop()
            stream.close()
            print("\nListening stopped.\n")
        except Exception as e:
            print(f"\nFailed to stop stream: {e}")

def quit_app(root):
    stop_stream()
    root.destroy()

# === GUI Setup ===
def launch_gui():
    root = tk.Tk()
    root.title("Real-Time Sentiment Detection")

    start_btn = tk.Button(root, text="Start Listening", command=start_stream, bg='green', fg='white', width=20)
    start_btn.grid(row=0, column=0, padx=10, pady=10)

    stop_btn = tk.Button(root, text="Stop Listening", command=stop_stream, bg='orange', fg='black', width=20)
    stop_btn.grid(row=0, column=1, padx=10, pady=10)

    quit_btn = tk.Button(root, text="Quit", command=lambda: quit_app(root), bg='red', fg='white', width=20)
    quit_btn.grid(row=0, column=2, padx=10, pady=10)

    log_output = ScrolledText(root, height=25, width=100, bg='black', fg='lime', font=("Courier", 10))
    log_output.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

    sys.stdout = TextRedirector(log_output)
    print("Application ready. Press 'Start Listening' to begin...\n")

    root.protocol("WM_DELETE_WINDOW", lambda: quit_app(root))
    root.mainloop()

# === Entry Point ===
if __name__ == "__main__":
    launch_gui()
