import os
import json
import csv
from audio_input.audio_loader import load_audio, list_labeled_audio_files
from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from model_inference.dummy_model import DummySentimentModel, predict
from pipeline.logger import log

class PipelineRunner:
    def __init__(self, input_dir, output_dir, config):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.model = DummySentimentModel(num_classes=len(config["class_labels"]))

    def run(self):
        results = []
        log("🔄 Running pipeline...")

        for filepath, true_label in list_labeled_audio_files(self.input_dir):
            audio, sr = load_audio(filepath)
            segments = segment_audio(audio, sr,
                                     window_sec=self.config["segment_duration"],
                                     hop_sec=self.config["segment_overlap"])
            for i, segment in enumerate(segments):
                spec = compute_spectrogram(segment, sr)
                pred_label, probs = predict(self.model, spec)

                results.append({
                    "file": os.path.basename(filepath),
                    "true_label": true_label,
                    "segment": i,
                    "start_time": round(i * self.config["segment_overlap"], 2),
                    "predicted_label": self.config["class_labels"][pred_label],
                    "confidence": [round(float(p), 4) for p in probs]
                })

        self._save_outputs(results)

    def _save_outputs(self, results):
        os.makedirs(self.output_dir, exist_ok=True)

        # JSON
        json_path = os.path.join(self.output_dir, "predictions.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # CSV
        csv_path = os.path.join(self.output_dir, "predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        log(f"✅ Predictions saved to {self.output_dir}")
