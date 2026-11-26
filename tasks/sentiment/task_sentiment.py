"""Crowd sentiment in football stadiums: approval, disapproval or neutral.

Built on the framework's ``ClassificationTask``, which already provides the
loss (from ``criterion`` / ``class_weights`` in config.yaml), training and the
standard output format. This file only changes:

* the model: ResNet18 with a one-channel input and a small classifier head;
* the metrics: adds weighted F1 to accuracy and macro F1;
* the output: adds ``probabilities`` next to ``scores`` (same values, the name
  older GRASS outputs used);
* the InfluxDB record written by batch inference;
* an optional one-line LLM explanation of a prediction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
from torchvision import models

from audio_ops import ClassificationTask, register_task


@register_task("sentiment")
class SentimentTask(ClassificationTask):

    def build_model(self, input_shape) -> nn.Module:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            1, model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
            padding=model.conv1.padding, bias=False,
        )
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, self.num_classes),
        )
        return model

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        from sklearn.metrics import f1_score

        metrics = super().compute_metrics(y_true, y_pred)
        pred = y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred
        metrics["f1_weighted"] = float(f1_score(y_true, pred, average="weighted",
                                                zero_division=0))
        return metrics

    def postprocess(self, raw_output) -> Dict[str, Any]:
        result = super().postprocess(raw_output)
        result["probabilities"] = dict(result["scores"])
        return result

    # ── Batch inference → InfluxDB (measurement ``audio_inference``) ──────────
    # The live service writes its own measurement, ``inference_sentiment``, with
    # label, confidence and score_<class> per stream; that is what the Grafana
    # dashboard reads.

    def format_influx_record(self, segment_id: str, prediction: Any,
                             audio_offset_sec: Optional[float] = None) -> Dict[str, Any]:
        if isinstance(prediction, list):
            prediction = prediction[0]
        label = prediction.get("label", "unknown")
        fields: Dict[str, Any] = {"label": label,
                                  "confidence": prediction.get("confidence", 0.0)}
        for cls, p in prediction.get("scores", {}).items():
            fields[f"prob_{cls}"] = p
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec
        return {"fields": fields, "tags": {"predicted_label": label}}

    # ── Optional LLM explanation (ENABLE_LLM=true) ────────────────────────────

    def postprocess_with_llm(self, raw_result, context=None, llm_client=None):
        if llm_client is None:
            return raw_result
        pred = raw_result if isinstance(raw_result, dict) else raw_result[0]
        explanation = llm_client.complete(
            f"In one sentence, explain what a '{pred['label']}' crowd sentiment "
            f"(confidence {pred['confidence']:.0%}) in stadium audio means.",
            system="You are an audio analysis assistant. Be very concise.",
        )
        if explanation:
            pred["explanation"] = explanation
        return raw_result
