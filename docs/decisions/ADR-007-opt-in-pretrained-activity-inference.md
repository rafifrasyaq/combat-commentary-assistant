# ADR-007: Optional unverified pretrained activity inference (M5B)

- **Status:** Accepted for local experimentation, pending evaluation
- **Context:** M5A already outputs bounded clips, whereas M4 scores pixel differences
  and cannot identify strikes or distinguish camera changes reliably.
- **Decision:** Use `MCG-NJU/videomae-base-finetuned-kinetics` with PyAV for
  decoding and uniform `model.config.num_frames` sampling; avoid TorchCodec
  on this Windows baseline. Import heavy dependencies only when enabled.
  Reuse a model instance per Python process and serialize GPU inference.
  Keep the Python/Java `/analyze` response contract unchanged.
- **Activation:** `ACTION_RECOGNITION_ENABLED=true`; disabled by default.
  A standalone CLI reuses prior job clips without new uploads.
- **Artifacts:** `action_predictions.json` plus an optional additive
  `actionRecognition` section and manifest schemaVersion 4.
- **Semantics:** Kinetics activity labels and uncalibrated softmax scores only;
  NEVER describe as verified punch, contact, or strike technique. Low-top-score
  display may abstain (`UNKNOWN`), but high score does not prove a strike.
- **Failure:** An enabled inference error makes new processing fail explicitly,
  instead of silently completing with fabricated results.
- **Trade-offs:** Requires GPU, extra dependencies and model download; Java has
  a 120s HTTP timeout, so bootstrap the model/cache before enabling automatic
  jobs. Kinetics checkpoint noncommercial license limits deployment.
- **Deferred:** Training/fine-tuning, benchmarking, model serving queue,
  detection independent of M4 motion heuristic, API for activity predictions.
