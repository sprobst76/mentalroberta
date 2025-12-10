# Repository Guidelines

## Project Structure & Module Organization
- Package code lives in `mentalroberta/`.
- Core model: `mentalroberta/model.py`; inference CLI: `mentalroberta/inference.py`.
- Training & data prep: `mentalroberta/training/` (train loop + augmentation + download helpers).
- UI: `mentalroberta/apps/demo_app.py`; dev smoke test: `mentalroberta/tools/quick_test.py`.
- Data samples sit in `data/`; checkpoints save to `checkpoints/` (keep large files out of PRs unless essential).

## Setup, Build, and Development Commands
- Install deps inside a virtualenv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Smoke test model wiring: `python -m mentalroberta.tools.quick_test` (verifies forward pass and capsule lengths).
- Train on provided JSON: `python -m mentalroberta.training.train --data data/synthetic_data.json --epochs 3 --batch_size 8 --learning_rate 2e-5`.
- Run inference on a single text (requires a checkpoint): `python -m mentalroberta.inference --checkpoint checkpoints/model.pt --text "Ich fühle mich so leer..."`.
- Export ONNX (for CPU/edge/browser): `python -m mentalroberta.tools.export_onnx --checkpoint checkpoints/best_model.pt --output checkpoints/model.onnx --quantize`.
- Launch the demo (CPU/GPU both supported): `streamlit run mentalroberta/apps/demo_app.py` and choose the backend (PyTorch server, ONNX server-CPU, or ONNX download for client use).

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indentation; prefer descriptive lower_snake_case for functions/vars and PascalCase for classes (e.g., `MentalHealthDataset`).
- Keep modules small and focused; place new training utilities near related logic in `mentalroberta/training/`.
- Use type hints where practical; keep docstrings concise and task-oriented.
- Reuse preprocessing helpers (`MentalHealthDataset.preprocess` and `MentalHealthClassifier.preprocess`) instead of duplicating regex steps.

## Testing Guidelines
- Tests live in `tests/`; run `pytest -q`. They use dummy stubs to avoid network calls.
- For training changes, run a short epoch (`--epochs 1 --batch_size 4`) to confirm the loop, and capture F1 from stdout.
- For inference changes, run a small batch through `python -m mentalroberta.inference --interactive` or `--input sample.json --output preds.json` and spot-check labels.

## Commit & Pull Request Guidelines
- Use imperative, scoped commit messages (`feat: add capsule margin loss option`, `fix: guard tokenizer padding mask`); keep commits small and reviewable.
- PRs should include: brief summary, key commands run (train/test/demo), and any caveats about required hardware or dataset size.
- Link related issues when available; add screenshots only when changing the Streamlit UI.
- Do not commit large datasets or checkpoints; share download steps instead.
