import types

import pytest
import torch

from mentalroberta.model import MentalRoBERTaCaps
from mentalroberta.tools.export_onnx import export_onnx


@pytest.fixture
def dummy_roberta(monkeypatch):
    class DummyRoberta:
        def __init__(self, hidden_size=8, num_layers=2):
            self.config = types.SimpleNamespace(hidden_size=hidden_size)
            self.num_layers = num_layers

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
            batch, seq_len = input_ids.shape
            hidden_states = [
                torch.full((batch, seq_len, self.config.hidden_size), float(i))
                for i in range(self.num_layers + 1)
            ]
            return types.SimpleNamespace(hidden_states=hidden_states)

    monkeypatch.setattr("mentalroberta.model.AutoModel", DummyRoberta)
    return DummyRoberta


@pytest.mark.skipif(
    pytest.importorskip("onnxruntime", reason="onnxruntime required for ONNX comparison") is None,
    reason="onnxruntime not installed",
)
def test_onnx_export_matches_pytorch(tmp_path, dummy_roberta):
    import onnxruntime as ort

    torch.manual_seed(0)

    # Build small model and checkpoint
    model = MentalRoBERTaCaps(num_classes=3, num_layers=2, model_name="dummy")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)

    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        checkpoint=ckpt_path,
        output=onnx_path,
        model_name="dummy",
        num_classes=3,
        opset=17,
        max_length=8,
    )

    # Prepare inputs
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])

    # PyTorch output
    logits_pt, _ = model(input_ids, attention_mask)
    logits_pt = logits_pt.detach().numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    ort_logits, _ = session.run(
        None,
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
    )

    assert logits_pt.shape == ort_logits[0].shape
    assert torch.allclose(torch.from_numpy(ort_logits[0]), torch.from_numpy(logits_pt), atol=1e-4)
