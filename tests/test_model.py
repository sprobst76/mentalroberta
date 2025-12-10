import types

import pytest
import torch

from mentalroberta import model


@pytest.fixture
def dummy_roberta(monkeypatch):
    class DummyRoberta:
        def __init__(self, hidden_size=32, num_layers=6):
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

    monkeypatch.setattr(model, "AutoModel", DummyRoberta)
    return DummyRoberta


@pytest.fixture
def dummy_tokenizer(monkeypatch):
    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, text, return_tensors=None, max_length=None, truncation=None, padding=None):
            if isinstance(text, list):
                batch = len(text)
                return {
                    "input_ids": torch.ones((batch, 4), dtype=torch.long),
                    "attention_mask": torch.ones((batch, 4), dtype=torch.long),
                }

            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    monkeypatch.setattr(model, "AutoTokenizer", DummyTokenizer)
    return DummyTokenizer


def test_capsule_layer_outputs_within_unit_length():
    layer = model.CapsuleLayer(
        input_dim=8,
        num_primary_caps=3,
        primary_cap_dim=4,
        num_class_caps=2,
        class_cap_dim=5,
        num_routing_iterations=2,
    )
    dummy_input = torch.randn(4, 8)

    outputs = layer(dummy_input)
    lengths = torch.sqrt((outputs ** 2).sum(dim=-1))

    assert outputs.shape == (4, 2, 5)
    assert torch.all(lengths <= 1.0001)
    assert torch.all(lengths >= 0)


def test_model_forward_uses_configured_layer(monkeypatch, dummy_roberta):
    captured = {}

    def fake_capsule_forward(self, x):
        captured["cls_input"] = x.detach().clone()
        return torch.zeros((x.size(0), self.num_class_caps, self.class_cap_dim), device=x.device)

    monkeypatch.setattr(model.CapsuleLayer, "forward", fake_capsule_forward, raising=False)

    net = model.MentalRoBERTaCaps(num_classes=5, num_layers=3, model_name="dummy")
    net.eval()

    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.ones((2, 4), dtype=torch.long)

    logits, capsule_outputs = net(input_ids, attention_mask)

    assert logits.shape == (2, 5)
    assert capsule_outputs.shape == (2, 5, net.capsule.class_cap_dim)
    assert "cls_input" in captured
    assert torch.allclose(captured["cls_input"], torch.full((2, net.hidden_size), 3.0))


def test_get_capsule_lengths_matches_norm(monkeypatch, dummy_roberta):
    net = model.MentalRoBERTaCaps(num_classes=3, class_cap_dim=4, model_name="dummy")
    capsule_outputs = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]],
            [[-1.0, -1.0, -1.0, -1.0], [2.0, 2.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        ]
    )

    lengths = net.get_capsule_lengths(capsule_outputs)
    expected = torch.linalg.norm(capsule_outputs, dim=-1)

    assert torch.allclose(lengths, expected)


def test_classifier_preprocess_and_predict(monkeypatch, dummy_roberta, dummy_tokenizer):
    def fake_capsule_forward(self, x):
        return torch.zeros((x.size(0), self.num_class_caps, self.class_cap_dim), device=x.device)

    monkeypatch.setattr(model.CapsuleLayer, "forward", fake_capsule_forward, raising=False)

    classifier = model.MentalRoBERTaCapsClassifier(num_classes=3, model_name="dummy")

    cleaned = classifier.preprocess("Check http://example.com @user <b>hi!</b>")
    assert "http" not in cleaned and "@" not in cleaned and "<b>" not in cleaned

    prediction, probs = classifier.predict("Simple text for routing", return_probs=True)

    assert prediction in classifier.labels
    assert set(probs.keys()) == set(classifier.labels)
