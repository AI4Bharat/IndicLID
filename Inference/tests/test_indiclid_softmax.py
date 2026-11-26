import importlib.util
import math
import pathlib
import sys
import types
from unittest.mock import MagicMock


class _FakeTensor:
    def __init__(self, data):
        self.data = data

    def to(self, _device):
        return self

    def item(self):
        return self.data


class _BatchEncoding(dict):
    def to(self, _device):
        return self


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _softmax(logits, dim=1):
    assert dim == 1
    out = []
    for row in logits.data:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return _FakeTensor(out)


def _max(tensor, dim):
    assert dim == 1
    scores = []
    indices = []
    for row in tensor.data:
        best_idx = max(range(len(row)), key=lambda i: row[i])
        scores.append(_FakeTensor(row[best_idx]))
        indices.append(_FakeTensor(best_idx))
    return scores, indices


def _load_indiclid_module():
    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = _NoGrad
    torch_mod.max = _max
    torch_mod.nn = types.ModuleType("torch.nn")
    torch_mod.nn.functional = types.ModuleType("torch.nn.functional")
    torch_mod.nn.functional.softmax = _softmax
    torch_mod.nn.Module = object
    torch_mod.utils = types.ModuleType("torch.utils")
    torch_mod.utils.data = types.ModuleType("torch.utils.data")
    torch_mod.utils.data.Dataset = object
    torch_mod.utils.data.DataLoader = MagicMock()
    torch_mod.optim = MagicMock()

    stubs = {
        "tqdm": MagicMock(),
        "fasttext": MagicMock(),
        "torch": torch_mod,
        "torch.nn": torch_mod.nn,
        "torch.nn.functional": torch_mod.nn.functional,
        "torch.utils": torch_mod.utils,
        "torch.utils.data": torch_mod.utils.data,
        "torch.optim": torch_mod.optim,
        "transformers": MagicMock(),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    try:
        sys.modules.update(stubs)
        module_path = (
            pathlib.Path(__file__).resolve().parents[1] / "ai4bharat" / "IndicLID.py"
        )
        spec = importlib.util.spec_from_file_location("indiclid_mod", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for k, prev in saved.items():
            if prev is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = prev


def test_indiclid_bert_returns_softmax_probabilities():
    mod = _load_indiclid_module()
    lid = mod.IndicLID.__new__(mod.IndicLID)
    lid.device = "cpu"
    lid.IndicLID_lang_code_dict_reverse = {0: "hin_Latn", 1: "tam_Latn", 2: "eng_Latn"}

    logits = _FakeTensor([[6.42, 1.0, 0.0], [0.5, 2.0, 0.5]])
    lid.get_dataloaders = lambda indices, texts, bs: [
        (list(indices), list(texts)),
    ]
    lid.IndicLID_BERT_tokenizer = lambda inputs, **kwargs: _BatchEncoding(
        {
            "input_ids": _FakeTensor(inputs),
            "token_type_ids": _FakeTensor(inputs),
            "attention_mask": _FakeTensor(inputs),
        }
    )
    lid.IndicLID_BERT = lambda *args, **kwargs: types.SimpleNamespace(logits=logits)

    output = lid.IndicBERT_roman_inference([(0, "namaste"), (1, "vanakkam")], {}, 2)

    assert output[0][1] == "hin_Latn"
    assert output[1][1] == "tam_Latn"
    assert output[0][3] == "IndicLID-BERT"

    expected_p0 = math.exp(6.42) / (math.exp(6.42) + math.exp(1.0) + math.exp(0.0))
    expected_p1 = math.exp(2.0) / (math.exp(0.5) + math.exp(2.0) + math.exp(0.5))
    assert 0.0 <= output[0][2] <= 1.0
    assert 0.0 <= output[1][2] <= 1.0
    assert math.isclose(output[0][2], expected_p0, rel_tol=1e-9)
    assert math.isclose(output[1][2], expected_p1, rel_tol=1e-9)
