# DIGER

Cornac implementation of **DIGER** (Differentiable Semantic ID for Generative
Recommendation, Fu et al., SIGIR 2026,
[arXiv:2601.19711](https://arxiv.org/abs/2601.19711)). A residual quantizer turns
item content embeddings into semantic IDs; a T5 generator predicts the next
item's ID. Frequency-gated Gumbel assignments let recommendation gradients
update the codebooks alongside the generator.

## Requirements

```bash
pip install -r cornac/models/diger/requirements.txt
```

PyTorch and Transformers load only when fitting or predicting. Item embeddings
must cover every item admitted by the evaluation splits.

## Usage

Supply embeddings through `FeatureModality` (see
[`examples/diger_example.py`](../../../examples/diger_example.py)):

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import DIGER
from cornac.models.diger import BEAUTY_ST5_CONFIG

eval_method = NextItemEvaluation.from_splits(
    train_data=train, val_data=val, test_data=test, mode="last",
    item_feature=FeatureModality(features=item_embeddings, ids=item_ids),
)
model = DIGER(**{**BEAUTY_ST5_CONFIG, "device": "auto", "seed": 2025})
```

`BEAUTY_ST5_CONFIG` and `BEAUTY_OPENAI_CONFIG` share the validation-selected
Beauty recipe: generator/codebook learning rates 0.0005/0.00003, batch size 256,
120 epochs, warmup 500 steps, weight decay 0.05, maximum history 210, and beam 20.
Validation NDCG@10 is evaluated every epoch in batches of 32; training stops after
15 non-improving epochs and restores the paired generator, tokenizer and catalog.

## Implementation notes

- Features are aligned to Cornac item indices without further scaling. Training
  uses every history/next-item prefix; validation requires each session's final
  target. Test sessions do not enter training or model selection.
- A fresh fit pretrains a feature-width-matched RQ autoencoder (20 warmup epochs,
  residual Lloyd initialization, 100 reconstruction/VQ epochs), then freezes its
  encoder. This initialization is an adaptation of the released recipe. A bundle
  from `training.pretrain_tokenizer` can be reused through `initializer` only with
  the same ordered features and tokenizer options.
- Joint training updates the codebooks and generator together; full-catalog
  Sinkhorn assignments refresh each epoch. A deterministic suffix resolves
  semantic-ID collisions. Unconstrained beam search scores valid full addresses;
  set `n_beams` at least as large as the largest evaluation cutoff.

## Results

Retained Beauty study results below are mean ± sample SD over joint-training
seeds 2025/2026/2027, using 12,068 items and 22,363 aligned test queries. Metrics
are fractions. Paper rows come from
[Table 3, arXiv v3](https://arxiv.org/pdf/2601.19711v3#page=7); FrqUD denotes
frequency-based uncertainty decay, and an em dash denotes an unreported cutoff.

| Source / item features | Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | --- | --- |
| Retained ST5, 768 dimensions | TIGER | 0.062141 ± 0.001020 | 0.033266 ± 0.000622 | 0.096350 ± 0.001617 | 0.041854 ± 0.000719 |
| Retained ST5, 768 dimensions | DIGER | 0.062767 ± 0.000921 | 0.034244 ± 0.000162 | 0.093294 ± 0.003556 | 0.041948 ± 0.000816 |
| Retained OpenAI, 3072 dimensions | TIGER | 0.066494 ± 0.002351 | 0.034982 ± 0.001217 | 0.101745 ± 0.002747 | 0.043828 ± 0.001167 |
| Retained OpenAI, 3072 dimensions | DIGER | 0.069087 ± 0.001804 | 0.036960 ± 0.000707 | 0.105338 ± 0.001701 | 0.046061 ± 0.000378 |
| DIGER paper, Table 3 | TIGER | 0.0610 | 0.0331 | — | — |
| DIGER paper, Table 3 | DIGER (FrqUD) | 0.0683 | 0.0372 | — | — |

These are prior-study results, before the unified Cornac implementation; training
has not been rerun with this wrapper. TIGER uses a fixed 100-epoch preset with
beam 50 and a fresh tokenizer per seed; DIGER uses the selected recipe above
with beam 20 and one seed-2024 initializer per feature condition. DIGER's SD
therefore measures joint-training variation conditional on that initializer.
Both models share the retained features and aligned queries, with distinct
training/decoding recipes. The paper uses a different catalog (12,101 items),
content embeddings and training recipe, so its rows provide context.

The presets do not include retained embeddings or checkpoints. The example can
load precomputed features or generate Sentence-T5 embeddings (requires
`sentence-transformers`); regenerated texts/features do not promise these scores.
