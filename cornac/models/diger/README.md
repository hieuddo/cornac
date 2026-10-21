# DIGER

`cornac.models.DIGER` provides T5 next-item generation with a residual-quantizer
(RQ) tokenizer, frequency-gated hard-forward/soft-backward assignments, and joint
recommendation/codebook updates.
PyTorch and Transformers load only when fitting or predicting.

Install `pip install -r cornac/models/diger/requirements.txt` and see
[the Beauty example](../../../examples/diger_example.py). Features are supplied
through Cornac's usual interface:

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import DIGER
from cornac.models.diger import BEAUTY_ST5_CONFIG

evaluation = NextItemEvaluation.from_splits(
    train_data=train_data, val_data=val_data, test_data=test_data,
    fmt="USIT", item_feature=FeatureModality(features=embeddings, ids=item_ids),
)
model = DIGER(**BEAUTY_ST5_CONFIG, device="auto")
model.fit(evaluation.train_set, evaluation.val_set)
scores = model.score(user_idx=0, history_items=[0, 1])  # internal Cornac indices
path = model.save("saved_models")
restored = DIGER.load(path)
```

Features must cover every admitted item. `FeatureModality` aligns raw IDs to
Cornac indices; DIGER checks alignment, shape and finite values before fitting.
No further feature scaling is applied. Supply dense semantic embeddings, not
item IDs or interaction vectors. Training expands every session into all
history/next-item prefixes; validation uses one final target per session.
Singleton and explicitly catalog-only sessions are skipped. Validation is
required and test sessions are never passed to the training controller.
`max_len` caps history items consistently during fitting and inference.

With no `initializer`, fit first trains an RQ autoencoder matched to the feature
width: 20 reconstruction warmup epochs, residual Lloyd initialization (20
iterations per level), and 100 epochs of variance-scaled nearest-code
straight-through feature reconstruction plus VQ/commitment MSE. This is a
declared adaptation; it is not the provenance of DIGER's released 4096-wide
initializer. Features and the pretrained encoder are frozen during joint fit.
The reconstruction decoder is discarded. Use `initializer_options` for a small
smoke test or `initializer` to supply a bundle from
`cornac.models.diger.training.pretrain_tokenizer`; bundles strictly check the
ordered padded feature hash, semantic width and tokenizer recipe.

Joint training uses simultaneous dual AdamW updates, separate gradient clipping
at 1 and cosine schedules over the full declared horizon. Balanced full-catalog
Sinkhorn assignments refresh after each epoch. Strict validation NDCG@10
improvement selects the generator/tokenizer/catalog together; ties retain the
earlier epoch. Unconstrained positional beam search returns full addresses,
including a deterministic collision suffix. Invalid addresses score no items.
Choose `n_beams` at least as large as the largest evaluation cutoff.

Cornac `save` stores tensors on CPU and restores the live model's device.
Inference falls back to CPU when a saved CUDA device is unavailable.

## Beauty tuning evidence

`BEAUTY_ST5_CONFIG` and `BEAUTY_OPENAI_CONFIG` expose the validation-selected
recipes from the retained Beauty study. Both selected generator LR **0.0005** and
codebook LR **0.00003** from six candidates. Defaults use the same tokenizer/T5
architectures, 120-epoch horizon, batch 256, evaluation batch 32, warmup 500,
weight decay 0.05, patience 15, maximum history 210 and beam 20. Winning discovery
validation NDCG@10 was **0.0489174** (ST5) and **0.0542642** (OpenAI). Test labels
were not used to choose these rates.

### Beauty: DIGER versus TIGER

Retained study results are mean ± sample SD over seeds 2025/2026/2027,
on 12,068 items and 22,363 aligned test queries. Metrics are fractions,
not percentages. Paper rows quote the main comparison in
[Fu et al., DIGER, Table 3 (arXiv v3)](https://arxiv.org/pdf/2601.19711v3#page=7);
the paper's main DIGER configuration uses frequency-based uncertainty decay
(FrqUD). An em dash means the paper does not report that cutoff.

| Source / item features | Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | --- | --- |
| Retained ST5, 768 dimensions | TIGER | 0.062141 ± 0.001020 | 0.033266 ± 0.000622 | 0.096350 ± 0.001617 | 0.041854 ± 0.000719 |
| Retained ST5, 768 dimensions | DIGER | 0.062767 ± 0.000921 | 0.034244 ± 0.000162 | 0.093294 ± 0.003556 | 0.041948 ± 0.000816 |
| Retained OpenAI, 3072 dimensions | TIGER | 0.066494 ± 0.002351 | 0.034982 ± 0.001217 | 0.101745 ± 0.002747 | 0.043828 ± 0.001167 |
| Retained OpenAI, 3072 dimensions | DIGER | 0.069087 ± 0.001804 | 0.036960 ± 0.000707 | 0.105338 ± 0.001701 | 0.046061 ± 0.000378 |
| DIGER paper, Table 3 | TIGER | 0.0610 | 0.0331 | — | — |
| DIGER paper, Table 3 | DIGER (FrqUD) | 0.0683 | 0.0372 | — | — |

The retained models share the same feature files and aligned test queries,
but retain their own training and decoding recipes. TIGER uses a fixed
100-epoch PAISCHER-derived preset, validation every five epochs on 2,000
queries, beam 50, and a freshly trained tokenizer for each seed. DIGER uses
the validation-selected 120-epoch recipe above and beam 20. Each DIGER feature
condition reuses one seed-2024 pretrained initializer across the three
joint-training seeds; its spread is conditional on that initializer rather
than independent initializer variance. Full addresses were injective for
all 12 retained model/catalog pairs.

Paper rows provide context rather than a matched reproduction: Beauty is
called B-Shop in v3, and [Table 1](https://arxiv.org/pdf/2601.19711v3#page=5)
reports 12,101 items, compared with the retained study's 12,068. The paper's
content-embedding and training recipes also differ from the retained study.
The retained TIGER rows are this study's baseline, separate from the earlier
single-seed results in [TIGER's README](../tiger/README.md).

Local provenance: `runs/paper-r-beauty-deadline-2026-10-12/heldout-summary/summary.csv`
(`view=all`; SHA-256
`7fa58e4b1ff480c15fd3b42ec55ec78b06cccb2d9ef7cc24baf2d71fd9b8fc39`).
Its `validation.json` records exact query alignment and verified artifact hashes
for all 12 seed bundles. These study records and large assets are maintained
separately from the model package.

Retained DIGER results come from the prior T5/RQ study implementation, before this
unification. Training has not been rerun on the refactored Cornac implementation;
they are not a reproduction of the paper's released results. ST5-description
and OpenAI feature files were retained without rescaling; their regeneration and matched source texts were not independently
qualified. The example can generate ST5 features from Cornac metadata or load an
NPZ of precomputed embeddings, but neither the preset nor regenerated text
promises the retained numbers. Ordered catalogs and initializer hashes must
match to reuse a retained initializer.
