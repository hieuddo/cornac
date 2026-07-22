# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Official RPG recipe (Amazon 2014) as an importable config.

Transcribed from the official code
https://github.com/facebookresearch/RPG_KDD2025 -- backbone/tokenizer/decoding
from ``genrec/models/RPG/config.yaml`` and training hyperparameters from
``genrec/default.yaml``:

* embeddings ``text-embedding-3-large`` (3072-d) -> whitened PCA to 512;
* OPQ tokenizer ``OPQ32,IVF1,PQ32x8`` (32 codebooks x 256 codes);
* GPT-2 backbone: seq len 50, d_model 448, 2 layers, 4 heads, inner 1024,
  ``gelu_new``, dropout embd 0.5 / attn 0.5 / resid 0.0, layer-norm eps 1e-12;
* MTP temperature 0.07;
* graph decoding: beam 50, edges 50, propagation 3, chunk 1024;
* AdamW lr 3e-4, weight decay 0.0, batch 256, cosine schedule with 10k warmup,
  grad-norm clip 1.0.

Usage::

    from cornac.models.rpg import RPG_CONFIG, RPG

    model = RPG(**{**RPG_CONFIG, "seed": 2024})

The official trainer runs up to 150 epochs, evaluates every epoch in batches
of 32, and early-stops after 20 non-improving ``ndcg@10`` epochs. These settings
are reproduced below. Per-dataset paper-best settings (from the README's
reproduce commands) ship as :data:`RPG_BEAUTY_CONFIG` /
:data:`RPG_SPORTS_CONFIG` / :data:`RPG_TOYS_CONFIG`; ``n_codebook=64`` is the
CDs_and_Vinyl long-semantic-ID setting.
"""

RPG_CONFIG = dict(
    pca_dim=512,
    n_codebook=32,
    codebook_size=256,
    max_len=50,
    d_model=448,
    n_layer=2,
    n_head=4,
    n_inner=1024,
    activation="gelu_new",
    resid_dropout=0.0,
    embd_dropout=0.5,
    attn_dropout=0.5,
    layer_norm_eps=1e-12,
    initializer_range=0.02,
    temperature=0.07,
    scoring="graph",
    n_beams=50,
    graph_edges=50,
    propagation_steps=3,
    graph_chunk_size=1024,
    learning_rate=3e-4,
    weight_decay=0.0,
    batch_size=256,
    n_epochs=150,
    lr_schedule="cosine",
    warmup_steps=10000,
    max_grad_norm=1.0,
    model_selection="best",
    val_metric="ndcg",
    val_k=10,
    val_eval_every=1,
    val_batch_size=32,
    early_stopping_patience=20,
    val_sample=None,
)

# Per-dataset reproduce settings, verbatim from the official README's
# reproduce commands (the repo's config.yaml/default.yaml above are its
# *defaults*, not the paper-best values). lr, temperature and n_codebook
# change training; n_beams/graph_edges/propagation_steps only the graph
# decode. Note Sports/Toys tune *shorter* IDs (16 codebooks); 64 is the
# CDs_and_Vinyl setting.

RPG_BEAUTY_CONFIG = dict(
    RPG_CONFIG,
    learning_rate=0.01,
    temperature=0.03,
    n_codebook=32,
    n_beams=20,
    graph_edges=200,
    propagation_steps=3,
)

RPG_SPORTS_CONFIG = dict(
    RPG_CONFIG,
    learning_rate=0.003,
    temperature=0.03,
    n_codebook=16,
    n_beams=100,
    graph_edges=30,
    propagation_steps=5,
)

RPG_TOYS_CONFIG = dict(
    RPG_CONFIG,
    learning_rate=0.003,
    temperature=0.03,
    n_codebook=16,
    n_beams=200,
    graph_edges=20,
    propagation_steps=3,
)
