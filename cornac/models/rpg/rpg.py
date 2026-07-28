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
"""Neural modules for RPG (Hou et al., KDD 2025).

``RPGBackbone`` is a GPT-2 decoder over items whose embedding is the mean of
their (unordered) OPQ semantic-ID token embeddings, with a multi-token-
    prediction (MTP) head per codebook that predicts every digit of the next
    items' semantic IDs *in parallel* at all supervised sequence positions via
    temperature-scaled cosine similarity to the (shared) token embedding table.
    ``build_adjacency`` and ``graph_decode``
implement the similarity-graph-guided beam decoding used at inference so that
scoring cost does not depend on scoring every catalog item exhaustively.

Faithful to the official implementation
https://github.com/facebookresearch/RPG_KDD2025 (``genrec/models/RPG/model.py``):
the ``ResBlock`` heads, the ``wte(tokens).mean(dim=-2)`` item embedding, the
``F.normalize`` + cosine/temperature MTP loss, ``build_ii_sim_mat`` (mean over
digits of per-digit code cosine similarity) and ``graph_propagation``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class ResBlock(nn.Module):
    """Residual SiLU block used as an MTP prediction head (zero-init linear so
    each head starts as the identity)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class RPGBackbone(nn.Module):
    """GPT-2 backbone with one MTP head per codebook (RPG paper, Sec. 3).

    A single token-embedding table (``gpt2.wte``) holds all digits' codes with
    per-digit offsets; index 0 is padding and the final index is an unused
    ``eos`` slot so ``wte.weight[1:-1]`` is exactly the ``n_codebook *
    codebook_size`` code embeddings (laid out digit-by-digit). An item's input
    embedding is the mean of its ``n_codebook`` token embeddings; each
    supervised position's hidden state feeds ``n_codebook`` residual heads,
    each scored against its codebook by temperature-scaled cosine similarity.
    """

    def __init__(
        self,
        n_codebook,
        codebook_size,
        max_len,
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
    ):
        super().__init__()
        self.n_codebook = int(n_codebook)
        self.codebook_size = int(codebook_size)
        self.temperature = temperature
        # token 0 = pad; digit d codes occupy [1 + d*K, 1 + (d+1)*K); last = unused eos
        vocab_size = self.n_codebook * self.codebook_size + 2
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation,
            resid_pdrop=resid_dropout,
            embd_pdrop=embd_dropout,
            attn_pdrop=attn_dropout,
            layer_norm_epsilon=layer_norm_eps,
            initializer_range=initializer_range,
            use_cache=False,
        )
        self.gpt2 = GPT2Model(cfg)
        self.pred_heads = nn.ModuleList([ResBlock(d_model) for _ in range(self.n_codebook)])

        offsets = self.codebook_size * torch.arange(self.n_codebook, dtype=torch.long) + 1
        self.register_buffer("offsets", offsets)
        # item -> offset tokens; row per item plus a trailing all-pad (0) row for
        # the history pad index. Filled by ``set_item_tokens`` after tokenizing.
        self.register_buffer(
            "item_id2tokens", torch.zeros(1, self.n_codebook, dtype=torch.long)
        )

    def set_item_tokens(self, sid_table):
        """Register the per-item OPQ codes as offset tokens (plus a pad row).

        ``sid_table``: (n_items, n_codebook) un-offset codes in ``[0, K)``.
        """
        codes = torch.as_tensor(np.asarray(sid_table), dtype=torch.long)
        tokens = codes + self.offsets.cpu()
        pad_row = torch.zeros(1, self.n_codebook, dtype=torch.long)
        table = torch.cat([tokens, pad_row], dim=0).to(self.offsets.device)
        self.item_id2tokens = table

    def _hidden_states(self, input_ids, attention_mask):
        """GPT-2 hidden states with real-item positions starting at zero."""
        input_tokens = self.item_id2tokens[input_ids]  # (B, L, n_codebook)
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)  # (B, L, d)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp_min(0)
        out = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out.last_hidden_state

    def _last_hidden(self, input_ids, attention_mask):
        """GPT-2 hidden state at the most-recent real position for each row."""
        hidden = self._hidden_states(input_ids, attention_mask)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        last_positions = (attention_mask.long() * positions).max(dim=-1).values
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden[rows, last_positions]

    def _mtp_logits(self, hidden):
        """Per-digit code logits from ``(..., d)`` hidden states.

        Returns ``(..., n_codebook, codebook_size)``: temperature-scaled cosine
        similarity between each head's normalized output and the normalized
        codebook token embeddings.
        """
        states = torch.stack(
            [self.pred_heads[i](hidden) for i in range(self.n_codebook)], dim=-2
        )
        states = F.normalize(states, dim=-1)
        token_emb = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
        token_emb = token_emb.view(self.n_codebook, self.codebook_size, -1)
        logits = (
            torch.einsum("...id,ikd->...ik", states, token_emb) / self.temperature
        )
        return logits

    def forward(self, input_ids, attention_mask, target_iids):
        """Official masked all-position MTP loss.

        ``target_iids`` has shape ``(B, L)``. Valid entries are next-item IDs;
        ``-100`` entries are ignored. Cross-entropy is averaged over all valid
        positions for each codebook and then averaged across codebooks.
        """
        hidden = self._hidden_states(input_ids, attention_mask)
        label_mask = target_iids != -100
        logits = self._mtp_logits(hidden[label_mask])
        target_codes = self.item_id2tokens[target_iids[label_mask]] - self.offsets
        losses = [
            F.cross_entropy(logits[:, i], target_codes[:, i])
            for i in range(self.n_codebook)
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def next_item_logits(self, input_ids, attention_mask):
        """Per-digit next-item code logits, shape (B, n_codebook, codebook_size)."""
        return self._mtp_logits(self._last_hidden(input_ids, attention_mask))

    @torch.no_grad()
    def build_adjacency(self, n_items, edges, chunk_size=1024):
        """kNN item-item similarity graph over the OPQ codes (``build_ii_sim_mat``).

        sim(i, j) = mean over digits d of the [0, 1]-rescaled cosine similarity
        between the digit-d code embeddings of items i and j. Returns the top
        ``edges`` neighbours per item as an int64 numpy array (n_items, k),
        ``k = min(edges, n_items)``. As in the reference implementation, an
        item's own node can be one of its neighbours.
        """
        device = self.offsets.device
        token_emb = F.normalize(
            self.gpt2.wte.weight[1:-1].view(self.n_codebook, self.codebook_size, -1),
            dim=-1,
        )
        token_sims = torch.bmm(token_emb, token_emb.transpose(1, 2))
        token_sims = 0.5 * (token_sims + 1.0)  # (n_codebook, K, K) in [0, 1]

        codes = self.item_id2tokens[:n_items] - self.offsets  # (n_items, n_codebook)
        k = min(int(edges), n_items)
        adj = torch.empty(n_items, k, dtype=torch.long, device=device)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            ci = codes[start:end]  # (b, n_codebook)
            sim = torch.zeros(end - start, n_items, device=device)
            for d in range(self.n_codebook):
                sim += token_sims[d][ci[:, d]][:, codes[:, d]]
            sim /= self.n_codebook
            adj[start:end] = sim.topk(k, dim=1).indices
        return adj.cpu().numpy()


def graph_decode(token_logits_flat, adjacency, item_tokens, num_beams, prop_steps, rng):
    """Similarity-graph-guided beam decoding (``graph_propagation``).

    Starting from ``num_beams`` random items, repeatedly expand to the graph
    neighbours of the current beam, score each candidate by the mean of its
    ``n_codebook`` next-item token logits, and keep the top ``num_beams`` for
    the next step. The final beam is returned after propagation.

    Parameters
    ----------
    token_logits_flat : numpy.ndarray, shape (n_codebook * codebook_size,)
        Flattened per-digit next-item logits (digit-by-digit).
    adjacency : numpy.ndarray, shape (n_items, edges)
        Precomputed kNN neighbour indices.
    item_tokens : numpy.ndarray, shape (n_items, n_codebook)
        Per-item *offset* tokens (so ``item_tokens - 1`` indexes
        ``token_logits_flat``).
    num_beams, prop_steps : int
    rng : numpy.random.RandomState

    Returns
    -------
    (beam, scores) : two aligned numpy arrays over the final beam item indices.
    """
    n_items = adjacency.shape[0]

    def score_nodes(nodes):
        return token_logits_flat[item_tokens[nodes] - 1].mean(axis=1)

    beam = rng.randint(0, n_items, size=min(num_beams, n_items))
    for _ in range(prop_steps):
        neighbors = np.unique(adjacency[beam].reshape(-1))
        nb_scores = score_nodes(neighbors)
        top = np.argsort(-nb_scores)[:num_beams]
        beam = neighbors[top]
    return beam.astype("int64", copy=False), score_nodes(beam)
