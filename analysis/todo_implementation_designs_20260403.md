# Implementation Designs for TODO Items

**Generated**: 2026-04-03
**Status**: Design Phase

---

## 1. Iterative Run (Lottery Ticket Style)

### 1.1 Core Concept

Iteratively prune features and retrain, similar to Lottery Ticket Hypothesis:

```
Round 1: Train with all m features → get importance scores
Round 2: Remove bottom-k% features, retrain with remaining features
Round 3+: Repeat until reaching target feature count
```

**Key Question**: Reset weights each round or inherit?

---

### 1.2 Variant A: Hard Iterative Pruning (Classic Lottery Ticket)

**Pseudocode**:

```python
def iterative_hard_pruning(
    X_train, y_train, X_test, y_test,
    n_features_target: int,
    prune_ratio: float = 0.2,
    n_rounds: int = 5,
    rewind_to_init: bool = True,
):
    """
    Classic Lottery Ticket style iterative pruning.

    Args:
        n_features_target: Stop when features <= this number
        prune_ratio: Fraction of features to remove each round
        rewind_to_init: If True, reset weights to initialization each round
    """
    current_features = list(range(X_train.shape[1]))
    best_masks = []

    # Save initial weights for rewinding
    if rewind_to_init:
        init_weights = copy.deepcopy(model.state_dict())

    for round_idx in range(n_rounds):
        # Train model
        model = create_model(n_features=len(current_features))
        if rewind_to_init:
            model.load_state_dict(init_weights)  # Lottery Ticket rewind

        train(model, X_train[:, current_features], y_train)

        # Get feature importance
        scores = model.get_feature_scores()
        scores = np.abs(scores.detach().cpu().numpy())

        # Prune bottom-k%
        n_to_prune = int(len(current_features) * prune_ratio)
        keep_indices = np.argsort(scores)[n_to_prune:]
        prune_indices = np.argsort(scores)[:n_to_prune]

        # Record mask
        mask = np.zeros(len(current_features))
        mask[keep_indices] = 1
        best_masks.append(mask)

        # Update feature set
        current_features = [current_features[i] for i in keep_indices]

        print(f"Round {round_idx}: {len(current_features)} features remaining")

        if len(current_features) <= n_features_target:
            break

    return current_features, best_masks
```

**Hyperparameters**:
- `prune_ratio`: 0.1, 0.2, 0.3
- `rewind_to_init`: True (Lottery Ticket) / False (weight inheritance)
- `n_rounds`: typically 5-10

---

### 1.3 Variant B: Soft Iterative with Knowledge Distillation

**Pseudocode**:

```python
def iterative_with_distillation(
    X_train, y_train,
    n_features_target: int,
    prune_ratio: float = 0.2,
    temperature: float = 4.0,
    alpha: float = 0.5,  # balance task loss vs distillation loss
):
    """
    Iterative pruning with teacher-student distillation.
    Student learns from both ground truth and teacher predictions.
    """
    current_features = list(range(X_train.shape[1]))

    # Train teacher model with all features
    teacher = create_model(n_features=len(current_features))
    teacher = train(teacher, X_train[:, current_features], y_train)
    teacher.eval()

    for round_idx in range(10):
        # Create student with subset of features
        student = create_model(n_features=len(current_features))

        # Distillation loss
        for epoch in range(epochs):
            # Task loss
            logits = student(X_train[:, current_features])
            task_loss = F.cross_entropy(logits, y_train)

            # Distillation loss
            with torch.no_grad():
                teacher_logits = teacher(X_train[:, current_features])
            distill_loss = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            loss = alpha * task_loss + (1 - alpha) * distill_loss
            loss.backward()
            optimizer.step()

        # Prune
        scores = student.get_feature_scores()
        n_to_prune = int(len(current_features) * prune_ratio)
        keep_indices = np.argsort(scores)[n_to_prune:]
        current_features = [current_features[i] for i in keep_indices]

        # Teacher = best student so far
        teacher.load_state_dict(student.state_dict())

        if len(current_features) <= n_features_target:
            break

    return current_features
```

**Hyperparameters**:
- `temperature`: 2.0, 4.0, 6.0 (softmax temperature)
- `alpha`: 0.3, 0.5, 0.7 (task vs distillation balance)

---

### 1.4 Variant C: Gradual ADMM Tightening

**Pseudocode**:

```python
def gradual_admm_tightening(
    model, X_train, y_train,
    initial_rho: float = 0.01,
    final_rho: float = 1.0,
    n_phases: int = 5,
    target_sparsity: float = 0.8,
):
    """
    Gradually increase ADMM penalty to induce sparsity.
    No explicit feature removal - features naturally become zero.
    """
    rho_schedule = np.linspace(initial_rho, final_rho, n_phases)

    for phase_idx, rho in enumerate(rho_schedule):
        print(f"Phase {phase_idx}: rho={rho:.4f}")

        # ADMM training with current rho
        for epoch in range(epochs_per_phase):
            # Augmented Lagrangian
            loss = task_loss + (rho / 2) * torch.norm(gate - z + u) ** 2

            # Proximal update (encourages sparsity)
            z = proximal_ratio_norm(gate, lambda_=lambda_schedule[phase_idx])

            # Dual update
            u = u + gate - z

        # Count alive features
        alive = (torch.abs(gate) > 1e-4).sum().item()
        print(f"  Alive features: {alive}")

    # Final hard thresholding
    final_mask = torch.abs(gate) > 1e-4
    return final_mask
```

**Key**: Integrates with existing ADMM framework, minimal code changes.

---

### 1.5 Variant D: Iterative Gate Refinement

**Pseudocode**:

```python
def iterative_gate_refinement(
    model, X_train, y_train,
    threshold: float = 0.01,
    n_iterations: int = 10,
    warmup_epochs: int = 50,
    admm_epochs: int = 100,
):
    """
    Let gate converge, then hard-remove small features and continue.
    """
    alive_features = list(range(model.input_size))

    for iteration in range(n_iterations):
        # Train with current features
        model = train_admm(
            model, X_train[:, alive_features], y_train,
            warmup_epochs=warmup_epochs,
            admm_epochs=admm_epochs,
        )

        # Get gate values
        gate = model.get_gate_values()
        gate_abs = torch.abs(gate).detach().cpu().numpy()

        # Find features to remove
        to_remove = gate_abs < threshold
        to_keep = gate_abs >= threshold

        n_removed = to_remove.sum()
        print(f"Iteration {iteration}: removing {n_removed} features")

        if n_removed == 0:
            print("No features removed, stopping")
            break

        # Update alive features
        alive_features = [alive_features[i] for i in range(len(alive_features)) if to_keep[i]]

        # Reinitialize model with smaller input
        model = create_model(n_features=len(alive_features))

    return alive_features, gate
```

---

## 2. Polynomial Feature Expansion + Expanded-Space Selection

### 2.1 Expansion Design

#### Variant A: sklearn PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures

def polynomial_expansion(X, degree: int = 2, interaction_only: bool = False):
    """
    Expand features using polynomial combinations.

    degree=2: [x1, x2] -> [x1, x2, x1², x1*x2, x2²]
    degree=3: adds [x1³, x1²*x2, x1*x2², x2³]
    """
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=interaction_only,  # if True: no x², only x1*x2
    )
    X_expanded = poly.fit_transform(X)

    # Get feature names for interpretability
    feature_names = poly.get_feature_names_out()

    return X_expanded, feature_names, poly


# Example usage
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
X_exp, names, poly = polynomial_expansion(X, degree=2)

print(f"Original: {X.shape[1]} features")
print(f"Expanded: {X_exp.shape[1]} features")
print(f"Features: {names[:5]}...")  # ['x0', 'x1', 'x0²', 'x0 x1', 'x1²']
```

**Degree comparison**:

| Degree | Original m | Expanded m' | Example |
|--------|------------|-------------|---------|
| 1 | 10 | 10 | x1, x2, ... |
| 2 | 10 | 65 | + x1², x1*x2, x2², ... |
| 3 | 10 | 285 | + x1³, x1²*x2, ... |
| 2 | 100 | 5151 | Too many! |

**Recommendation**: degree=2 for m<50, else use learned expansion.

---

#### Variant B: Learned Expansion (Current Implementation)

Already implemented in `mentor_models.py:ExpandedFeatureSelectionMLP`:

```python
# Each feature learns an expansion vector
self.expansion_weight = nn.Parameter(torch.empty(input_size, expansion_dim))

# Forward: x[j] * W_exp[j] -> token of size expansion_dim
tokens = x.unsqueeze(-1) * self.expansion_weight.unsqueeze(0)
tokens = self.expansion_activation(tokens)  # CRITICAL: non-linearity
```

---

#### Variant C: Fourier Features

```python
def fourier_features(X, n_bands: int = 10, max_freq: float = 1.0):
    """
    Fourier feature encoding for continuous features.
    sin/cos encoding captures periodic patterns.
    """
    bands = torch.linspace(0, max_freq, n_bands)

    features = [X]
    for b in bands:
        features.append(torch.sin(2 * np.pi * b * X))
        features.append(torch.cos(2 * np.pi * b * X))

    return torch.cat(features, dim=-1)


# Example: 10 features with 10 bands -> 10 + 2*10*10 = 210 features
```

**Use case**: When features have periodic/seasonal patterns.

---

### 2.2 Selection Design

#### Variant A: Original-Feature Selection (Current)

```python
# One gate per original feature
# All expanded features from same original share the gate

score_j = |g_j|  # gate for original feature j
rank by score_j
select top-k original features (and their expanded forms)
```

**Pro**: Simple, interpretable
**Con**: Cannot select partial expansions

---

#### Variant B: Expanded-Space Selection

```python
# Each expanded feature has independent gate

def expanded_space_selection(model, X_expanded, k):
    """
    Select top-k from expanded features directly.
    """
    # One gate per expanded feature
    gate = model.gate  # shape: [m_expanded]
    scores = torch.abs(gate)

    # Select top-k expanded features
    top_k_indices = torch.topk(scores, k).indices

    return top_k_indices


# Issue: might select x1² but not x1
# Solution: post-hoc grouping or constraints
```

**Issue**: Selected set may be inconsistent (e.g., x1² selected but x1 not).

---

#### Variant C: Group Lasso on Expanded Features

```python
def group_lasso_expansion(model, X, expansion_groups):
    """
    Group sparsity: all expanded features from same original feature
    are selected or dropped together.

    expansion_groups: list of lists
        [[0, 1, 2], [3, 4], ...]  # indices of expanded features per original
    """
    penalty = 0.0

    for group in expansion_groups:
        # L2 norm of gate values in this group
        group_gates = model.gate[group]
        penalty += torch.norm(group_gates, p=2)

    return penalty


# Training
loss = task_loss + lambda_ * group_lasso_expansion(model, X, groups)
```

**Pro**: Feature integrity preserved
**Con**: Less fine-grained control

---

#### Variant D: Hierarchical Selection

```python
def hierarchical_selection(model, X, k_original, k_expanded_per_feature):
    """
    Two-level selection:
    Level 1: Select k_original original features
    Level 2: Within each selected, select k_expanded expanded features
    """
    # Level 1: Original feature gates
    original_scores = model.original_gate  # shape: [m]
    top_originals = torch.topk(original_scores, k_original).indices

    # Level 2: Expanded gates within selected features
    selected_expanded = []
    for j in top_originals:
        expanded_indices = model.expansion_indices[j]  # indices for feature j
        expanded_scores = model.expanded_gate[expanded_indices]
        top_expanded = torch.topk(expanded_scores, k_expanded_per_feature).indices
        selected_expanded.extend(expanded_indices[top_expanded])

    return selected_expanded


# Example: m=100, expansion=4, k_original=20, k_expanded=2
# Total selected: 20 * 2 = 40 expanded features
```

**Pro**: Flexible, interpretable hierarchy
**Con**: More hyperparameters

---

### 2.3 Combined Implementation

```python
class PolynomialFeatureSelection(nn.Module):
    """
    Polynomial expansion + Group-based selection.
    """
    def __init__(
        self,
        input_size: int,
        n_classes: int,
        degree: int = 2,
        hidden_dims: list = [32, 32],
    ):
        super().__init__()
        self.degree = degree
        self.input_size = input_size

        # Expansion
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)

        # Compute expanded size
        self.expanded_size = self._compute_expanded_size(input_size, degree)

        # Gates: one per expanded feature
        self.gate = nn.Parameter(torch.zeros(self.expanded_size))

        # Predictor
        layers = []
        in_dim = self.expanded_size
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Mish(), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.predictor = nn.Sequential(*layers)

        # Group indices (expanded -> original mapping)
        self.group_indices = self._build_groups()

    def _compute_expanded_size(self, m, degree):
        # sklearn formula: (m + d)! / (m! * d!) - 1
        from math import comb
        total = sum(comb(m + d, d) for d in range(1, degree + 1))
        return int(total)

    def _build_groups(self):
        """Map expanded indices to original feature groups."""
        # This requires tracking which expanded feature comes from which original
        # Implementation depends on sklearn's get_feature_names_out parsing
        pass

    def forward(self, x):
        # Expand
        x_np = x.detach().cpu().numpy()
        x_exp = self.poly.transform(x_np)
        x_exp = torch.tensor(x_exp, dtype=x.dtype, device=x.device)

        # Apply gates
        g = self.gate_values()
        x_gated = x_exp * g

        # Predict
        return self.predictor(x_gated)

    def gate_values(self):
        return torch.sigmoid(self.gate) if self.bounded_gate else self.gate

    def get_feature_scores(self):
        """Return scores grouped by original feature."""
        scores = torch.abs(self.gate)
        group_scores = {}
        for orig_idx, exp_indices in self.group_indices.items():
            group_scores[orig_idx] = scores[exp_indices].mean()
        return group_scores
```

---

## 3. Transformer Pretrain (Backbone Tuning)

### 3.1 Problem

Current transformer backbone fails on synthetic FS benchmarks:
- best-k ≈ 0
- AUC ≈ 0.5 (random)

**TODO**: Pretrain backbone with proper learning rate, without ADMM/gating first.

---

### 3.2 Pretrain Design

#### Stage 1: Masked Feature Reconstruction (MAE-style)

```python
class MaskedFeaturePretraining(nn.Module):
    """
    Pretrain transformer by reconstructing masked features.
    Similar to MAE but for tabular features.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 1,
        mask_ratio: float = 0.3,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Encoder (same as GatedTokenTransformerFS)
        self.feature_embedding = nn.Parameter(torch.empty(input_size, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, input_size + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction head
        self.decoder = nn.Linear(d_model, 1)  # reconstruct raw feature value

    def forward(self, x):
        """
        Mask some features, encode, then reconstruct.
        """
        batch_size, n_features = x.shape

        # Random mask
        mask = torch.rand(batch_size, n_features) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0  # or use learned mask token

        # Create tokens
        tokens = x_masked.unsqueeze(-1) * self.feature_embedding.unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.position_embedding

        # Encode
        encoded = self.encoder(seq)
        feature_tokens = encoded[:, 1:, :]  # exclude CLS

        # Reconstruct
        reconstructed = self.decoder(feature_tokens).squeeze(-1)

        # Loss: only on masked positions
        loss = F.mse_loss(reconstructed[mask], x[mask])

        return reconstructed, loss


def pretrain_transformer(
    X_train,
    d_model: int = 16,
    n_heads: int = 4,
    n_layers: int = 1,
    mask_ratio: float = 0.3,
    lr: float = 1e-5,  # TODO: start with small lr
    epochs: int = 100,
):
    """
    Pretrain transformer backbone with masked reconstruction.
    """
    model = MaskedFeaturePretraining(
        input_size=X_train.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mask_ratio=mask_ratio,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        reconstructed, loss = model(X_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: reconstruction_loss = {loss.item():.4f}")

    return model
```

---

#### Stage 2: Supervised Fine-tuning with ADMM Gate

```python
def finetune_with_admm(
    pretrained_encoder,
    X_train, y_train,
    n_classes: int = 2,
    lr: float = 1e-4,
    warmup_epochs: int = 50,
    admm_epochs: int = 150,
    feat_drop: float = 0.6,
):
    """
    Fine-tune pretrained encoder with ADMM gate for feature selection.
    """
    # Build full model
    model = GatedTokenTransformerFS(
        input_size=X_train.shape[1],
        n_classes=n_classes,
        d_model=pretrained_encoder.d_model,
        feat_drop=feat_drop,
    )

    # Load pretrained weights (except gate)
    model.feature_embedding.data = pretrained_encoder.feature_embedding.data
    model.encoder.load_state_dict(pretrained_encoder.encoder.state_dict())

    # Add ADMM gate (new parameters, not pretrained)
    # ... (use existing ADMM training loop)

    return model
```

---

### 3.3 Hyperparameter Search

```python
# Learning rate sweep
lr_candidates = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4]

# Mask ratio sweep
mask_ratio_candidates = [0.1, 0.3, 0.5, 0.7]

# Architecture sweep
d_model_candidates = [8, 16, 32]
n_layers_candidates = [1, 2, 3]
n_heads_candidates = [2, 4, 8]
```

**Recommendation**: Start with lr=1e-5, mask_ratio=0.3, d_model=16.

---

### 3.4 Expected Outcomes

| Scenario | Expected Result |
|----------|-----------------|
| Pretrain works (loss ↓) | Backbone learns feature structure |
| Fine-tune improves best-k | Transformer can do FS with proper pretrain |
| Fine-tune still fails | Transformer fundamentally unsuited for this task |

---

## Implementation Priority

1. **Iterative Run Variant C** (Gradual ADMM Tightening) - easiest, integrates with existing code
2. **Polynomial Expansion + Group Selection** - most impactful for method
3. **Transformer Pretrain** - clarify if transformer can be salvaged

---

## Next Steps

1. Implement `GradualADMMTightening` wrapper
2. Implement `PolynomialFeatureSelection` class
3. Run transformer pretrain experiments