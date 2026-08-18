"""
Dual-head CNN decoder for the (3,3) heavy-hex surface code.

This is the one file you need to complete. Fill in every `TODO` block
below — the rest of the pipeline (dataset generation, training loop,
evaluation, MWPM baseline, hardware run) is already done and will run as
soon as you finish this file.

Input : (B, 2*num_cycles, 4, 5) uint8 syndrome tensor
        - 4x5 diamond embedding of the 8 ancillas
          (see dataset_generation/heavyhex33_stim.py, ANC_COORD)
        - channels alternate [Z-plane, X-plane] per cycle
          (default num_cycles=3 -> in_channels=6)
Output: qubit_logits   (B, 17) — per-qubit X-error head
                                 (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)  — logical Z flip head
                                 (official metric: head-LER)

Keep the interfaces exactly as they are — the training/eval/hardware
scripts call them as-is:
  * HeavyHexCNN(in_channels, num_qubits, ...) with
    forward(x) -> (qubit_logits, logical_logits)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).

Hints:
  * The grid is small (4x5): 3x3 convs with padding=1 and no pooling work
    well. BatchNorm + ReLU per conv is a good default.
  * Flatten -> a shared fully-connected layer -> two linear heads.
  * Use torch.nn.functional.binary_cross_entropy_with_logits (the heads
    output logits, not probabilities).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_QUBITS = 17
GRID_H, GRID_W = 4, 5


class HeavyHexCNN(nn.Module):
    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 conv_channels=64, fc_dim=256, dropout=0.1):
        super().__init__()
        
        # ============ TODO 1/3 & 2/3 — feature extractor + heads ============
        # Feature extractor: Conv2d(3x3, padding=1) blocks with BatchNorm + ReLU
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )
        
        # Shared FC layer
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * GRID_H * GRID_W, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Two heads
        self.head_qubit = nn.Linear(fc_dim, num_qubits)      # (B, 17)
        self.head_logical = nn.Linear(fc_dim, 1)              # (B, 1)

    def forward(self, x):
        x = x.float()
        # Feature extraction
        x = self.features(x)
        # Shared FC layers
        x = self.shared(x)
        # Two heads
        qubit_logits = self.head_qubit(x)
        logical_logits = self.head_logical(x)
        
        return qubit_logits, logical_logits


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). qubit_pos_weight (optional, shape (17,)) goes into the
    per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p).
    """
    # ============ TODO 3/3 — loss computation ============
    # Logical head loss (MAIN loss)
    loss_logical = F.binary_cross_entropy_with_logits(
        logical_logits.squeeze(-1), y_logical.float()
    )
    
    # Per-qubit head loss (AUXILIARY loss)
    loss_qubit = F.binary_cross_entropy_with_logits(
        qubit_logits, y_qubit.float(), pos_weight=qubit_pos_weight
    )
    
    # Total loss: logical-first
    total_loss = loss_logical + aux_weight * loss_qubit
    
    return total_loss, loss_logical, loss_qubit