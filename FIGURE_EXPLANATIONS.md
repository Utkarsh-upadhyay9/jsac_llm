# Figure Explanations (Core Subset)

This document summarizes five core secrecy-oriented figures used in the reduced (non-ISAC) scope.

## 1. Convergence of Secrecy Rate (Hybrid vs LLM vs MLP)
Shows long-horizon training stability. All curves begin with similar exploratory volatility; Hybrid converges fastest and highest, followed by a narrowly trailing LLM, then MLP. Mild residual oscillations reflect adaptive fine-tuning rather than instability.

## 2. Secrecy Rate vs. RIS Elements
Demonstrates monotonic secrecy improvement with additional RIS elements (array gain and refined phase control). Diminishing returns appear after mid-scale growth: marginal secrecy gain per added element shrinks, showing practical deployment trade-offs.

## 3. Secrecy Rate vs. BS Transmit Power
Secrecy rate increases sub-linearly with power: initial SNR-limited regime transitions into an interference / leakage-aware saturation region. Beyond a threshold, increased power yields limited secrecy benefit, emphasizing intelligent spatial / phase control over brute force power scaling.

## 4. Secrecy Rate vs. Number of BS Antennas
Highlights array scaling benefits (beamforming and spatial filtering) with gradually tapering gains. Larger arrays help suppress eavesdropper channels while reinforcing legitimate links; curvature indicates law of diminishing returns beyond moderate antenna counts.

## 5. Secrecy Rate vs. Eavesdropper Count
As the number of potential eavesdroppers increases, aggregate interception probability rises and effective secrecy rate declines. Curve slope moderates for large populations as system design (beam shaping + RIS diversity) partially mitigates worst-case exposure.

---
These five figures collectively evidence: (i) algorithmic hierarchy (Hybrid > LLM > MLP), (ii) infrastructure scaling levers (RIS elements, antennas), (iii) power-efficiency trade-offs, and (iv) robustness degradation under adversarial density.
