# JSAC LLM

Reinforcement learning research code for secrecy / joint sensing & communication with different actor architectures (MLP, LLM-based, Hybrid) and RIS / IAB enhancements.

## Features
- Synthetic + (extendable) environment for secrecy rate optimization
- Comparison of actor types (DDPG variants)
- Plot generation scripts in `comprehensive_plots.py` and single-purpose `plot_*.py` files
- Easily extensible for additional channel / RIS models
- Hybrid actor architecture combining numerical + language-conditioned policy inputs
- Delay Alignment Modulation (DAM) support hooks (see `jsac_active_ris_dam.py`)

## Quick Start
```bash
git clone https://github.com/Utkarsh-upadhyay9/jsac_llm.git
cd jsac_llm
pip install -r requirements.txt
python generate_all_plots.py  # regenerate figures into plots/
```

## Core Scripts
- `jsac_active_ris_dam.py`: Environment + actor definitions including DAM logic
- `jsac.py`: Base RL logic (DDPG variant) without extended hybrid pieces
- `generate_all_plots.py`: Batch driver to rebuild all figures
- `comprehensive_plots.py`: Single entry for the 8 figure suite + main convergence
- `plot_convergence_final.py`: Standalone extended convergence visualization

## Generating Individual Figures
```bash
python plot_4_secrecy_vs_ris.py          # RIS elements impact
python plot_5_secrecy_vs_power.py        # Power sweep
python plot_8_secrecy_vs_bs_antennas.py  # Antennas sweep
```
All outputs appear under `plots/`.

## Reproducibility Notes
Set a fixed NumPy random seed (already done inside plotting scripts) for deterministic synthetic curves. For RL training reproducibility (if extended), additionally set `torch.manual_seed(SEED)` and control CUDA determinism flags.

## Roadmap (Short)
- [ ] Add multi-eavesdropper evaluation utility
- [ ] Add config-driven experiment runner
- [ ] Optional WandB / TensorBoard logging wrapper
- [ ] Parameter sweep script for antenna / RIS scaling laws

## Directory Overview
- `jsac*.py` core RL / secrecy logic variants
- `plot_*.py` individual figure scripts
- `comprehensive_plots.py` batch figure generation
- `plots/` output figures

## Contributing
See `CONTRIBUTING.md`.

### Quick Contribution Guidelines
1. Create a feature branch: `git checkout -b feature/name`
2. Add tests / example usage when adding new environment dynamics
3. Run lint (optional) & regenerate affected plots before PR
4. Open PR referencing any related issues

## License
Research use; add a LICENSE file if distributing.

## Citation
(Provide paper / preprint once available.)
