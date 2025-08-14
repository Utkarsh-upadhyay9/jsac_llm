# JSAC LLM

Reinforcement learning research code for secrecy / joint sensing & communication with different actor architectures (MLP, LLM-based, Hybrid) and RIS / IAB enhancements.

## Features
- Synthetic + (extendable) environment for secrecy rate optimization
- Comparison of actor types (DDPG variants)
- Plot generation scripts in `comprehensive_plots.py` and single-purpose `plot_*.py` files
- Easily extensible for additional channel / RIS models

## Quick Start
```bash
git clone https://github.com/Utkarsh-upadhyay9/jsac_llm.git
cd jsac_llm
pip install -r requirements.txt
python generate_all_plots.py  # regenerate figures into plots/
```

## Directory Overview
- `jsac*.py` core RL / secrecy logic variants
- `plot_*.py` individual figure scripts
- `comprehensive_plots.py` batch figure generation
- `plots/` output figures

## Contributing
See `CONTRIBUTING.md`.

## License
Research use; add a LICENSE file if distributing.

## Citation
(Provide paper / preprint once available.)
