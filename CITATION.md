# Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gold_rl_trading_2025,
  title  = {XAUUSD Reinforcement Learning Trading Pipeline},
  author = {M0-AR},
  year   = {2025},
  url    = {https://github.com/M0-AR/gold-trading-bot}
}
```

## Acknowledgments

This work is based on the original pipeline by **ZiadFrancis**:

```bibtex
@software{ziadfrancis_rl_trading_2024,
  title  = {Reinforcement Trading Part 2},
  author = {ZiadFrancis},
  year   = {2024},
  url    = {https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2}
}
```

We trained from scratch on 23 years of XAUUSD M1 data, reproduced the 35-fold sliding walk-forward validation, and extended the pipeline with parallel CPU training, resume support, and performance optimizations. All results in this repository are independent runs — not copied from the original.

## Related work

This pipeline builds on:

- **Stable Baselines3** — PPO implementation: [Raffin et al., 2021](https://github.com/DLR-RM/stable-baselines3)
- **Gymnasium** — RL environment interface: [Farama Foundation, 2023](https://gymnasium.farama.org/)
- **Walk-forward optimization** — Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
