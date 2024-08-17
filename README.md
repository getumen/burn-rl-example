# burn-rl-example

```bash
cargo run --bin dqn_trainer -- --artifacts-path artifacts --env-name Acrobot-v1 --prioritized --dueling --double-dqn
cargo run --bin dqn_trainer -- --artifacts-path artifacts --env-name CartPole-v1 --prioritized --dueling --double-dqn
cargo run --bin dqn_trainer -- --artifacts-path artifacts --env-name MountainCar-v0 --prioritized --dueling --double-dqn
cargo run --bin dqn_trainer -- --artifacts-path artifacts --env-name LunarLander-v2 --prioritized --dueling --double-dqn
```

## plot rewards

```
rye run python python/plot_rewards.py --dir artifacts/LunarLander-v2/20240714_182951/
```