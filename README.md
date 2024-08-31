# burn-rl-example

use Python 3.11

```bash
cargo run --bin trainer -- categorical --artifacts-path artifacts --env-name Acrobot-v1 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
cargo run --bin trainer -- categorical --artifacts-path artifacts --env-name CartPole-v1 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
cargo run --bin trainer -- categorical --artifacts-path artifacts --env-name MountainCar-v0 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
cargo run --bin trainer -- quantile --artifacts-path artifacts --env-name LunarLander-v2 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
cargo run --bin trainer -- categorical --artifacts-path artifacts --env-name Breakout-v4 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
cargo run --bin trainer -- categorical --artifacts-path artifacts --env-name SuperMarioBros-v3 --batch-size 64 --n-step 3 --bellman-gamma 0.99 --prioritized --dueling --double-dqn --noisy --render
```

## plot rewards

```
rye run python python/plot_rewards.py --dir artifacts/LunarLander-v2/20240714_182951/
```