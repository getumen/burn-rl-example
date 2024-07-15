import click
import pathlib
import json
from matplotlib import pyplot as plt
import numpy as np

@click.command()
@click.option('--dir', help='experiment dir')
def main(dir):
    path = pathlib.Path(dir)
    if not path.exists():
        print(f"{dir} does not exist")
        return
    epoch_dirs = sorted([(int(x.name), x) for x in path.iterdir() if x.is_dir()], key=lambda x: x[0])
    rewards = []
    for epoch, epoch_dir in epoch_dirs:
        with open(epoch_dir / 'train.jsonl', 'r') as f:
            reward = 0
            for line in f:
                data = json.loads(line)
                reward += data['reward']
            rewards.append((epoch, reward))
    x, y = zip(*rewards)
    avg_y = np.convolve(y, np.ones(20)/20, mode='same')
    plt.plot(x, y, label='reward')
    plt.plot(x, avg_y, label='avg reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Rewards')
    plt.savefig('rewards.png')


if __name__ == '__main__':
    main()