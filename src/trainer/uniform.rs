use std::{fs::File, io::Write, path::PathBuf}; // Add the Write trait import

use anyhow::Context as _;
use rand::{seq::SliceRandom, Rng as _};
use serde_json::json;

use crate::{Action, ActionSpace, Agent, Env, Experience, State};

pub struct UniformReplayTrainer {
    episode: usize,
    gamma: f32,
    init_exploration: f32,
    final_exploration: f32,
    exploration_decay: f32,
    artifacts_dir: PathBuf,
    render: bool,
}

impl UniformReplayTrainer {
    pub fn new(
        episode: usize,
        gamma: f32,
        init_exploration: f32,
        final_exploration: f32,
        exploration_decay: f32,
        artifacts_dir: PathBuf,
        render: bool,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&artifacts_dir).with_context(|| "create artifact dir")?;
        Ok(Self {
            episode,
            gamma,
            init_exploration,
            final_exploration,
            exploration_decay,
            artifacts_dir,
            render,
        })
    }

    pub fn train_loop<S: State>(
        &self,
        agent: &mut impl Agent<S>,
        env: &mut impl Env,
        memory: &mut UniformReplayMemory<S>,
    ) -> anyhow::Result<()> {
        for epi in 0..self.episode {
            let mut step = 0;
            let mut cumulative_reward = 0.0;
            let mut observation = env.reset()?;
            let mut reward;
            let mut is_done = false;
            let mut state = S::new(observation.clone());
            let episode_artifacts_dir = self.artifacts_dir.join(format!("{}", epi));
            std::fs::create_dir_all(&episode_artifacts_dir).with_context(|| {
                format!("create episode artifact dir {:?}", episode_artifacts_dir)
            })?;
            let train_log_path = episode_artifacts_dir.join("train.jsonl");
            let mut train_logger =
                File::create(&train_log_path).with_context(|| "create train log file")?;
            while !is_done {
                step += 1;

                if self.render {
                    env.render()?;
                }

                let action = if (self.init_exploration * self.exploration_decay.powf(epi as f32))
                    .max(self.final_exploration)
                    < rand::thread_rng().gen::<f32>()
                {
                    agent.policy(&observation)
                } else {
                    let action_space = env.action_space();
                    match action_space {
                        ActionSpace::Discrete(n) => {
                            let action = rand::thread_rng().gen_range(0..*n);
                            Action::Discrete(action)
                        }
                    }
                };

                (observation, reward, is_done) = env.step(&action)?;
                state = agent.make_state(&observation, &state);
                cumulative_reward += reward;
                let log = json!({
                    "episode": epi,
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward
                });
                writeln!(&mut train_logger, "{}", log).with_context(|| "write train log")?;

                let experience = Experience {
                    state: state.clone(),
                    action,
                    reward,
                    is_done,
                };
                memory.push(experience);

                if epi > 0 {
                    let batch = memory.sample();
                    agent.update(self.gamma, &batch)?;
                }
            }
            agent.save(&episode_artifacts_dir)?;
        }
        Ok(())
    }
}

pub struct UniformReplayMemory<S: State> {
    buffer: Vec<Experience<S>>,
    max_buffer_size: usize,
    batch_size: usize,
    counter: usize,
}

impl<S: State> UniformReplayMemory<S> {
    pub fn new(max_buffer_size: usize, batch_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_buffer_size,
            batch_size,
            counter: 0,
        }
    }

    fn push(&mut self, experience: Experience<S>) {
        if self.buffer.len() < self.max_buffer_size {
            self.buffer.push(experience);
            self.counter += 1;
        } else {
            self.buffer[self.counter] = experience;
            self.counter += 1;
            if self.counter == self.max_buffer_size {
                self.counter = 0;
            }
        }
    }

    fn sample(&self) -> Vec<Experience<S>> {
        self.buffer
            .choose_multiple(&mut rand::thread_rng(), self.batch_size)
            .cloned()
            .collect()
    }
}
