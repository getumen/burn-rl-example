pub mod dqn;
pub mod env;

use std::{
    collections::VecDeque,
    fs::File,
    io::Write as _,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use burn::tensor::{backend::Backend, Tensor};
use rand::{prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub trait State: Default + Clone {}

#[derive(Debug, Clone)]
pub struct Experience<S: State> {
    state: S,
    action: Action,
    reward: f32,
    is_done: bool,
}

impl<S: State> Experience<S> {
    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn action(&self) -> &Action {
        &self.action
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }

    pub fn is_done(&self) -> bool {
        self.is_done
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSpace {
    Discrete(i64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Discrete(i64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObservationSpace {
    Box { shape: Vec<i64> },
}

impl ObservationSpace {
    pub fn shape(&self) -> &[i64] {
        match self {
            ObservationSpace::Box { shape, .. } => shape,
        }
    }
}

pub trait Estimator<B: Backend> {
    fn predict(&self, observation: Tensor<B, 2>) -> Tensor<B, 2>;
}

pub trait Agent<S: State>: Clone + Send {
    fn policy(&self, observation: &[f32]) -> Action;
    fn update(&mut self, gamma: f32, experiences: &[Experience<S>]) -> anyhow::Result<()>;
    fn make_state(&self, next_observation: &[f32], state: &S) -> S;
    fn save<P: AsRef<Path>>(&self, artifacts_dir: P) -> anyhow::Result<()>;
    fn load<P: AsRef<Path>>(&mut self, restore_dir: P) -> anyhow::Result<()>;
}

pub struct Trainer {
    episode: usize,
    buffer_size: usize,
    batch_size: usize,
    gamma: f32,
    init_exploration: f32,
    final_exploration: f32,
    exploration_decay: f32,
    artifacts_dir: PathBuf,
}

impl Trainer {
    pub fn new(
        episode: usize,
        buffer_size: usize,
        batch_size: usize,
        gamma: f32,
        init_exploration: f32,
        final_exploration: f32,
        exploration_decay: f32,
        artifacts_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&artifacts_dir).with_context(|| "create artifact dir")?;
        Ok(Self {
            episode,
            buffer_size,
            batch_size,
            gamma,
            init_exploration,
            final_exploration,
            exploration_decay,
            artifacts_dir,
        })
    }

    pub fn train_loop<S: State>(
        &self,
        agent: &mut impl Agent<S>,
        env: &mut impl env::Env,
    ) -> anyhow::Result<()> {
        let mut experiences = VecDeque::new();
        for epi in 0..self.episode {
            let mut step = 0;
            let mut cumulative_reward = 0.0;
            let mut observation = env.reset()?;
            let mut reward;
            let mut is_done = false;
            let mut state = S::default();
            let episode_artifacts_dir = self.artifacts_dir.join(format!("{}", epi));
            std::fs::create_dir_all(&episode_artifacts_dir).with_context(|| {
                format!("create episode artifact dir {:?}", episode_artifacts_dir)
            })?;
            let train_log_path = episode_artifacts_dir.join("train.jsonl");
            let mut train_logger =
                File::create(&train_log_path).with_context(|| "create train log file")?;
            while !is_done {
                step += 1;

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
                let next_state = agent.make_state(&observation, &state);
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
                    state,
                    action,
                    reward,
                    is_done,
                };
                experiences.push_back(experience);

                if experiences.len() > self.batch_size {
                    let batch = experiences
                        .make_contiguous()
                        .choose_multiple(&mut rand::thread_rng(), self.batch_size)
                        .cloned()
                        .collect::<Vec<_>>();
                    agent.update(self.gamma, &batch)?;
                }

                if experiences.len() > self.buffer_size {
                    experiences.pop_front();
                }

                state = next_state;
            }
            agent.save(&episode_artifacts_dir)?;
        }
        Ok(())
    }
}
