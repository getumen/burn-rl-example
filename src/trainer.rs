use std::collections::VecDeque;

use anyhow::anyhow;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{Experience, State};

pub mod prioritized;
pub mod uniform;

pub struct RandomPolicy {
    init_exploration: f32,
    final_exploration: f32,
    exploration_decay: f32,
}

impl RandomPolicy {
    pub fn new(init_exploration: f32, final_exploration: f32, exploration_decay: f32) -> Self {
        Self {
            init_exploration,
            final_exploration,
            exploration_decay,
        }
    }

    fn epsilon(&self, epi: usize) -> f32 {
        (self.init_exploration * self.exploration_decay.powf(epi as f32))
            .max(self.final_exploration)
    }

    pub fn sample(&self, epi: usize) -> bool {
        rand::random::<f32>() < self.epsilon(epi)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RewardMapping {
    Identity,
    Clip { min: f32, max: f32 },
    Rescaling { epsilon: f32 },
    SymLog,
}

pub struct NStepExperience<S: State + Serialize + DeserializeOwned + 'static> {
    n_step: usize,
    n_step_buffer: VecDeque<Experience<S>>,
    gamma: f32,
    reward_mapping: RewardMapping,
}

impl<S: State + Serialize + DeserializeOwned + 'static> NStepExperience<S> {
    pub fn new(n_step: usize, gamma: f32, reward_mapping: RewardMapping) -> Self {
        Self {
            n_step,
            n_step_buffer: VecDeque::with_capacity(n_step + 1),
            gamma,
            reward_mapping: reward_mapping.clone(),
        }
    }

    pub fn push(&mut self, experience: Experience<S>) -> anyhow::Result<Option<Experience<S>>> {
        self.n_step_buffer.push_back(experience.clone());
        if self.n_step_buffer.len() < self.n_step {
            return Ok(None);
        }

        let mut total_reward = 0.0;
        let mut finally_done = false;

        for (i, exp) in self.n_step_buffer.iter().enumerate() {
            let mask = if exp.is_done() { 0.0 } else { 1.0 };
            total_reward += self.gamma.powi(i as i32) * mask * exp.reward;
            if exp.is_done() {
                finally_done = true;
                break;
            }
        }

        let front = self
            .n_step_buffer
            .pop_front()
            .ok_or(anyhow!("n_step_buffer is empty"))?;

        let total_reward = match self.reward_mapping {
            RewardMapping::Identity => total_reward,
            RewardMapping::Clip { min, max } => total_reward.clamp(min, max),
            RewardMapping::Rescaling { epsilon } => {
                total_reward.signum() * ((1.0 + total_reward.abs()).sqrt() - 1.0)
                    + epsilon * total_reward
            }
            RewardMapping::SymLog => total_reward.signum() * (1.0 + total_reward.abs()).ln(),
        };

        let experience = Experience {
            state: front.state().clone(),
            action: front.action().clone(),
            reward: total_reward,
            is_done: finally_done,
        };
        Ok(Some(experience))
    }
}
