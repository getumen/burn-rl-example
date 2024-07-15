pub mod dqn;
pub mod env;
pub mod trainer;

use std::path::Path;

use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};

pub trait State: Clone {
    fn new(observation: Vec<f32>) -> Self;
}

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

pub trait Env {
    fn action_space(&self) -> &ActionSpace;
    fn observation_space(&self) -> &ObservationSpace;
    fn reset(&mut self) -> anyhow::Result<Vec<f32>>;
    fn step(&mut self, action: &Action) -> anyhow::Result<(Vec<f32>, f32, bool)>;
    fn render(&self) -> anyhow::Result<()>;
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

pub trait PrioritizedReplay<S: State> {
    fn temporaral_difference_error(&self, gamma: f32, experiences: &[Experience<S>]) -> Vec<f32>;
}

pub trait PrioritizedReplayAgent<S: State>: Agent<S> + PrioritizedReplay<S> {}
