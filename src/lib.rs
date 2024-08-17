pub mod agent;
pub mod batch;
pub mod env;
pub mod layers;
pub mod model;
pub mod trainer;

use std::{fmt::Debug, path::Path};

use burn::tensor::{backend::Backend, Tensor};
use redb::TypeName;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub trait State: Clone + Debug + Send {
    fn new(observation: Vec<f32>) -> Self;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl<S: State + Serialize + DeserializeOwned + Debug + 'static> redb::Value for Experience<S> {
    type SelfType<'a> = Experience<S>;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let data = zstd::stream::decode_all(data).unwrap();
        rmp_serde::from_slice(data.as_slice()).unwrap()
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'a,
        Self: 'b,
    {
        let data = rmp_serde::to_vec(value).unwrap();
        zstd::stream::encode_all(data.as_slice(), 3).unwrap()
    }

    fn type_name() -> TypeName {
        TypeName::new("Experience")
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

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationSpace<const D: usize> {
    Box { shape: [usize; D] },
}

pub trait Env<const D: usize> {
    fn action_space(&self) -> &ActionSpace;
    fn observation_space(&self) -> &ObservationSpace<D>;
    fn reset(&mut self) -> anyhow::Result<Vec<f32>>;
    fn step(&mut self, action: &Action) -> anyhow::Result<(Vec<f32>, f32, bool)>;
    fn render(&self) -> anyhow::Result<()>;
}

impl <const D: usize> ObservationSpace<D> {
    pub fn shape(&self) -> &[usize; D] {
        match self {
            ObservationSpace::Box { shape, .. } => shape,
        }
    }
}

pub trait Estimator<B: Backend> {
    fn predict<const D: usize>(&self, observation: Tensor<B, D>) -> Tensor<B, 2>;
}

pub trait Agent<S: State>: Clone + Send {
    fn policy(&self, observation: &[f32]) -> Action;
    fn update(
        &mut self,
        gamma: f32,
        experiences: &[Experience<S>],
        weights: &[f32],
    ) -> anyhow::Result<()>;
    fn make_state(&self, next_observation: &[f32], state: &S) -> S;
    fn save<P: AsRef<Path>>(&self, artifacts_dir: P) -> anyhow::Result<()>;
    fn load<P: AsRef<Path>>(&mut self, restore_dir: P) -> anyhow::Result<()>;
}

pub trait PrioritizedReplay<S: State> {
    fn temporaral_difference_error(&self, gamma: f32, experiences: &[Experience<S>]) -> Vec<f32>;
}

pub trait PrioritizedReplayAgent<S: State>: Agent<S> + PrioritizedReplay<S> {}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeepQNetworkState {
    pub observation: Vec<f32>,
    pub next_observation: Vec<f32>,
}

impl State for DeepQNetworkState {
    fn new(observation: Vec<f32>) -> Self {
        Self {
            observation: Vec::new(),
            next_observation: observation,
        }
    }
}
