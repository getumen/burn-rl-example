use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::AutodiffBackend, Data, Shape, Tensor},
};

use crate::{Action, ActionSpace, DeepQNetworkState, Experience};

#[derive(Debug, Clone)]
pub struct DeepQNetworkBathcer<B: AutodiffBackend> {
    device: B::Device,
    action_space: ActionSpace,
}

impl<B: AutodiffBackend> DeepQNetworkBathcer<B> {
    pub fn new(device: B::Device, action_space: ActionSpace) -> Self {
        Self {
            device,
            action_space,
        }
    }
}

impl<B: AutodiffBackend> Batcher<Experience<DeepQNetworkState>, DeepQNetworkBatch<B>>
    for DeepQNetworkBathcer<B>
{
    fn batch(&self, items: Vec<Experience<DeepQNetworkState>>) -> DeepQNetworkBatch<B> {
        let (observation, next_observation, action, reward, done): (
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
        ) = items
            .into_iter()
            .filter(|x| !x.state.observation.is_empty() && !x.state.next_observation.is_empty())
            .map(|x| {
                let obs_len = x.state.observation.len();
                let next_obs_len = x.state.next_observation.len();
                (
                    Tensor::from_data(
                        Data::new(x.state.observation, Shape::new([1, obs_len])).convert(),
                        &Default::default(),
                    ),
                    Tensor::from_data(
                        Data::new(x.state.next_observation, Shape::new([1, next_obs_len]))
                            .convert(),
                        &Default::default(),
                    ),
                    match (x.action, self.action_space) {
                        (Action::Discrete(value), ActionSpace::Discrete(num_class)) => {
                            Tensor::one_hot(value as usize, num_class as usize, &Default::default())
                        }
                    },
                    match self.action_space {
                        ActionSpace::Discrete(num_class) => Tensor::from_data(
                            Data::new(vec![x.reward], Shape::new([1, 1])).convert(),
                            &Default::default(),
                        )
                        .repeat(1, num_class as usize),
                    },
                    match self.action_space {
                        ActionSpace::Discrete(num_class) => Tensor::from_data(
                            Data::new(vec![x.is_done as i32], Shape::new([1, 1])).convert(),
                            &Default::default(),
                        )
                        .repeat(1, num_class as usize),
                    },
                )
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |(mut a, mut b, mut c, mut d, mut e), (v, w, x, y, z)| {
                    a.push(v);
                    b.push(w);
                    c.push(x);
                    d.push(y);
                    e.push(z);
                    (a, b, c, d, e)
                },
            );

        let observation = Tensor::cat(observation, 0).to_device(&self.device);
        let next_observation = Tensor::cat(next_observation, 0).to_device(&self.device);
        let action = Tensor::cat(action, 0).to_device(&self.device);
        let reward = Tensor::cat(reward, 0).to_device(&self.device);
        let done = Tensor::cat(done, 0).to_device(&self.device);

        DeepQNetworkBatch {
            observation,
            reward,
            action,
            next_observation,
            done,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepQNetworkBatch<B: AutodiffBackend> {
    pub observation: Tensor<B, 2>,
    pub reward: Tensor<B, 2>,
    pub action: Tensor<B, 2>,
    pub next_observation: Tensor<B, 2>,
    pub done: Tensor<B, 2>,
}