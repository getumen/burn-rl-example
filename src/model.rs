use crate::{ActionSpace, Estimator, ObservationSpace};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct LinearValueLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> LinearValueLayer<B> {
    pub fn new(device: &B::Device, input_size: usize, action_space: &ActionSpace) -> Self {
        Self {
            linear1: LinearConfig::new(input_size, 64).init(device),
            linear2: match action_space {
                ActionSpace::Discrete(n) => LinearConfig::new(64, *n as usize).init(device),
            },
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.linear1.forward(x));
        self.linear2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct DuelingLayer<B: Backend> {
    value_linear1: Linear<B>,
    value_linear2: Linear<B>,
    advantage_linear1: Linear<B>,
    advantage_linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> DuelingLayer<B> {
    pub fn new(device: &B::Device, input_size: usize, action_space: &ActionSpace) -> Self {
        Self {
            value_linear1: LinearConfig::new(input_size, 64).init(device),
            value_linear2: LinearConfig::new(64, 1).init(device),
            advantage_linear1: LinearConfig::new(input_size, 64).init(device),
            advantage_linear2: match action_space {
                ActionSpace::Discrete(n) => LinearConfig::new(64, *n as usize).init(device),
            },
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let value = self
            .activation
            .forward(self.value_linear1.forward(x.clone()));
        let value = self.value_linear2.forward(value);
        let advantage = self
            .activation
            .forward(self.advantage_linear1.forward(x.clone()));
        let advantage = self.advantage_linear2.forward(advantage);
        let advantage = advantage.clone() - advantage.clone().mean_dim(1);
        value + advantage
    }
}

#[derive(Module, Debug)]
pub enum ValueLayer<B: Backend> {
    Linear(LinearValueLayer<B>),
    Dueling(DuelingLayer<B>),
}

#[derive(Module, Debug)]
pub struct DeepQNetworkModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    value_layer: ValueLayer<B>,
    activation: Relu,
}

impl<B: Backend> DeepQNetworkModel<B> {
    pub fn new(
        device: &B::Device,
        observation_space: &ObservationSpace,
        action_space: &ActionSpace,
        dueling: bool,
    ) -> Self {
        let value_layer = if dueling {
            ValueLayer::Dueling(DuelingLayer::new(device, 64, action_space))
        } else {
            ValueLayer::Linear(LinearValueLayer::new(device, 64, action_space))
        };
        Self {
            linear1: LinearConfig::new(
                observation_space.shape().iter().product::<i64>() as usize,
                64,
            )
            .init(device),
            linear2: LinearConfig::new(64, 64).init(device),
            value_layer,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Estimator<B> for DeepQNetworkModel<B> {
    fn predict(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = observation;
        let x = self.activation.forward(self.linear1.forward(x));
        let x = self.activation.forward(self.linear2.forward(x));
        match &self.value_layer {
            ValueLayer::Linear(layer) => layer.forward(x),
            ValueLayer::Dueling(layer) => layer.forward(x),
        }
    }
}
