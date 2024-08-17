use crate::{
    layers::{NoisyLinear, NoisyLinearConfig},
    ActionSpace, Estimator, ObservationSpace,
};
use burn::{
    module::Module, nn::{conv::{Conv2d, Conv2dConfig}, Linear, LinearConfig, Relu}, prelude::Backend, tensor::Tensor
};

#[derive(Module, Debug)]
pub enum LinearLayerType<B: Backend> {
    Linear(Linear<B>),
    NoisyLinear(NoisyLinear<B>),
}

impl<B: Backend> LinearLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            LinearLayerType::Linear(layer) => layer.forward(x),
            LinearLayerType::NoisyLinear(layer) => layer.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearValueLayer<B: Backend> {
    linear1: LinearLayerType<B>,
    linear2: LinearLayerType<B>,
    activation: Relu,
}

impl<B: Backend> LinearValueLayer<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        action_space: &ActionSpace,
        noisy: bool,
    ) -> Self {
        Self {
            linear1: if noisy {
                LinearLayerType::NoisyLinear(
                    NoisyLinearConfig::new(input_size, input_size).init(device),
                )
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            linear2: match action_space {
                ActionSpace::Discrete(n) => {
                    if noisy {
                        LinearLayerType::NoisyLinear(
                            NoisyLinearConfig::new(64, *n as usize).init(device),
                        )
                    } else {
                        LinearLayerType::Linear(LinearConfig::new(64, *n as usize).init(device))
                    }
                }
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
    value_linear1: LinearLayerType<B>,
    value_linear2: LinearLayerType<B>,
    advantage_linear1: LinearLayerType<B>,
    advantage_linear2: LinearLayerType<B>,
    activation: Relu,
}

impl<B: Backend> DuelingLayer<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        action_space: &ActionSpace,
        noisy: bool,
    ) -> Self {
        Self {
            value_linear1: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(input_size, 64).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            value_linear2: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(64, 1).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(64, 1).init(device))
            },
            advantage_linear1: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(input_size, 64).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            advantage_linear2: match action_space {
                ActionSpace::Discrete(n) => {
                    if noisy {
                        LinearLayerType::NoisyLinear(
                            NoisyLinearConfig::new(64, *n as usize).init(device),
                        )
                    } else {
                        LinearLayerType::Linear(LinearConfig::new(64, *n as usize).init(device))
                    }
                }
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
pub enum FeatureExtractionLayer<B: Backend> {
    Linear(Linear<B>),
    Conv2d(Conv2d<B>),
}

#[derive(Module, Debug)]
pub struct DeepQNetworkModel<B: Backend> {
    layer1: FeatureExtractionLayer<B>,
    layer2: FeatureExtractionLayer<B>,
    value_layer: ValueLayer<B>,
    activation: Relu,
}

impl<B: Backend> DeepQNetworkModel<B> {
    pub fn new<const D: usize>(
        device: &B::Device,
        observation_space: &ObservationSpace<D>,
        action_space: &ActionSpace,
        dueling: bool,
        noisy: bool,
    ) -> Self {
        let value_layer = if dueling {
            ValueLayer::Dueling(DuelingLayer::new(device, 64, action_space, noisy))
        } else {
            ValueLayer::Linear(LinearValueLayer::new(device, 64, action_space, noisy))
        };
        Self {
            layer1: if D == 2 {
                FeatureExtractionLayer::Linear(LinearConfig::new(
                    observation_space.shape()[1],
                    64,
                )
                .init(device))
            } else if D == 4 {
                FeatureExtractionLayer::Conv2d(Conv2dConfig::new([3, 16], [5, 5]).with_stride([2, 2])
                .init(device))
            } else {
                unimplemented!()

            },
            layer2: if D==2 {
                FeatureExtractionLayer::Linear(LinearConfig::new(64, 64).init(device))
            } else if D==4 {
                FeatureExtractionLayer::Conv2d(Conv2dConfig::new([16, 32], [5, 5]).with_stride([2, 2])
                .init(device))
            } else {
                unimplemented!()
            },
            value_layer,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Estimator<B> for DeepQNetworkModel<B> {
    fn predict<const D: usize>(&self, observation: Tensor<B, D>) -> Tensor<B, 2> {
        let x = observation;
        let x = match (&self.layer1, &self.layer2) {
            (FeatureExtractionLayer::Linear(layer1), FeatureExtractionLayer::Linear(layer2)) => {
                let shape = x.shape().dims;
                self.activation.forward(layer2.forward(self.activation.forward(layer1.forward(x.reshape([shape[0], shape[1]])))))
            },
            (FeatureExtractionLayer::Conv2d(layer1), FeatureExtractionLayer::Conv2d(layer2)) => {
                let shape = x.shape().dims;
                let x = self.activation.forward(layer1.forward(x.reshape([shape[0], shape[1], shape[2], shape[3]])));
                let x = self.activation.forward(layer2.forward(x));
                x.flatten(0, 1)
            },
            _ => unimplemented!(),
        };
        match &self.value_layer {
            ValueLayer::Linear(layer) => layer.forward(x),
            ValueLayer::Dueling(layer) => layer.forward(x),
        }
    }
}
