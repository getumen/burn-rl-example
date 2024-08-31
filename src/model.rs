use crate::{
    layers::{NoisyLinear, NoisyLinearConfig},
    ActionSpace, Distributional, Estimator, ObservationSpace,
};
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, Relu,
    },
    prelude::Backend,
    tensor::{Data, Shape, Tensor},
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
    pub fn new(device: &B::Device, input_size: usize, output_size: usize, noisy: bool) -> Self {
        Self {
            linear1: if noisy {
                LinearLayerType::NoisyLinear(
                    NoisyLinearConfig::new(input_size, input_size).init(device),
                )
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            linear2: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(64, output_size).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(64, output_size).init(device))
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
    num_class: usize,
    atoms: usize,
}

impl<B: Backend> DuelingLayer<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        num_class: usize,
        atoms: usize,
        noisy: bool,
    ) -> Self {
        Self {
            value_linear1: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(input_size, 64).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            value_linear2: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(64, atoms).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(64, atoms).init(device))
            },
            advantage_linear1: if noisy {
                LinearLayerType::NoisyLinear(NoisyLinearConfig::new(input_size, 64).init(device))
            } else {
                LinearLayerType::Linear(LinearConfig::new(input_size, 64).init(device))
            },
            advantage_linear2: if noisy {
                LinearLayerType::NoisyLinear(
                    NoisyLinearConfig::new(64, num_class * atoms).init(device),
                )
            } else {
                LinearLayerType::Linear(LinearConfig::new(64, num_class * atoms).init(device))
            },
            activation: Relu::new(),
            num_class,
            atoms,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = x.shape().dims[0];
        let value = self
            .activation
            .forward(self.value_linear1.forward(x.clone()));
        let value = self.value_linear2.forward(value);
        let advantage = self
            .activation
            .forward(self.advantage_linear1.forward(x.clone()));
        let advantage = self.advantage_linear2.forward(advantage);
        let advantage = advantage.clone() - advantage.clone().mean_dim(1);
        let output = value.reshape([batch_size, 1, self.atoms])
            + advantage.reshape([batch_size, self.num_class, self.atoms]);
        output.reshape([batch_size, self.num_class * self.atoms])
    }
}

#[derive(Module, Debug)]
pub enum ValueLayer<B: Backend> {
    Linear(LinearValueLayer<B>),
    Dueling(DuelingLayer<B>),
}

impl<B: Backend> ValueLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            ValueLayer::Linear(layer) => layer.forward(x),
            ValueLayer::Dueling(layer) => layer.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub enum FeatureExtractionLayer<B: Backend> {
    Linear(Linear<B>),
    Conv2d(Conv2d<B>),
}

#[derive(Clone, Debug)]
pub enum OutputLayerConfig {
    Expectation,
    CategoricalDistribution {
        atoms: usize,
        min_value: f32,
        max_value: f32,
    },
    QuantileRegression {
        quantiles: usize,
    },
}

impl OutputLayerConfig {
    pub fn score_num(&self) -> usize {
        match self {
            OutputLayerConfig::Expectation => 1,
            OutputLayerConfig::CategoricalDistribution { atoms, .. } => *atoms,
            OutputLayerConfig::QuantileRegression { quantiles } => *quantiles,
        }
    }
}

#[derive(Module, Debug)]
pub struct CategoricalDistributionLayer<B: Backend> {
    value_layer: ValueLayer<B>,
    z: Tensor<B, 1>,
    atoms: usize,
    action_num: usize,
}

impl<B: Backend> CategoricalDistributionLayer<B> {
    pub fn new(
        value_layer: ValueLayer<B>,
        action_num: usize,
        atoms: usize,
        min_value: f32,
        max_value: f32,
        device: &B::Device,
    ) -> Self {
        let z = (0..atoms)
            .map(|i| min_value + (max_value - min_value) * (i as f32) / (atoms as f32 - 1.0))
            .collect::<Vec<_>>();
        let z = Tensor::from_data(Data::new(z, Shape::from([atoms])).convert(), device);

        Self {
            value_layer,
            z,
            atoms,
            action_num,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let shape = x.shape().dims;
        let prob = self.forward_distribution(x);
        let q_means =
            (prob.clone() * self.z.clone().no_grad().reshape([1, 1, self.atoms])).sum_dim(2);
        q_means.reshape([shape[0], self.action_num])
    }

    pub fn forward_distribution(&self, x: Tensor<B, 2>) -> Tensor<B, 3> {
        let shape = x.shape().dims;
        let x = self.value_layer.forward(x);
        let x = x.reshape([shape[0], self.action_num, self.atoms]);
        burn::tensor::activation::softmax(x, 2)
    }
}

#[derive(Module, Debug)]
pub struct QuantileRegressionLayer<B: Backend> {
    value_layer: ValueLayer<B>,
    quantiles: usize,
    action_num: usize,
}

impl<B: Backend> QuantileRegressionLayer<B> {
    pub fn new(value_layer: ValueLayer<B>, quantiles: usize, action_num: usize) -> Self {
        Self {
            value_layer,
            quantiles,
            action_num,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let shape = x.shape().dims;
        let quantiles = self.forward_distribution(x);
        let q_means = quantiles.mean_dim(2);
        q_means.reshape([shape[0], self.action_num])
    }

    pub fn forward_distribution(&self, x: Tensor<B, 2>) -> Tensor<B, 3> {
        let shape = x.shape().dims;
        let x = self.value_layer.forward(x);
        let x = x.reshape([shape[0], self.action_num, self.quantiles]);
        x
    }
}

#[derive(Module, Debug)]

pub enum OutputLayer<B: Backend> {
    Expectation(ValueLayer<B>),
    CategoricalDistribution(CategoricalDistributionLayer<B>),
    QuantileRegression(QuantileRegressionLayer<B>),
}

#[derive(Module, Debug)]
pub struct DeepQNetworkModel<B: Backend> {
    layer1: FeatureExtractionLayer<B>,
    layer2: FeatureExtractionLayer<B>,
    output_layer: OutputLayer<B>,
    activation: Relu,
}

impl<B: Backend> DeepQNetworkModel<B> {
    pub fn new<const D: usize>(
        device: &B::Device,
        observation_space: &ObservationSpace<D>,
        action_space: &ActionSpace,
        dueling: bool,
        noisy: bool,
        output_layer_config: OutputLayerConfig,
    ) -> Self {
        let stride = 4;
        let kernel_size = 4;
        let value_layer_input_dim = if D == 2 {
            64
        } else if D == 4 {
            let shape = observation_space.shape();
            let shape = [shape[1], shape[2], shape[3]];
            let shape = [
                16,
                (shape[1] - kernel_size) / stride + 1,
                (shape[2] - kernel_size) / stride + 1,
            ];
            let shape = [
                32,
                (shape[1] - kernel_size) / stride + 1,
                (shape[2] - kernel_size) / stride + 1,
            ];
            shape.iter().product()
        } else {
            unimplemented!()
        };
        let layer1 = if D == 2 {
            FeatureExtractionLayer::Linear(
                LinearConfig::new(observation_space.shape()[1], 64).init(device),
            )
        } else if D == 4 {
            FeatureExtractionLayer::Conv2d(
                Conv2dConfig::new([3, 16], [kernel_size, kernel_size])
                    .with_stride([stride, stride])
                    .init(device),
            )
        } else {
            unimplemented!()
        };

        let layer2 = if D == 2 {
            FeatureExtractionLayer::Linear(LinearConfig::new(64, 64).init(device))
        } else if D == 4 {
            FeatureExtractionLayer::Conv2d(
                Conv2dConfig::new([16, 32], [kernel_size, kernel_size])
                    .with_stride([stride, stride])
                    .init(device),
            )
        } else {
            unimplemented!()
        };

        let output_layer = match output_layer_config {
            OutputLayerConfig::Expectation => {
                let value_layer = if dueling {
                    ValueLayer::Dueling(DuelingLayer::new(
                        device,
                        value_layer_input_dim,
                        1,
                        action_space.size(),
                        noisy,
                    ))
                } else {
                    ValueLayer::Linear(LinearValueLayer::new(
                        device,
                        value_layer_input_dim,
                        action_space.size(),
                        noisy,
                    ))
                };
                OutputLayer::Expectation(value_layer)
            }
            OutputLayerConfig::CategoricalDistribution {
                atoms,
                max_value,
                min_value,
            } => {
                let value_layer = if dueling {
                    ValueLayer::Dueling(DuelingLayer::new(
                        device,
                        value_layer_input_dim,
                        action_space.size(),
                        atoms,
                        noisy,
                    ))
                } else {
                    ValueLayer::Linear(LinearValueLayer::new(
                        device,
                        value_layer_input_dim,
                        atoms * action_space.size(),
                        noisy,
                    ))
                };
                OutputLayer::CategoricalDistribution(CategoricalDistributionLayer::new(
                    value_layer,
                    action_space.size(),
                    atoms,
                    min_value,
                    max_value,
                    device,
                ))
            }
            OutputLayerConfig::QuantileRegression { quantiles } => {
                let value_layer = if dueling {
                    ValueLayer::Dueling(DuelingLayer::new(
                        device,
                        value_layer_input_dim,
                        action_space.size(),
                        quantiles,
                        noisy,
                    ))
                } else {
                    ValueLayer::Linear(LinearValueLayer::new(
                        device,
                        value_layer_input_dim,
                        quantiles * action_space.size(),
                        noisy,
                    ))
                };
                OutputLayer::QuantileRegression(QuantileRegressionLayer::new(
                    value_layer,
                    quantiles,
                    action_space.size(),
                ))
            }
        };
        Self {
            layer1,
            layer2,
            output_layer,
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
                self.activation.forward(
                    layer2.forward(
                        self.activation
                            .forward(layer1.forward(x.reshape([shape[0], shape[1]]))),
                    ),
                )
            }
            (FeatureExtractionLayer::Conv2d(layer1), FeatureExtractionLayer::Conv2d(layer2)) => {
                let shape = x.shape().dims;
                let x = self
                    .activation
                    .forward(layer1.forward(x.reshape([shape[0], shape[1], shape[2], shape[3]])));
                let x = self.activation.forward(layer2.forward(x));
                x.flatten(1, 3)
            }
            _ => unimplemented!(),
        };
        match &self.output_layer {
            OutputLayer::Expectation(layer) => match layer {
                ValueLayer::Linear(layer) => layer.forward(x),
                ValueLayer::Dueling(layer) => layer.forward(x),
            },
            OutputLayer::CategoricalDistribution(layer) => layer.forward(x),
            OutputLayer::QuantileRegression(layer) => layer.forward(x),
        }
    }
}

impl<B: Backend> Distributional<B> for DeepQNetworkModel<B> {
    fn get_distribution<const D: usize>(&self, observation: Tensor<B, D>) -> Tensor<B, 3> {
        let x = observation;
        let x = match (&self.layer1, &self.layer2) {
            (FeatureExtractionLayer::Linear(layer1), FeatureExtractionLayer::Linear(layer2)) => {
                let shape = x.shape().dims;
                self.activation.forward(
                    layer2.forward(
                        self.activation
                            .forward(layer1.forward(x.reshape([shape[0], shape[1]]))),
                    ),
                )
            }
            (FeatureExtractionLayer::Conv2d(layer1), FeatureExtractionLayer::Conv2d(layer2)) => {
                let shape = x.shape().dims;
                let x = self
                    .activation
                    .forward(layer1.forward(x.reshape([shape[0], shape[1], shape[2], shape[3]])));
                let x = self.activation.forward(layer2.forward(x));
                x.flatten(1, 3)
            }
            _ => unimplemented!(),
        };
        match &self.output_layer {
            OutputLayer::Expectation(_) => unimplemented!("Expectation not supported"),
            OutputLayer::CategoricalDistribution(layer) => layer.forward_distribution(x),
            OutputLayer::QuantileRegression(layer) => layer.forward_distribution(x),
        }
    }
}
