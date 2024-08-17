use burn::{
    config::Config,
    module::{Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Data, Shape, Tensor},
};
use rand_distr::{Distribution, Normal};

#[derive(Config, Debug)]
pub struct NoisyLinearConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub mu_initializer: Initializer,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct NoisyLinear<B: Backend> {
    pub weight_mu: Param<Tensor<B, 2>>,
    pub weight_sigma: Param<Tensor<B, 2>>,
    pub bias_mu: Option<Param<Tensor<B, 1>>>,
    pub bias_sigma: Option<Param<Tensor<B, 1>>>,
}

impl NoisyLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NoisyLinear<B> {
        let shape = [self.d_input, self.d_output];
        let sigma_initializer = Initializer::Constant {
            value: 0.5 / (self.d_input as f64).sqrt(),
        };
        let weight_mu =
            self.mu_initializer
                .init_with(shape, Some(self.d_input), Some(self.d_output), device);
        let weight_sigma =
            sigma_initializer.init_with(shape, Some(self.d_input), Some(self.d_output), device);
        let bias_mu = if self.bias {
            Some(self.mu_initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };
        let bias_sigma = if self.bias {
            Some(sigma_initializer.init_with([self.d_output], Some(self.d_input), Some(1), device))
        } else {
            None
        };

        NoisyLinear {
            weight_mu,
            weight_sigma,
            bias_mu,
            bias_sigma,
        }
    }
}

impl<B: Backend> NoisyLinear<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if D == 1 {
            return Self::forward::<2>(self, input.unsqueeze()).flatten(0, 1);
        }
        let device = input.device();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let d_input = self.weight_mu.shape().dims[0];
        let d_output = self.weight_mu.shape().dims[1];
        let epsilon_in = (0..d_input)
            .into_iter()
            .map(|_| normal.sample(&mut rng))
            .map(|x: f32| x.signum() * x.abs().sqrt())
            .collect();
        let epsilon_in = Tensor::from_data(
            Data::new(epsilon_in, Shape::new([d_input, 1])).convert(),
            &device,
        );
        let epsilon_out = (0..self.weight_mu.shape().dims[1])
            .into_iter()
            .map(|_| normal.sample(&mut rng))
            .map(|x: f32| x.signum() * x.abs().sqrt())
            .collect();
        let epsilon_out = Tensor::from_data(
            Data::new(epsilon_out, Shape::new([1, d_output])).convert(),
            &device,
        );

        let weight_epsilon = epsilon_in.matmul(epsilon_out.clone());

        let weight = self.weight_mu.val() + self.weight_sigma.val() * weight_epsilon;

        let output = input.matmul(weight.unsqueeze());

        match (&self.bias_mu, &self.bias_sigma) {
            (Some(bias_mu), Some(bias_sigma)) => {
                let bais_epsion = epsilon_out.clone().squeeze(0);
                let bias = bias_mu.val() + bias_sigma.val() * bais_epsion;
                output + bias.unsqueeze()
            }
            _ => output,
        }
    }
}
