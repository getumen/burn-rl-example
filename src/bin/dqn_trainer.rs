use std::path::PathBuf;

use anyhow::Context;
use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    lr_scheduler::constant::ConstantLr,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::AdamConfig,
    tensor::{backend::Backend, Tensor},
};
use burn_rl_example::{
    dqn::DeepQNetworkAgent,
    env::{Env, GymnasiumEnv},
    ActionSpace, Agent, Estimator, ObservationSpace, Trainer,
};
use clap::Parser;
use pyo3::Python;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    artifacts_path: PathBuf,
    #[arg(short, long)]
    env_name: String,
    #[arg(short, long)]
    restore_path: Option<PathBuf>,
}

#[derive(Module, Debug)]
pub struct DeepQNetworkModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> DeepQNetworkModel<B> {
    pub fn new(
        device: &B::Device,
        observation_space: &ObservationSpace,
        action_space: &ActionSpace,
    ) -> Self {
        let linear1 = LinearConfig::new(
            observation_space.shape().iter().product::<i64>() as usize,
            8,
        )
        .init(device);
        let linear2 = match action_space {
            ActionSpace::Discrete(n) => LinearConfig::new(8, *n as usize).init(device),
        };
        let activation = Relu::new();

        Self {
            linear1,
            linear2,
            activation,
        }
    }
}

impl<B: Backend> Estimator<B> for DeepQNetworkModel<B> {
    fn predict(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = observation;
        let x = self.activation.forward(self.linear1.forward(x));
        self.linear2.forward(x)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    type Backend = LibTorch;
    type AutodiffBackend = Autodiff<Backend>;
    let device = if tch::utils::has_cuda() {
        println!("CUDA is available");
        LibTorchDevice::Cuda(0)
    } else if tch::utils::has_vulkan() {
        println!("Vulkan is available");
        LibTorchDevice::Vulkan
    } else {
        println!("CUDA and Vulkan are not available");
        LibTorchDevice::Cpu
    };

    let now = chrono::Local::now();
    let artifacts_path = args.artifacts_path.clone();
    let artifacts_path = artifacts_path.join(&args.env_name);

    let yyyymmdd_hhmmss = now.format("%Y%m%d_%H%M%S");
    let artifacts_path = artifacts_path.join(yyyymmdd_hhmmss.to_string());

    Python::with_gil(|py| -> anyhow::Result<()> {
        let mut env =
            GymnasiumEnv::new(py, &args.env_name).with_context(|| "create gymnasium env")?;

        let model = DeepQNetworkModel::<AutodiffBackend>::new(
            &device,
            env.observation_space(),
            env.action_space(),
        );
        let optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();

        let mut agent = DeepQNetworkAgent::new(
            model,
            optimizer,
            ConstantLr::new(0.001),
            env.action_space().clone(),
            device,
        );

        if let Some(restore_path) = &args.restore_path {
            agent.load(restore_path).with_context(|| "load agent")?;
        }

        let trainer = Trainer::new(1000, 50000, 32, 0.99, 0.5, artifacts_path)?;

        trainer.train_loop(&mut agent, &mut env)?;

        Ok(())
    })?;

    Ok(())
}
