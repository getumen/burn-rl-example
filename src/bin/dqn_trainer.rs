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
    env::gymnasium::GymnasiumEnv,
    trainer::{
        prioritized::{PrioritizedReplayMemory, PrioritizedReplayTrainer},
        uniform::{UniformReplayMemory, UniformReplayTrainer},
    },
    ActionSpace, Agent, Env as _, Estimator, ObservationSpace,
};
use chrono::Local;
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
    #[arg(short, long)]
    prioritized: bool,
}

#[derive(Module, Debug)]
pub struct DeepQNetworkModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    value_linear1: Linear<B>,
    value_linear2: Linear<B>,
    advantage_linear1: Linear<B>,
    advantage_linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> DeepQNetworkModel<B> {
    pub fn new(
        device: &B::Device,
        observation_space: &ObservationSpace,
        action_space: &ActionSpace,
    ) -> Self {
        Self {
            linear1: LinearConfig::new(
                observation_space.shape().iter().product::<i64>() as usize,
                64,
            )
            .init(device),
            linear2: LinearConfig::new(64, 64).init(device),
            value_linear1: LinearConfig::new(64, 64).init(device),
            value_linear2: LinearConfig::new(64, 1).init(device),
            advantage_linear1: LinearConfig::new(64, 64).init(device),
            advantage_linear2: match action_space {
                ActionSpace::Discrete(n) => LinearConfig::new(64, *n as usize).init(device),
            },
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Estimator<B> for DeepQNetworkModel<B> {
    fn predict(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = observation;
        let x = self.activation.forward(self.linear1.forward(x));
        let x = self.activation.forward(self.linear2.forward(x));
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

    let now = Local::now();
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
            1000,
            true,
        );

        if let Some(restore_path) = &args.restore_path {
            agent.load(restore_path).with_context(|| "load agent")?;
        }

        if args.prioritized {
            let mut memory = PrioritizedReplayMemory::new(2usize.pow(14), 64, 0.6);

            let trainer =
                PrioritizedReplayTrainer::new(10000, 0.99, 1.0, 0.01, 0.98, artifacts_path, true)?;

            trainer.train_loop(&mut agent, &mut env, &mut memory)?;
        } else {
            let mut memory = UniformReplayMemory::new(2usize.pow(16), 64);

            let trainer =
                UniformReplayTrainer::new(10000, 0.99, 1.0, 0.01, 0.98, artifacts_path, true)?;

            trainer.train_loop(&mut agent, &mut env, &mut memory)?;
        }

        Ok(())
    })?;

    Ok(())
}
