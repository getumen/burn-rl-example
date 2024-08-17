use std::path::PathBuf;

use anyhow::Context;
use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    lr_scheduler::constant::ConstantLr,
    optim::AdamConfig,
};
use burn_rl_example::{
    agent::DeepQNetworkAgent,
    env::gym_super_mario_bros::GymSuperMarioBrosEnv,
    model::DeepQNetworkModel,
    trainer::{
        prioritized::{PrioritizedReplayMemory, PrioritizedReplayTrainer},
        uniform::{UniformReplayMemory, UniformReplayTrainer},
    },
    Agent, Env as _,
};
use chrono::Local;
use clap::Parser;
use pyo3::Python;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    artifacts_path: PathBuf,
    #[arg(long)]
    env_name: String,
    #[arg(long)]
    restore_path: Option<PathBuf>,
    #[arg(long)]
    batch_size: usize,
    #[arg(long)]
    prioritized: bool,
    #[arg(long)]
    dueling: bool,
    #[arg(long)]
    double_dqn: bool,
    #[arg(long)]
    noisy: bool,
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
        let mut env = GymSuperMarioBrosEnv::new(py, &args.env_name)
            .with_context(|| "create gymnasium env")?;

        let model = DeepQNetworkModel::<AutodiffBackend>::new(
            &device,
            env.observation_space(),
            env.action_space(),
            args.dueling,
            args.noisy,
        );
        let optimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .with_epsilon(0.01 / args.batch_size as f32)
            .init();

        let mut agent = DeepQNetworkAgent::new(
            model,
            optimizer,
            ConstantLr::new(0.00025),
            env.observation_space().clone(),
            env.action_space().clone(),
            device,
            1000,
            args.double_dqn,
        );

        if let Some(restore_path) = &args.restore_path {
            agent.load(restore_path).with_context(|| "load agent")?;
        }

        if args.prioritized {
            let mut memory = PrioritizedReplayMemory::new(2usize.pow(20), args.batch_size, 0.6)?;

            let trainer = PrioritizedReplayTrainer::new(10000, 0.99, artifacts_path, true)?;

            trainer.train_loop(&mut agent, &mut env, &mut memory)?;
        } else {
            let mut memory = UniformReplayMemory::new(2usize.pow(20), args.batch_size)?;

            let trainer =
                UniformReplayTrainer::new(10000, 0.99, 1.0, 0.01, 0.99, artifacts_path, true)?;

            trainer.train_loop(&mut agent, &mut env, &mut memory)?;
        }

        Ok(())
    })?;

    Ok(())
}