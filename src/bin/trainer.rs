use std::path::PathBuf;

use anyhow::Context;
use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    lr_scheduler::constant::ConstantLr,
    optim::AdamConfig,
};
use burn_rl_example::{
    agent::{
        categorical::CategoricalDeepQNetworkAgent, expectation::DeepQNetworkAgent,
        quantile::QuantileRegressionAgent,
    },
    env::{
        gym_super_mario_bros::GymSuperMarioBrosEnv,
        gymnasium::{GymnasiumEnv1D, GymnasiumEnv3D},
    },
    model::{DeepQNetworkModel, OutputLayerConfig},
    trainer::{
        prioritized::{PrioritizedReplayMemory, PrioritizedReplayTrainer},
        uniform::{UniformReplayMemory, UniformReplayTrainer},
        RandomPolicy,
    },
    Agent, Env,
};
use chrono::Local;
use clap::{Parser, ValueEnum};
use pyo3::Python;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(value_enum)]
    distributional: Distributional,
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
    #[arg(long)]
    render: bool,
    #[arg(long)]
    n_step: usize,
    #[arg(long)]
    bellman_gamma: f32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Distributional {
    Expectation,
    Categorical,
    Quantile,
}

impl Distributional {
    fn to_output_layer_config(&self, _env_name: &str) -> OutputLayerConfig {
        match self {
            Distributional::Expectation => OutputLayerConfig::Expectation,
            Distributional::Categorical => OutputLayerConfig::CategoricalDistribution {
                atoms: 51,
                min_value: -250.0,
                max_value: 250.0,
            },
            Distributional::Quantile => OutputLayerConfig::QuantileRegression { quantiles: 51 },
        }
    }
}

fn run<const D: usize>(env: &mut impl Env<D>, args: Args) -> anyhow::Result<()> {
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

    let output_layer_config = args.distributional.to_output_layer_config(&args.env_name);

    let model = DeepQNetworkModel::<AutodiffBackend>::new(
        &device,
        env.observation_space(),
        env.action_space(),
        args.dueling,
        args.noisy,
        output_layer_config.clone(),
    );

    let optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
        .with_epsilon(0.01 / args.batch_size as f32)
        .init();

    let random_policy = if !args.noisy {
        Some(RandomPolicy::new(1.0, 0.01, 0.99))
    } else {
        None
    };

    match output_layer_config {
        OutputLayerConfig::Expectation => {
            let mut agent = DeepQNetworkAgent::new(
                model,
                optimizer,
                ConstantLr::new(0.00025),
                env.observation_space().clone(),
                *env.action_space(),
                device,
                1000,
                args.n_step,
                args.double_dqn,
            );

            if let Some(restore_path) = &args.restore_path {
                agent.load(restore_path).with_context(|| "load agent")?;
            }

            if args.prioritized {
                let mut memory = PrioritizedReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    0.6,
                    args.bellman_gamma,
                )?;

                let trainer =
                    PrioritizedReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            } else {
                let mut memory = UniformReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    args.bellman_gamma,
                )?;

                let trainer =
                    UniformReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            }
        }
        OutputLayerConfig::CategoricalDistribution {
            min_value,
            max_value,
            ..
        } => {
            let mut agent = CategoricalDeepQNetworkAgent::new(
                model,
                optimizer,
                ConstantLr::new(0.00025),
                env.observation_space().clone(),
                *env.action_space(),
                device,
                1000,
                args.n_step,
                args.double_dqn,
                min_value,
                max_value,
            );

            if let Some(restore_path) = &args.restore_path {
                agent.load(restore_path).with_context(|| "load agent")?;
            }

            if args.prioritized {
                let mut memory = PrioritizedReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    0.6,
                    args.bellman_gamma,
                )?;

                let trainer =
                    PrioritizedReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            } else {
                let mut memory = UniformReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    args.bellman_gamma,
                )?;

                let trainer =
                    UniformReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            }
        }
        OutputLayerConfig::QuantileRegression { .. } => {
            let mut agent = QuantileRegressionAgent::new(
                model,
                optimizer,
                ConstantLr::new(0.00025),
                env.observation_space().clone(),
                *env.action_space(),
                device,
                1000,
                args.n_step,
                args.double_dqn,
            );

            if let Some(restore_path) = &args.restore_path {
                agent.load(restore_path).with_context(|| "load agent")?;
            }

            if args.prioritized {
                let mut memory = PrioritizedReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    0.6,
                    args.bellman_gamma,
                )?;

                let trainer =
                    PrioritizedReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            } else {
                let mut memory = UniformReplayMemory::new(
                    2usize.pow(20),
                    args.batch_size,
                    args.n_step,
                    args.bellman_gamma,
                )?;

                let trainer =
                    UniformReplayTrainer::new(10000, args.bellman_gamma, artifacts_path, true)?;

                trainer.train_loop(&mut agent, env, &mut memory, &random_policy)?;
            }
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    Python::with_gil(|py| -> anyhow::Result<()> {
        let env_1d = [
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            "Pendulum-v0",
            "LunarLander-v2",
        ];
        let env_3d = ["Breakout-v4"];

        let super_mario_env = [
            "SuperMarioBros-v0",
            "SuperMarioBros-v1",
            "SuperMarioBros-v2",
            "SuperMarioBros-v3",
        ];

        let args = Args::parse();

        if super_mario_env.contains(&args.env_name.as_str()) {
            let mut env = GymSuperMarioBrosEnv::new(py, &args.env_name, args.render)
                .with_context(|| "create gymnasium env")?;
            run(&mut env, args)?;
        } else if env_1d.contains(&args.env_name.as_str()) {
            let mut env = GymnasiumEnv1D::new(py, &args.env_name, args.render)
                .with_context(|| "create gymnasium env")?;

            run(&mut env, args)?;
        } else if env_3d.contains(&args.env_name.as_str()) {
            let mut env = GymnasiumEnv3D::new(py, &args.env_name, args.render)
                .with_context(|| "create gymnasium env")?;
            run(&mut env, args)?;
        }
        Ok(())
    })?;

    Ok(())
}
