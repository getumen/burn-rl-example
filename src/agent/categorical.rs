use std::{fmt::Display, fs::File, path::Path};

use crate::{
    batch::DeepQNetworkBathcer, Action, ActionSpace, Agent, DeepQNetworkState, Distributional,
    Estimator, Experience, ObservationSpace, PrioritizedReplay, PrioritizedReplayAgent,
};
use anyhow::{anyhow, Context};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    lr_scheduler::LrScheduler,
    module::{AutodiffModule, ParamId},
    nn::loss::{HuberLossConfig, MseLoss},
    optim::{
        adaptor::OptimizerAdaptor,
        record::{AdaptorRecord, AdaptorRecordItem},
        GradientsParams, Optimizer, SimpleOptimizer,
    },
    prelude::Backend,
    record::{CompactRecorder, HalfPrecisionSettings, Record, Recorder},
    tensor::{backend::AutodiffBackend, ElementConversion, Shape, Tensor, TensorData},
};

use super::LossFunction;

#[derive(Debug, Config)]
pub struct CategoricalDeepQNetworkAgentConfig {
    teacher_update_freq: usize,
    n_step: usize,
    double_dqn: bool,
    min_value: f32,
    max_value: f32,
    loss_function: LossFunction,
}

#[derive(Clone)]
pub struct CategoricalDeepQNetworkAgent<
    B: AutodiffBackend,
    const D: usize,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B>,
> {
    model: M,
    teacher_model: M,
    optimizer: OptimizerAdaptor<O, M, B>,
    lr_scheduler: S,
    observation_space: ObservationSpace<D>,
    action_space: ActionSpace,
    device: B::Device,
    update_counter: usize,

    config: CategoricalDeepQNetworkAgentConfig,
}

impl<
        B: AutodiffBackend,
        const D: usize,
        M: AutodiffModule<B> + Estimator<B> + Distributional<B>,
        O: SimpleOptimizer<B::InnerBackend>,
        S: LrScheduler<B>,
    > CategoricalDeepQNetworkAgent<B, D, M, O, S>
{
    pub fn new(
        model: M,
        optimizer: OptimizerAdaptor<O, M, B>,
        lr_scheduler: S,
        observation_space: ObservationSpace<D>,
        action_space: ActionSpace,
        device: B::Device,

        config: CategoricalDeepQNetworkAgentConfig,
    ) -> Self {
        let teacher_model = model.clone().fork(&device);
        Self {
            model,
            teacher_model,
            optimizer,
            lr_scheduler,
            observation_space,
            action_space,
            device,
            update_counter: 0,
            config,
        }
    }
}

impl<B, const D: usize, M, O, S> PrioritizedReplay<DeepQNetworkState>
    for CategoricalDeepQNetworkAgent<B, D, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B> + Distributional<B>,
    M::InnerModule: Estimator<B::InnerBackend> + Distributional<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
    fn temporaral_difference_error(
        &self,
        gamma: f32,
        experiences: &[Experience<DeepQNetworkState>],
    ) -> anyhow::Result<Vec<f32>> {
        let batcher = DeepQNetworkBathcer::new(self.device.clone(), self.action_space);

        let mut shape = *self.observation_space.shape();
        shape[0] = experiences.len();

        let model = self.model.clone();
        let item = batcher.batch(experiences.to_vec());
        let observation = item.observation.clone();
        let q_value = model.predict(observation.reshape(shape));
        let next_target_q_value = self
            .teacher_model
            .valid()
            .predict(item.next_observation.clone().inner().reshape(shape));
        let next_target_q_value = match self.action_space {
            ActionSpace::Discrete(num_class) => {
                if self.config.double_dqn {
                    let next_q_value = model
                        .valid()
                        .predict(item.next_observation.clone().inner().reshape(shape));
                    let next_actions = next_q_value.argmax(1);
                    next_target_q_value
                        .gather(1, next_actions)
                        .repeat_dim(1, num_class as usize)
                } else {
                    next_target_q_value
                        .max_dim(1)
                        .repeat_dim(1, num_class as usize)
                }
            }
        };
        let next_target_q_value: Tensor<B, 2> =
            Tensor::from_inner(next_target_q_value).to_device(&self.device);

        let targets = next_target_q_value
            .clone()
            .inner()
            .mul_scalar(gamma.powi(self.config.n_step as i32))
            * (item.done.ones_like().inner() - item.done.clone().inner())
            + item.reward.clone().inner();

        let targets = q_value.clone().inner()
            * (item.action.ones_like().inner() - item.action.clone().inner())
            + targets * item.action.clone().inner();

        let td = match self.config.loss_function {
            LossFunction::Huber => HuberLossConfig::new(1.0)
                .init()
                .forward_no_reduction(q_value.inner(), targets),
            LossFunction::Squared => MseLoss::new().forward_no_reduction(q_value.inner(), targets),
        };

        let td: Vec<f32> = td
            .sum_dim(1)
            .into_data()
            .to_vec()
            .map_err(|e| anyhow!("Failed to calculate temporal difference error: {:?}", e))?;
        Ok(td)
    }
}

impl<B, const D: usize, M, O, S> Agent<DeepQNetworkState>
    for CategoricalDeepQNetworkAgent<B, D, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B> + Distributional<B>,
    M::InnerModule: Estimator<B::InnerBackend> + Distributional<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
    fn policy(&self, observation: &[f32]) -> Action {
        let shape = *self.observation_space.shape();
        let feature: Tensor<<B as AutodiffBackend>::InnerBackend, D> = Tensor::from_data(
            TensorData::new(observation.to_vec(), Shape::new(shape)).convert::<B::FloatElem>(),
            &self.device,
        );
        let scores = self.model.valid().predict(feature);
        println!("score: {:?}", scores.to_data().to_vec::<f32>());
        match self.action_space {
            ActionSpace::Discrete(..) => {
                let scores = scores.argmax(1);
                let scores = scores.flatten::<1>(0, 1).into_scalar();
                Action::Discrete(scores.elem())
            }
        }
    }

    fn update(
        &mut self,
        gamma: f32,
        experiences: &[Experience<DeepQNetworkState>],
        weights: &[f32],
    ) -> anyhow::Result<()> {
        let batcher = DeepQNetworkBathcer::new(self.device.clone(), self.action_space);

        let batch_size = experiences.len();
        let mut shape = *self.observation_space.shape();
        shape[0] = batch_size;

        let model = self.model.clone();
        let item = batcher.batch(experiences.to_vec());
        let next_probs = self
            .teacher_model
            .valid()
            .get_distribution(item.next_observation.clone().inner().reshape(shape));

        let prob_shape = next_probs.shape().dims;
        let num_atoms = prob_shape[2];

        let loss = match self.action_space {
            ActionSpace::Discrete(..) => {
                let next_actions = if self.config.double_dqn {
                    let next_q_value = model
                        .valid()
                        .predict(item.next_observation.clone().inner().reshape(shape));

                    next_q_value
                        .argmax(1)
                        .reshape([batch_size, 1, 1])
                        .repeat_dim(2, num_atoms)
                } else {
                    let next_q_value = self
                        .teacher_model
                        .valid()
                        .predict(item.next_observation.clone().inner().reshape(shape));

                    next_q_value
                        .argmax(1)
                        .reshape([batch_size, 1, 1])
                        .repeat_dim(2, num_atoms)
                };
                let next_dists = next_probs
                    .clone()
                    .gather(1, next_actions)
                    .reshape([batch_size, num_atoms, 1]);

                let target_probs = shift_and_projection(
                    next_dists,
                    item.reward.clone().inner(),
                    item.done.clone().inner(),
                    ShiftAndProjectionConfig {
                        batch_size,
                        num_atoms,
                        gamma,
                        n_step: self.config.n_step,
                        min_value: self.config.min_value,
                        max_value: self.config.max_value,
                    },
                );
                let target_probs = Tensor::from_inner(target_probs);

                let prob = self
                    .model
                    .get_distribution(item.observation.clone().reshape(shape));
                let prob = prob
                    .gather(
                        1,
                        item.action
                            .clone()
                            .argmax(1)
                            .reshape([batch_size, 1, 1])
                            .repeat_dim(2, num_atoms),
                    )
                    .reshape([batch_size, num_atoms]);

                -target_probs * (prob.clamp_min(1e-14)).log()
            }
        };
        let weights = Tensor::from_data(
            TensorData::new(weights.to_vec(), Shape::new([weights.len(), 1]))
                .convert::<B::FloatElem>(),
            &self.device,
        );
        let loss = loss.sum_dim(1) * weights;
        let loss = loss.mean();
        let grads: <B as AutodiffBackend>::Gradients = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        self.model = self.optimizer.step(self.lr_scheduler.step(), model, grads);

        self.update_counter += 1;
        if self.update_counter % self.config.teacher_update_freq == 0 {
            self.teacher_model = self.model.clone().fork(&self.device);
        }

        Ok(())
    }

    fn make_state(&self, next_observation: &[f32], state: &DeepQNetworkState) -> DeepQNetworkState {
        DeepQNetworkState {
            observation: state.next_observation.clone(),
            next_observation: next_observation.to_vec(),
        }
    }

    fn save<P: AsRef<Path>>(&self, artifacts_dir: P) -> anyhow::Result<()> {
        let artifacts_dir = artifacts_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&artifacts_dir)
            .with_context(|| format!("fail to create {:?}", artifacts_dir))?;
        self.model
            .clone()
            .save_file(artifacts_dir.join("model"), &CompactRecorder::new())
            .with_context(|| "fail to save model")?;
        let optimizer_record = self.optimizer.to_record();
        let optimizer_record = optimizer_record
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.into_item()))
            .collect::<hashbrown::HashMap<String, AdaptorRecordItem<O, B, HalfPrecisionSettings>>>(
            );

        let mut optimizer_file = File::create(artifacts_dir.join("optimizer.mpk"))
            .with_context(|| "create optimizer file")?;

        rmp_serde::encode::write(&mut optimizer_file, &optimizer_record)
            .with_context(|| "Failed to write optimizer record")?;

        let scheduler_record = self.lr_scheduler.to_record();
        let scheduler_record: <<S as LrScheduler<B>>::Record as Record<_>>::Item<
            HalfPrecisionSettings,
        > = scheduler_record.into_item();
        let mut scheduler_file = File::create(artifacts_dir.join("scheduler.mpk"))
            .with_context(|| "create scheduler file")?;
        rmp_serde::encode::write(&mut scheduler_file, &scheduler_record)
            .with_context(|| "Failed to write scheduler record")?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&mut self, restore_dir: P) -> anyhow::Result<()> {
        let restore_dir = restore_dir.as_ref().to_path_buf();
        let model_file = restore_dir.join("model.mpk");
        if model_file.exists() {
            let record = CompactRecorder::new()
                .load(model_file, &self.device)
                .with_context(|| "Failed to load model")?;
            self.model = self.model.clone().load_record(record);
        }
        let optimizer_file = restore_dir.join("optimizer.mpk");
        if optimizer_file.exists() {
            let optimizer_file =
                File::open(optimizer_file).with_context(|| "open optimizer file")?;
            let record: hashbrown::HashMap<String, AdaptorRecordItem<O, B, HalfPrecisionSettings>> =
                rmp_serde::decode::from_read(optimizer_file)
                    .with_context(|| "Failed to read optimizer record")?;
            let record = record
                .into_iter()
                .map(|(k, v)| (ParamId::from(k), AdaptorRecord::from_item(v, &self.device)))
                .collect::<hashbrown::HashMap<_, _>>();
            self.optimizer = self.optimizer.clone().load_record(record);
        }
        let scheduler_file = restore_dir.join("scheduler.mpk");
        if scheduler_file.exists() {
            let scheduler_file =
                File::open(scheduler_file).with_context(|| "open scheduler file")?;
            let record: <<S as LrScheduler<B>>::Record as Record<_>>::Item<HalfPrecisionSettings> =
                rmp_serde::decode::from_read(scheduler_file)
                    .with_context(|| "Failed to read scheduler record")?;
            let record =
                <<S as LrScheduler<B>>::Record as Record<_>>::from_item(record, &self.device);
            self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
        }

        Ok(())
    }
}

impl<B, const D: usize, M, O, S> PrioritizedReplayAgent<DeepQNetworkState>
    for CategoricalDeepQNetworkAgent<B, D, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B> + Distributional<B>,
    M::InnerModule: Estimator<B::InnerBackend> + Distributional<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
}

pub struct ShiftAndProjectionConfig {
    batch_size: usize,
    num_atoms: usize,
    gamma: f32,
    n_step: usize,
    min_value: f32,
    max_value: f32,
}

// https://github.com/Silvicek/distributional-dqn/blob/master/distdeepq/build_graph.py#L292-L317
fn shift_and_projection<B: Backend>(
    next_dists: Tensor<B, 3>, // [batch, num_class, num_atoms]
    rewards: Tensor<B, 2>,    // [batch, num_class]
    dones: Tensor<B, 2>,      // [batch, num_class]
    config: ShiftAndProjectionConfig,
) -> Tensor<B, 2> {
    let device = next_dists.device();
    let z = (0..config.num_atoms)
        .map(|i| {
            config.min_value
                + (config.max_value - config.min_value) * (i as f32)
                    / (config.num_atoms as f32 - 1.0)
        })
        .collect::<Vec<_>>();
    let z = Tensor::from_data(
        TensorData::new(z, Shape::new([1, config.num_atoms])).convert::<B::FloatElem>(),
        &device,
    ); // [1, num_atoms]
    let dz = (config.max_value - config.min_value) / (config.num_atoms as f32 - 1.0);
    let rewards = rewards.clone().mean_dim(1); // [batch, 1]
    let dones = dones.clone().mean_dim(1); // [batch, 1]
    let target =
        rewards + (dones.ones_like() - dones) * config.gamma.powi(config.n_step as i32) * z.clone(); // [batch, num_atoms]
    let target = target.clamp(config.min_value, config.max_value);
    let target = target.reshape([config.batch_size, 1, config.num_atoms]); // [batch_size, 1, num_atoms]
    let z = z // [1, num_atoms]
        .reshape([1, config.num_atoms, 1]); // [1, num_atoms, 1]
    let modify_coefficient = (target - z).abs() / dz; // [batch, num_atoms, num_atoms]
    let modify_coefficient = (modify_coefficient.ones_like() - modify_coefficient).clamp(0.0, 1.0);

    let target_probs = modify_coefficient.matmul(next_dists);

    target_probs.reshape([config.batch_size, config.num_atoms])
}

#[cfg(test)]
mod tests {

    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        tensor::Tensor,
    };

    use crate::agent::categorical::{shift_and_projection, ShiftAndProjectionConfig};

    #[test]
    fn test_shift_and_projection() {
        // spec. is eq.7 in https://arxiv.org/pdf/1707.06887

        type Backend = LibTorch;
        let device = LibTorchDevice::Cpu;

        for (name, num_class, num_atoms, gamma, max_value, min_value, n_step, reward) in vec![
            ("base", 2, 5, 0.5f32, 5.0f32, 0.0f32, 1usize, 1.0f32),
            ("num_class", 12, 4, 0.5f32, 5.0f32, 0.0f32, 1usize, 1.0f32),
            ("num_atoms", 2, 51, 0.5f32, 5.0f32, 0.0f32, 1usize, 1.0f32),
            ("gamma", 2, 5, 0.99f32, 5.0f32, 0.0f32, 1usize, 1.0f32),
            ("max_value", 2, 5, 0.5f32, 10.0f32, 0.0f32, 1usize, 1.0f32),
            ("min_value", 2, 5, 0.5f32, 5.0f32, -0.5f32, 1usize, 1.0f32),
            ("n_step", 2, 5, 0.5f32, 5.0f32, 0.0f32, 3usize, 1.0f32),
            ("reward", 2, 5, 0.5f32, 5.0f32, 0.0f32, 1usize, 0.0f32),
        ] {
            let batch_size = 1;
            let next_dists: Tensor<Backend, 3> =
                Tensor::ones([batch_size, num_atoms, 1], &device) / (num_atoms as f32);
            let rewards = Tensor::ones([batch_size, num_class], &device) * reward;
            let dones = Tensor::zeros([batch_size, num_class], &device);

            // expected

            let next_dist_scalar = 1.0 / (num_atoms as f32);

            let mut expected = vec![0.0; num_atoms];

            let z = (0..num_atoms)
                .map(|i| {
                    min_value + (max_value - min_value) * (i as f32) / (num_atoms as f32 - 1.0)
                })
                .collect::<Vec<_>>();
            let dz = (max_value - min_value) / (num_atoms as f32 - 1.0);

            for z in z {
                let q_value = reward + gamma.powi(n_step as i32) * z;
                let q_value = q_value.clamp(min_value, max_value);
                let q_value_index = (q_value - min_value) / dz;
                let lower_index = q_value_index.floor() as usize;
                let upper_index = q_value_index.ceil() as usize;

                let lower_prob = 1.0 - (q_value_index - lower_index as f32);
                let upper_prob = 1.0 - (upper_index as f32 - q_value_index);

                if lower_index == upper_index {
                    expected[lower_index] += 0.5 * next_dist_scalar;
                    expected[upper_index] += 0.5 * next_dist_scalar;
                } else {
                    expected[lower_index] += lower_prob * next_dist_scalar;
                    expected[upper_index] += upper_prob * next_dist_scalar;
                }
            }

            let result = shift_and_projection(
                next_dists,
                rewards,
                dones,
                ShiftAndProjectionConfig {
                    batch_size,
                    num_atoms,
                    gamma,
                    n_step,
                    min_value,
                    max_value,
                },
            );

            approx::assert_abs_diff_eq!(expected.iter().sum::<f32>(), 1.0f32, epsilon = 1e-6);
            approx::assert_abs_diff_eq!(
                result
                    .clone()
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .sum::<f32>(),
                1.0f32,
                epsilon = 1e-6
            );
            assert_eq!(
                result
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| format!("{}: {:.4}", name, x))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|x| format!("{}: {:.4}", name, x))
                    .collect::<Vec<_>>()
            );
        }
    }
}
