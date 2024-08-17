use std::{fmt::Display, fs::File, path::Path};

use anyhow::Context;
use burn::{
    data::dataloader::batcher::Batcher,
    lr_scheduler::LrScheduler,
    module::{AutodiffModule, ParamId},
    nn::loss::HuberLossConfig,
    optim::{
        adaptor::OptimizerAdaptor,
        record::{AdaptorRecord, AdaptorRecordItem},
        GradientsParams, Optimizer, SimpleOptimizer,
    },
    record::{CompactRecorder, HalfPrecisionSettings, Record, Recorder},
    tensor::{backend::AutodiffBackend, Data, ElementConversion, Shape, Tensor},
};

use crate::{
    batch::DeepQNetworkBathcer, Action, ActionSpace, Agent, DeepQNetworkState, Estimator,
    Experience, PrioritizedReplay, PrioritizedReplayAgent,
};

#[derive(Clone)]
pub struct DeepQNetworkAgent<
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B>,
> {
    model: M,
    teacher_model: M,
    optimizer: OptimizerAdaptor<O, M, B>,
    lr_scheduler: S,
    action_space: ActionSpace,
    device: B::Device,
    update_counter: usize,
    teacher_update_freq: usize,
    double_dqn: bool,
}

impl<
        B: AutodiffBackend,
        M: AutodiffModule<B> + Estimator<B>,
        O: SimpleOptimizer<B::InnerBackend>,
        S: LrScheduler<B>,
    > DeepQNetworkAgent<B, M, O, S>
{
    pub fn new(
        model: M,
        optimizer: OptimizerAdaptor<O, M, B>,
        lr_scheduler: S,
        action_space: ActionSpace,
        device: B::Device,
        teacher_update_freq: usize,
        double_dqn: bool,
    ) -> Self {
        let teacher_model = model.clone().fork(&device);
        Self {
            model,
            teacher_model,
            optimizer,
            lr_scheduler,
            action_space,
            device,
            update_counter: 0,
            teacher_update_freq,
            double_dqn,
        }
    }
}

impl<B, M, O, S> PrioritizedReplay<DeepQNetworkState> for DeepQNetworkAgent<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B>,
    M::InnerModule: Estimator<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
    fn temporaral_difference_error(
        &self,
        gamma: f32,
        experiences: &[Experience<DeepQNetworkState>],
    ) -> Vec<f32> {
        let batcher = DeepQNetworkBathcer::new(self.device.clone(), self.action_space.clone());

        let model = self.model.clone();
        let item = batcher.batch(experiences.to_vec());
        let observation = item.observation.clone();
        let q_value = model.predict(observation);
        let next_target_q_value = self
            .teacher_model
            .valid()
            .predict(item.next_observation.clone().inner());
        let next_target_q_value: Tensor<B, 2> =
            Tensor::from_inner(next_target_q_value).to_device(&self.device);
        let next_target_q_value = match self.action_space {
            ActionSpace::Discrete(num_class) => {
                if self.double_dqn {
                    let next_q_value = model.predict(item.next_observation.clone());
                    let next_actions = next_q_value.argmax(1);
                    next_target_q_value
                        .gather(1, next_actions)
                        .repeat(1, num_class as usize)
                } else {
                    next_target_q_value.max_dim(1).repeat(1, num_class as usize)
                }
            }
        };
        let targets = q_value.clone().inner()
            * (item.action.ones_like().inner() - item.action.clone().inner())
            + ((next_target_q_value.clone().inner()
                * (item.done.ones_like().inner() - item.done.clone().inner()))
            .mul_scalar(gamma)
                + item.reward.clone().inner())
                * item.action.clone().inner();
        let td: Vec<f32> = (q_value.inner() - targets)
            .sum_dim(1)
            .into_data()
            .value
            .iter()
            .map(|x| x.elem::<f32>())
            .map(|x| x.abs())
            .collect();
        td
    }
}

impl<B, M, O, S> Agent<DeepQNetworkState> for DeepQNetworkAgent<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B>,
    M::InnerModule: Estimator<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
    fn policy(&self, observation: &[f32]) -> Action {
        let feature = Tensor::from_data(
            Data::new(observation.to_vec(), Shape::new([1, observation.len()])).convert(),
            &self.device,
        );
        let scores = self.model.valid().predict(feature);
        println!(
            "feature: {:?} score: {:?}",
            observation,
            scores.to_data().value
        );
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
        let batcher = DeepQNetworkBathcer::new(self.device.clone(), self.action_space.clone());

        let model = self.model.clone();
        let item = batcher.batch(experiences.to_vec());
        let observation = item.observation.clone();
        let q_value = model.predict(observation);
        let next_target_q_value = self
            .teacher_model
            .valid()
            .predict(item.next_observation.clone().inner());
        let next_target_q_value: Tensor<B, 2> =
            Tensor::from_inner(next_target_q_value).to_device(&self.device);
        let next_target_q_value = match self.action_space {
            ActionSpace::Discrete(num_class) => {
                if self.double_dqn {
                    let next_q_value = model.predict(item.next_observation.clone());
                    let next_actions = next_q_value.argmax(1);
                    next_target_q_value
                        .gather(1, next_actions)
                        .repeat(1, num_class as usize)
                } else {
                    next_target_q_value.max_dim(1).repeat(1, num_class as usize)
                }
            }
        };
        let targets = q_value.clone().inner()
            * (item.action.ones_like().inner() - item.action.clone().inner())
            + ((next_target_q_value.clone().inner()
                * (item.done.ones_like().inner() - item.done.clone().inner()))
            .mul_scalar(gamma)
                + item.reward.clone().inner())
                * item.action.clone().inner();
        let targets = Tensor::from_inner(targets);
        let loss = HuberLossConfig::new(1.0)
            .init(&self.device)
            .forward_no_reduction(q_value, targets);
        let weights = Tensor::from_data(
            Data::new(weights.to_vec(), Shape::new([weights.len(), 1])).convert(),
            &self.device,
        );
        let loss = loss.sum_dim(1) * weights;
        let loss = loss.mean();
        let grads: <B as AutodiffBackend>::Gradients = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        self.model = self.optimizer.step(self.lr_scheduler.step(), model, grads);

        self.update_counter += 1;
        if self.update_counter % self.teacher_update_freq == 0 {
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

impl<B, M, O, S> PrioritizedReplayAgent<DeepQNetworkState> for DeepQNetworkAgent<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Display + Estimator<B>,
    M::InnerModule: Estimator<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend>,
    S: LrScheduler<B> + Clone,
{
}
