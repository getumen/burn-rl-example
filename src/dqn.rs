use std::{fmt::Display, fs::File, path::Path};

use anyhow::Context;
use burn::{
    data::dataloader::batcher::Batcher,
    lr_scheduler::LrScheduler,
    module::{AutodiffModule, ParamId},
    nn::loss::{HuberLossConfig, Reduction},
    optim::{
        adaptor::OptimizerAdaptor,
        record::{AdaptorRecord, AdaptorRecordItem},
        GradientsParams, Optimizer, SimpleOptimizer,
    },
    record::{CompactRecorder, HalfPrecisionSettings, Record, Recorder},
    tensor::{backend::AutodiffBackend, Data, ElementConversion, Shape, Tensor},
};

use crate::{Action, ActionSpace, Agent, Estimator, Experience, State};

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
        }
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
    ) -> anyhow::Result<()> {
        let batcher = DeepQNetworkBathcer::new(self.device.clone(), self.action_space.clone());

        let model = self.model.clone();
        let item = batcher.batch(experiences.to_vec());
        let observation = item.observation.clone();
        let q_value = model.predict(observation);
        let next_q_value = self
            .teacher_model
            .valid()
            .predict(item.next_observation.clone().inner());
        let next_q_value: Tensor<B, 2> = Tensor::from_inner(next_q_value).to_device(&self.device);
        let next_q_value = match self.action_space {
            ActionSpace::Discrete(num_class) => {
                next_q_value.max_dim(1).repeat(1, num_class as usize)
            }
        };
        let targets = q_value.clone().inner()
            * (item.action.ones_like().inner() - item.action.clone().inner())
            + ((next_q_value.clone().inner()
                * (item.done.ones_like().inner() - item.done.clone().inner()))
            .mul_scalar(gamma)
                + item.reward.clone().inner())
                * item.action.clone().inner();
        let targets = Tensor::from_inner(targets);
        let loss =
            HuberLossConfig::new(1.0)
                .init(&self.device)
                .forward(q_value, targets, Reduction::Mean);
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

#[derive(Debug, Clone, Default)]
pub struct DeepQNetworkState {
    pub observation: Vec<f32>,
    pub next_observation: Vec<f32>,
}

impl State for DeepQNetworkState {}

#[derive(Debug, Clone)]
pub struct DeepQNetworkBathcer<B: AutodiffBackend> {
    device: B::Device,
    action_space: ActionSpace,
}

impl<B: AutodiffBackend> DeepQNetworkBathcer<B> {
    pub fn new(device: B::Device, action_space: ActionSpace) -> Self {
        Self {
            device,
            action_space,
        }
    }
}

impl<B: AutodiffBackend> Batcher<Experience<DeepQNetworkState>, DeepQNetworkBatch<B>>
    for DeepQNetworkBathcer<B>
{
    fn batch(&self, items: Vec<Experience<DeepQNetworkState>>) -> DeepQNetworkBatch<B> {
        let (observation, next_observation, action, reward, done): (
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
            Vec<Tensor<B, 2>>,
        ) = items
            .into_iter()
            .filter(|x| !x.state.observation.is_empty() && !x.state.next_observation.is_empty())
            .map(|x| {
                let obs_len = x.state.observation.len();
                let next_obs_len = x.state.next_observation.len();
                (
                    Tensor::from_data(
                        Data::new(x.state.observation, Shape::new([1, obs_len])).convert(),
                        &Default::default(),
                    ),
                    Tensor::from_data(
                        Data::new(x.state.next_observation, Shape::new([1, next_obs_len]))
                            .convert(),
                        &Default::default(),
                    ),
                    match (x.action, self.action_space) {
                        (Action::Discrete(value), ActionSpace::Discrete(num_class)) => {
                            Tensor::one_hot(value as usize, num_class as usize, &Default::default())
                        }
                    },
                    match self.action_space {
                        ActionSpace::Discrete(num_class) => Tensor::from_data(
                            Data::new(vec![x.reward], Shape::new([1, 1])).convert(),
                            &Default::default(),
                        )
                        .repeat(1, num_class as usize),
                    },
                    match self.action_space {
                        ActionSpace::Discrete(num_class) => Tensor::from_data(
                            Data::new(vec![x.is_done as i32], Shape::new([1, 1])).convert(),
                            &Default::default(),
                        )
                        .repeat(1, num_class as usize),
                    },
                )
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |(mut a, mut b, mut c, mut d, mut e), (v, w, x, y, z)| {
                    a.push(v);
                    b.push(w);
                    c.push(x);
                    d.push(y);
                    e.push(z);
                    (a, b, c, d, e)
                },
            );

        let observation = Tensor::cat(observation, 0).to_device(&self.device);
        let next_observation = Tensor::cat(next_observation, 0).to_device(&self.device);
        let action = Tensor::cat(action, 0).to_device(&self.device);
        let reward = Tensor::cat(reward, 0).to_device(&self.device);
        let done = Tensor::cat(done, 0).to_device(&self.device);

        DeepQNetworkBatch {
            observation,
            reward,
            action,
            next_observation,
            done,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepQNetworkBatch<B: AutodiffBackend> {
    pub observation: Tensor<B, 2>,
    pub reward: Tensor<B, 2>,
    pub action: Tensor<B, 2>,
    pub next_observation: Tensor<B, 2>,
    pub done: Tensor<B, 2>,
}
