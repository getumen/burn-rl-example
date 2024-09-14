use std::{
    fs::File,
    io::{Cursor, Write as _},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::Receiver,
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use crate::{Action, ActionSpace, Env, Experience, PrioritizedReplayAgent, State};

use anyhow::Context as _;
use parking_lot::RwLock;
use rand::Rng as _;
use rocksdb::{DBCompressionType, DBWithThreadMode, MultiThreaded, Options};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use tempfile::TempDir;

use super::{NStepExperience, RandomPolicy};

pub struct PrioritizedReplayTrainer {
    episode: usize,
    gamma: f32,
    artifacts_dir: PathBuf,
    render: bool,
}

impl PrioritizedReplayTrainer {
    pub fn new(
        episode: usize,
        gamma: f32,
        artifacts_dir: PathBuf,
        render: bool,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&artifacts_dir).with_context(|| "create artifact dir")?;
        Ok(Self {
            episode,
            gamma,
            artifacts_dir,
            render,
        })
    }

    pub fn train_loop<S: State + Serialize + DeserializeOwned + 'static, const D: usize>(
        &self,
        agent: &mut impl PrioritizedReplayAgent<S>,
        env: &mut impl Env<D>,
        memory: &mut PrioritizedReplayMemory<S>,
        random_policy: &Option<RandomPolicy>,
    ) -> anyhow::Result<()> {
        for epi in 0..self.episode {
            let mut step = 0;
            let mut cumulative_reward = 0.0;
            let mut observation = env.reset()?;
            let mut reward;
            let mut is_done = false;
            let mut state = S::new(observation.clone());
            let episode_artifacts_dir = self.artifacts_dir.join(format!("{}", epi));
            std::fs::create_dir_all(&episode_artifacts_dir).with_context(|| {
                format!("create episode artifact dir {:?}", episode_artifacts_dir)
            })?;
            let train_log_path = episode_artifacts_dir.join("train.jsonl");
            let mut train_logger =
                File::create(&train_log_path).with_context(|| "create train log file")?;
            while !is_done {
                step += 1;

                if self.render {
                    env.render()?;
                }

                let action = if let Some(policy) = random_policy {
                    if policy.sample(epi) {
                        let action_space = env.action_space();
                        match action_space {
                            ActionSpace::Discrete(n) => {
                                let action = rand::thread_rng().gen_range(0..*n);
                                Action::Discrete(action)
                            }
                        }
                    } else {
                        agent.policy(&observation)
                    }
                } else {
                    agent.policy(&observation)
                };

                (observation, reward, is_done) = env.step(&action)?;
                state = agent.make_state(&observation, &state);
                cumulative_reward += reward;
                let log = json!({
                    "episode": epi,
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward
                });
                writeln!(&mut train_logger, "{}", log).with_context(|| "write train log")?;

                let experience = Experience {
                    state: state.clone(),
                    action,
                    reward,
                    is_done,
                };

                memory.push(experience)?;

                if epi == 0 {
                    continue;
                }
                if let Ok(batch) = memory.sample() {
                    agent.update(self.gamma, &batch.experiences, &batch.weights)?;
                    let td_errors =
                        agent.temporaral_difference_error(self.gamma, &batch.experiences)?;
                    let (indexes, td_errors) = batch
                        .indexes
                        .iter()
                        .zip(td_errors)
                        .filter(|(_, x)| !x.is_nan())
                        .unzip();
                    memory.update_priorities(indexes, td_errors);
                }
            }
            agent.save(&episode_artifacts_dir)?;
        }
        Ok(())
    }
}

pub struct PrioritizedBatch<S: State> {
    indexes: Vec<usize>,
    experiences: Vec<Experience<S>>,
    weights: Vec<f32>,
}

pub struct PrioritizedReplayMemory<S: State + Serialize + DeserializeOwned + 'static> {
    _buffer_dir: TempDir,
    buffer: Arc<DBWithThreadMode<MultiThreaded>>,
    priorities: Arc<RwLock<SumTree>>,
    nstep_experiences: NStepExperience<S>,
    max_buffer_size: usize,
    counter: Arc<AtomicUsize>,
    alpha: f32,

    max_priority: f32,

    batch_channel: Receiver<PrioritizedBatch<S>>,
    _samplers: Vec<JoinHandle<()>>,
}

impl<S: State + Serialize + DeserializeOwned + 'static> Drop for PrioritizedReplayMemory<S> {
    fn drop(&mut self) {
        let _ = DBWithThreadMode::<MultiThreaded>::destroy(&Options::default(), self.buffer.path());
    }
}

impl<S: State + Serialize + DeserializeOwned + 'static> PrioritizedReplayMemory<S> {
    pub fn new(
        max_buffer_size: usize,
        batch_size: usize,
        n_step: usize,
        alpha: f32,
        gamma: f32,
    ) -> anyhow::Result<Self> {
        let dir = TempDir::new()?;
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.optimize_for_point_lookup(4 * 1024);
        opts.set_compression_type(DBCompressionType::Zstd);
        let buffer = Arc::new(DBWithThreadMode::<MultiThreaded>::open(&opts, dir.path())?);

        let counter = Arc::new(AtomicUsize::new(0));
        let priorities = Arc::new(RwLock::new(SumTree::new(max_buffer_size)));

        let (tx, rx) = std::sync::mpsc::sync_channel(4);

        let sampler_num = 4;
        let mut _samplers = Vec::with_capacity(sampler_num);

        for _ in 0..sampler_num {
            let tx_clone = tx.clone();
            let buffer_clone = buffer.clone();
            let priorities_clone = priorities.clone();
            let counter_clone = counter.clone();

            let _sampler = std::thread::spawn(move || {
                while counter_clone.load(Ordering::Relaxed) < batch_size {
                    std::thread::sleep(Duration::from_millis(100));
                }
                loop {
                    let (indexes, experiences, weights) = {
                        let priorities: Vec<(usize, f32)> = {
                            let priorities = priorities_clone.read();
                            (0..batch_size).map(|_| priorities.sample()).collect()
                        };
                        let mut indexes = Vec::with_capacity(batch_size);
                        let mut experiences = Vec::with_capacity(batch_size);
                        let mut weights = Vec::with_capacity(batch_size);
                        for (index, weight) in priorities {
                            let experience =
                                if let Some(v) = buffer_clone.get(index.to_le_bytes()).unwrap() {
                                    rmp_serde::from_read(Cursor::new(v)).unwrap()
                                } else {
                                    println!("invalid index: {}", index);
                                    continue;
                                };
                            indexes.push(index);
                            experiences.push(experience);
                            weights.push(weight);
                        }
                        (indexes, experiences, weights)
                    };
                    tx_clone
                        .send(PrioritizedBatch {
                            indexes,
                            experiences,
                            weights,
                        })
                        .unwrap();
                }
            });
            _samplers.push(_sampler);
        }

        Ok(Self {
            _buffer_dir: dir,
            buffer,
            priorities,
            nstep_experiences: NStepExperience::new(n_step, gamma),
            max_buffer_size,
            counter,
            alpha,
            max_priority: 1.0,
            batch_channel: rx,
            _samplers,
        })
    }

    pub fn update_priorities(&mut self, indexes: Vec<usize>, td_errors: Vec<f32>) {
        for (index, td_error) in indexes.iter().zip(td_errors) {
            self.max_priority = self.max_priority.max(td_error);
            let priority = (td_error + 0.001).powf(self.alpha);
            let mut priorities = self.priorities.write();
            priorities.set(*index, priority);
        }
    }

    fn push(&mut self, experience: Experience<S>) -> anyhow::Result<()> {
        let experience = self.nstep_experiences.push(experience)?;
        let experience = if let Some(v) = experience {
            v
        } else {
            return Ok(());
        };

        let priority = (self.max_priority + 0.001).powf(self.alpha);
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.max_buffer_size;
        let value = rmp_serde::to_vec(&experience)?;

        let mut priorities = self.priorities.write();
        self.buffer.put(index.to_le_bytes(), value)?;
        priorities.set(index, priority);

        Ok(())
    }

    pub fn sample(&self) -> anyhow::Result<PrioritizedBatch<S>> {
        self.batch_channel.try_recv().with_context(|| "recv batch")
    }
}

struct SumTree {
    data: Vec<f32>,
    capacity: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        assert_eq!(capacity.count_ones(), 1, "capacity must be a power of 2");
        let data = vec![0.0f32; 2 * capacity];
        Self { data, capacity }
    }

    pub fn set(&mut self, index: usize, priority: f32) {
        let tree_index = index + self.capacity;
        self.data[tree_index] = priority;
        let mut parent = tree_index / 2;
        while parent > 0 {
            self.data[parent] = self.data[2 * parent] + self.data[2 * parent + 1];
            debug_assert!(
                !self.data[parent].is_nan(),
                "parent: {}, left child: {}, right child: {}, index: {}, priority: {}",
                self.data[parent],
                self.data[2 * parent],
                self.data[2 * parent + 1],
                index,
                priority,
            );
            parent /= 2;
        }
    }

    pub fn total(&self) -> f32 {
        self.data[1]
    }

    pub fn sample(&self) -> (usize, f32) {
        let max: f32 = self.total();
        let mut value = rand::thread_rng().gen_range(0.0..max);
        let mut cur = 1;
        while cur < self.capacity {
            let left = 2 * cur;
            let right = 2 * cur + 1;
            if value > self.data[left] {
                cur = right;
                value -= self.data[left];
            } else {
                cur = left;
            }
        }
        (cur - self.capacity, self.data[cur])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_tree() {
        let sample_num = 10000;

        let mut sum_tree = SumTree::new(4);
        sum_tree.set(0, 1.0);
        sum_tree.set(1, 2.0);
        sum_tree.set(2, 3.0);
        sum_tree.set(3, 4.0);
        assert_eq!(sum_tree.total(), 10.0);
        let mut count = [0; 4];
        for _ in 0..sample_num {
            let (index, _) = sum_tree.sample();
            count[index] += 1;
        }
        for (i, count) in count.iter().enumerate() {
            assert!(
                (*count as f32 - sample_num as f32 * (i + 1) as f32 / 10.0).abs()
                    < sample_num as f32 / 10.0
            );
        }
    }
}
