use std::{fs::File, io::Write as _, path::PathBuf};

use crate::{Env, Experience, PrioritizedReplayAgent, State};

use anyhow::Context as _;
use rand::Rng as _;
use redb::{Builder, Database, TableDefinition};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use tempfile::NamedTempFile;

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

                let action = agent.policy(&observation);

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

                if epi > 0 {
                    let (indexes, batch, weights) = memory.sample()?;
                    agent.update(self.gamma, &batch, &weights)?;
                    let td_errors = agent.temporaral_difference_error(self.gamma, &batch);
                    let (indexes, td_errors) = indexes
                        .into_iter()
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

pub struct PrioritizedReplayMemory<S: State + Serialize + DeserializeOwned + 'static> {
    buffer: Database,
    table_definition: TableDefinition<'static, u32, Experience<S>>,
    priorities: SumTree,
    max_buffer_size: usize,
    batch_size: usize,
    counter: usize,
    alpha: f32,

    max_priority: f32,
}

impl<S: State + Serialize + DeserializeOwned + 'static> PrioritizedReplayMemory<S> {
    pub fn new(max_buffer_size: usize, batch_size: usize, alpha: f32) -> anyhow::Result<Self> {
        let file = NamedTempFile::new().with_context(|| "create temp file")?;
        let db = Builder::new().create(file)?;
        let table_definition: TableDefinition<u32, Experience<S>> =
            TableDefinition::new("experience");

        let wrote_txn = db.begin_write()?;
        {
            wrote_txn.open_table(table_definition)?;
        }
        wrote_txn.commit()?;
        Ok(Self {
            buffer: db,
            table_definition,
            priorities: SumTree::new(max_buffer_size),
            max_buffer_size,
            batch_size,
            counter: 0,
            alpha,

            max_priority: 1.0,
        })
    }

    pub fn update_priorities(&mut self, indexes: Vec<usize>, td_errors: Vec<f32>) {
        for (index, td_error) in indexes.iter().zip(td_errors) {
            self.max_priority = self.max_priority.max(td_error);
            let priority = (td_error + 0.001).powf(self.alpha);
            self.priorities.set(*index, priority);
        }
    }

    fn push(&mut self, experience: Experience<S>) -> anyhow::Result<()> {
        let write_txn = self.buffer.begin_write()?;
        {
            let mut table = write_txn.open_table(self.table_definition)?;
            let priority = (self.max_priority + 0.001).powf(self.alpha);
            if self.counter == self.max_buffer_size {
                self.counter = 0;
            }
            table.insert(self.counter as u32, experience)?;
            self.priorities.set(self.counter, priority);
            self.counter += 1;
        }
        write_txn.commit()?;

        Ok(())
    }

    pub fn sample(&self) -> anyhow::Result<(Vec<usize>, Vec<Experience<S>>, Vec<f32>)> {
        let read_txn = self.buffer.begin_read()?;
        let table = read_txn.open_table(self.table_definition)?;
        let mut indexes = Vec::with_capacity(self.batch_size);
        let mut experiences = Vec::with_capacity(self.batch_size);
        let mut weights = Vec::with_capacity(self.batch_size);
        let mut counter = 0;
        while counter < self.batch_size {
            let (index, weight) = self.priorities.sample();
            let experience = if let Some(v) = table.get(index as u32).unwrap() {
                v.value()
            } else {
                println!("invalid index: {}", index);
                continue;
            };
            indexes.push(index);
            experiences.push(experience);
            weights.push(weight);

            counter += 1;
        }
        Ok((indexes, experiences, weights))
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
            assert!(!self.data[parent].is_nan());
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
    use claim::assert_lt;

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
        for i in 0..4 {
            assert_lt!(
                (count[i] as f32 - sample_num as f32 * (i + 1) as f32 / 10.0).abs(),
                sample_num as f32 / 10.0
            );
        }
    }
}
