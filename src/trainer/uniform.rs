use std::{fs::File, io::Write, path::PathBuf}; // Add the Write trait import

use anyhow::Context as _;
use rand::Rng as _;
use redb::{Builder, Database, ReadableTableMetadata, TableDefinition};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use tempfile::NamedTempFile;

use crate::{Action, ActionSpace, Agent, Env, Experience, State};

pub struct UniformReplayTrainer {
    episode: usize,
    gamma: f32,
    init_exploration: f32,
    final_exploration: f32,
    exploration_decay: f32,
    artifacts_dir: PathBuf,
    render: bool,
}

impl UniformReplayTrainer {
    pub fn new(
        episode: usize,
        gamma: f32,
        init_exploration: f32,
        final_exploration: f32,
        exploration_decay: f32,
        artifacts_dir: PathBuf,
        render: bool,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&artifacts_dir).with_context(|| "create artifact dir")?;
        Ok(Self {
            episode,
            gamma,
            init_exploration,
            final_exploration,
            exploration_decay,
            artifacts_dir,
            render,
        })
    }

    pub fn train_loop<S: State + Serialize + DeserializeOwned + 'static, const D: usize>(
        &self,
        agent: &mut impl Agent<S>,
        env: &mut impl Env<D>,
        memory: &mut UniformReplayMemory<S>,
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

                let action = if (self.init_exploration * self.exploration_decay.powf(epi as f32))
                    .max(self.final_exploration)
                    < rand::thread_rng().gen::<f32>()
                {
                    agent.policy(&observation)
                } else {
                    let action_space = env.action_space();
                    match action_space {
                        ActionSpace::Discrete(n) => {
                            let action = rand::thread_rng().gen_range(0..*n);
                            Action::Discrete(action)
                        }
                    }
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

                if epi > 0 {
                    let batch = memory.sample()?;
                    let weights = vec![1.0 / batch.len() as f32; batch.len()];
                    agent.update(self.gamma, &batch, &weights)?;
                }
            }
            agent.save(&episode_artifacts_dir)?;
        }
        Ok(())
    }
}

pub struct UniformReplayMemory<S: State + Serialize + DeserializeOwned + 'static> {
    buffer: Database,
    table_definition: TableDefinition<'static, u32, Experience<S>>,
    max_buffer_size: usize,
    batch_size: usize,
    counter: usize,
}

impl<S: State + Serialize + DeserializeOwned + 'static> UniformReplayMemory<S> {
    pub fn new(max_buffer_size: usize, batch_size: usize) -> anyhow::Result<Self> {
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
            max_buffer_size,
            batch_size,
            counter: 0,
        })
    }

    fn push(&mut self, experience: Experience<S>) -> anyhow::Result<()> {
        let write_txn = self.buffer.begin_write()?;
        {
            let mut table = write_txn.open_table(self.table_definition)?;
            table.insert(self.counter as u32, experience)?;
            if self.counter == self.max_buffer_size {
                self.counter = 0;
            }
        }
        write_txn.commit()?;
        self.counter += 1;

        Ok(())
    }

    fn sample(&self) -> anyhow::Result<Vec<Experience<S>>> {
        let mut batch = Vec::new();

        let read_txn = self.buffer.begin_read().unwrap();
        {
            let table = read_txn.open_table(self.table_definition).unwrap();
            let len = table.len()?;
            let indexes = (0..self.batch_size)
                .map(|_| rand::thread_rng().gen_range(0..len))
                .collect::<Vec<_>>();
            for index in indexes {
                let experience = table.get(index as u32)?;
                batch.push(experience.unwrap().value());
            }
        }
        Ok(batch)
    }
}
