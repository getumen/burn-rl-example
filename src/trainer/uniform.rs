use std::{
    fs::File,
    io::{Cursor, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::Receiver,
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
}; // Add the Write trait import

use anyhow::Context as _;
use rand::Rng as _;
use rocksdb::{DBCompressionType, DBWithThreadMode, MultiThreaded, Options};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use tempfile::TempDir;

use crate::{Action, ActionSpace, Agent, Env, Experience, State};

use super::{NStepExperience, RandomPolicy};

pub struct UniformReplayTrainer {
    episode: usize,
    gamma: f32,
    artifacts_dir: PathBuf,
    render: bool,
}

impl UniformReplayTrainer {
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
        agent: &mut impl Agent<S>,
        env: &mut impl Env<D>,
        memory: &mut UniformReplayMemory<S>,
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
    _buffer_dir: TempDir,
    buffer: Arc<DBWithThreadMode<MultiThreaded>>,
    nstep_experiences: NStepExperience<S>,
    max_buffer_size: usize,
    counter: Arc<AtomicUsize>,
    batch_channel: Receiver<Vec<Experience<S>>>,
    _samplers: Vec<JoinHandle<()>>,
}

impl<S: State + Serialize + DeserializeOwned + 'static> Drop for UniformReplayMemory<S> {
    fn drop(&mut self) {
        let _ = DBWithThreadMode::<MultiThreaded>::destroy(&Options::default(), self.buffer.path());
    }
}

impl<S: State + Serialize + DeserializeOwned + 'static> UniformReplayMemory<S> {
    pub fn new(
        max_buffer_size: usize,
        batch_size: usize,
        n_step: usize,
        gamma: f32,
    ) -> anyhow::Result<Self> {
        let dir = TempDir::new()?;
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.optimize_for_point_lookup(4 * 1024);
        opts.set_compression_type(DBCompressionType::Zstd);
        let buffer = Arc::new(DBWithThreadMode::<MultiThreaded>::open(&opts, dir.path())?);
        let counter = Arc::new(AtomicUsize::new(0));

        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        let sampler_num = 4;
        let mut _samplers = Vec::with_capacity(sampler_num);

        for _ in 0..sampler_num {
            let buffer_clone = buffer.clone();
            let counter_clone = counter.clone();

            let tx_clone = tx.clone();
            let _sampler = std::thread::spawn(move || {
                while counter_clone.load(Ordering::Relaxed) < batch_size {
                    std::thread::sleep(Duration::from_millis(100));
                }
                loop {
                    let batch = {
                        let mut batch = Vec::new();
                        let counter = counter_clone.load(Ordering::Relaxed);

                        let len = if counter > max_buffer_size {
                            max_buffer_size
                        } else {
                            counter
                        };
                        let indexes = (0..batch_size)
                            .map(|_| rand::thread_rng().gen_range(0..len))
                            .collect::<Vec<_>>();
                        for index in indexes {
                            let experience =
                                if let Some(v) = buffer_clone.get(index.to_le_bytes()).unwrap() {
                                    rmp_serde::from_read(Cursor::new(v)).unwrap()
                                } else {
                                    println!("invalid index: {}", index);
                                    continue;
                                };
                            batch.push(experience);
                        }
                        batch
                    };
                    tx_clone.send(batch).unwrap();
                }
            });
        }

        Ok(Self {
            _buffer_dir: dir,
            buffer,
            nstep_experiences: NStepExperience::new(n_step, gamma),
            max_buffer_size,
            counter,
            batch_channel: rx,
            _samplers,
        })
    }

    fn push(&mut self, experience: Experience<S>) -> anyhow::Result<()> {
        let experience = self.nstep_experiences.push(experience)?;
        let experience = if let Some(v) = experience {
            v
        } else {
            return Ok(());
        };

        let value = rmp_serde::to_vec(&experience)?;
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.max_buffer_size;
        self.buffer.put(index.to_le_bytes(), value)?;

        Ok(())
    }

    fn sample(&self) -> anyhow::Result<Vec<Experience<S>>> {
        self
            .batch_channel
            .try_recv()
            .with_context(|| "recv batch")
    }
}
