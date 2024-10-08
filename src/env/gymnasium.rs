use anyhow::Context as _;
use pyo3::{
    types::{IntoPyDict as _, PyAnyMethods as _, PyTypeMethods as _},
    Bound, PyAny, Python,
};

use crate::{Action, ActionSpace, Env, ObservationSpace};

pub struct GymnasiumEnv1D<'py> {
    py: Python<'py>,
    env: Bound<'py, PyAny>,
    action_space: ActionSpace,
    observation_space: ObservationSpace<2>,
}

impl<'py> GymnasiumEnv1D<'py> {
    pub fn new(py: Python<'py>, env_name: &str, render: bool) -> anyhow::Result<Self> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (".venv/lib/python3.11/site-packages",))
            .with_context(|| "fail to append path. use rye sync")?;

        let gym = py.import_bound("gymnasium")?;
        let make_func = gym.getattr("make")?;
        let mode = [("render_mode", "human")].into_py_dict_bound(py);
        let kwargs = if render { Some(&mode) } else { None };
        let env = make_func
            .call((env_name,), kwargs)
            .with_context(|| "fail to call make function")?;
        let action_space = env
            .getattr("action_space")
            .with_context(|| "fail to get action space")?;

        let action_space = match action_space.get_type().name()?.as_ref() {
            "Discrete" => {
                let n = action_space.getattr("n")?;
                let action_space: i64 = n.extract()?;
                ActionSpace::Discrete(action_space)
            }
            _ => unimplemented!("Unsupported action space"),
        };
        let observation_space = env.getattr("observation_space")?;
        let observation_space = match observation_space.get_type().name()?.as_ref() {
            "Box" => {
                let shape = observation_space.getattr("shape")?;
                let shape: Vec<i64> = shape.extract()?;
                let shape = [1, shape[0] as usize];
                ObservationSpace::Box { shape }
            }
            _ => unimplemented!("Unsupported observation space"),
        };
        Ok(Self {
            py,
            env,
            action_space,
            observation_space,
        })
    }

    fn ndarray_to_vec(array: Bound<PyAny>) -> anyhow::Result<Vec<f32>> {
        let result = array.call_method("reshape", (-1,), None)?;
        let result = result.extract()?;
        Ok(result)
    }
}

impl<'py> Env<2> for GymnasiumEnv1D<'py> {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &ObservationSpace<2> {
        &self.observation_space
    }

    fn reset(&mut self) -> anyhow::Result<Vec<f32>> {
        let result: Bound<'py, PyAny> = self.env.call_method("reset", (), None)?;
        let result = if result.get_type().name()? == "tuple" {
            result.get_item(0)?
        } else {
            result
        };
        Self::ndarray_to_vec(result)
    }

    fn step(&mut self, action: &Action) -> anyhow::Result<(Vec<f32>, f32, bool)> {
        let py_action = action.to_object(self.py);
        let step = self.env.call_method("step", (py_action,), None)?;
        let observation = Self::ndarray_to_vec(step.get_item(0)?)?;
        let reward: f32 = step.get_item(1)?.extract()?;
        let terminated: bool = step.get_item(2)?.extract()?;
        let truncated: bool = step.get_item(3)?.extract()?;
        let done = terminated || truncated;
        Ok((observation, reward, done))
    }

    fn render(&self) -> anyhow::Result<()> {
        self.env.call_method("render", (), None)?;
        Ok(())
    }
}

pub struct GymnasiumEnv3D<'py> {
    py: Python<'py>,
    env: Bound<'py, PyAny>,
    action_space: ActionSpace,
    observation_space: ObservationSpace<4>,
}

impl<'py> GymnasiumEnv3D<'py> {
    pub fn new(py: Python<'py>, env_name: &str, render: bool) -> anyhow::Result<Self> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (".venv/lib/python3.11/site-packages",))
            .with_context(|| "fail to append path. use rye sync")?;

        let gym = py.import_bound("gymnasium")?;
        let make_func = gym.getattr("make")?;
        let mode = [("render_mode", "human")].into_py_dict_bound(py);
        let kwargs = if render { Some(&mode) } else { None };
        let env = make_func
            .call((env_name,), kwargs)
            .with_context(|| "fail to call make function")?;
        let action_space = env
            .getattr("action_space")
            .with_context(|| "fail to get action space")?;

        let action_space = match action_space.get_type().name()?.as_ref() {
            "Discrete" => {
                let n = action_space.getattr("n")?;
                let action_space: i64 = n.extract()?;
                ActionSpace::Discrete(action_space)
            }
            _ => unimplemented!("Unsupported action space"),
        };
        let observation_space = env.getattr("observation_space")?;
        let observation_space = match observation_space.get_type().name()?.as_ref() {
            "Box" => {
                let shape = observation_space.getattr("shape")?;
                let shape: Vec<i64> = shape.extract()?;
                let shape = [1, shape[2] as usize, shape[0] as usize, shape[1] as usize];
                ObservationSpace::Box { shape }
            }
            _ => unimplemented!("Unsupported observation space"),
        };
        Ok(Self {
            py,
            env,
            action_space,
            observation_space,
        })
    }

    fn ndarray_to_vec(array: Bound<PyAny>) -> anyhow::Result<Vec<f32>> {
        let result = array.call_method("reshape", (-1,), None)?;
        let result = result.extract()?;
        Ok(result)
    }
}

impl<'py> Env<4> for GymnasiumEnv3D<'py> {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &ObservationSpace<4> {
        &self.observation_space
    }

    fn reset(&mut self) -> anyhow::Result<Vec<f32>> {
        let result: Bound<'py, PyAny> = self.env.call_method("reset", (), None)?;
        let result = if result.get_type().name()? == "tuple" {
            result.get_item(0)?
        } else {
            result
        };
        Self::ndarray_to_vec(result)
    }

    fn step(&mut self, action: &Action) -> anyhow::Result<(Vec<f32>, f32, bool)> {
        let py_action = action.to_object(self.py);
        let step = self.env.call_method("step", (py_action,), None)?;
        let observation = Self::ndarray_to_vec(step.get_item(0)?)?;
        let reward: f32 = step.get_item(1)?.extract()?;
        let terminated: bool = step.get_item(2)?.extract()?;
        let truncated: bool = step.get_item(3)?.extract()?;
        let done = terminated || truncated;
        Ok((observation, reward, done))
    }

    fn render(&self) -> anyhow::Result<()> {
        self.env.call_method("render", (), None)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use crate::{env::gymnasium::GymnasiumEnv1D, Action, Env as _, ObservationSpace};

    use super::*;

    #[test]
    fn test_gym_env_discrete() -> anyhow::Result<()> {
        for (env_name, action_space, observation_space) in [
            (
                "Acrobot-v1",
                ActionSpace::Discrete(3),
                ObservationSpace::Box { shape: [1, 6] },
            ),
            (
                "CartPole-v1",
                ActionSpace::Discrete(2),
                ObservationSpace::Box { shape: [1, 4] },
            ),
            (
                "MountainCar-v0",
                ActionSpace::Discrete(3),
                ObservationSpace::Box { shape: [1, 2] },
            ),
        ] {
            let _result: anyhow::Result<()> = Python::with_gil(|py| {
                let mut env = GymnasiumEnv1D::new(py, env_name, true)?;
                assert_eq!(env.action_space(), &action_space);
                assert_eq!(env.observation_space(), &observation_space);
                let observation = env.reset()?;
                assert_eq!(observation.len(), observation_space.shape()[1]);
                let (observation, reward, _is_done) = env.step(&Action::Discrete(0))?;
                assert_eq!(observation.len(), observation_space.shape()[1]);
                // check that experience is printed
                // because pyo3 failed silently
                println!("{}", reward);
                Ok(())
            });
        }

        {
            let (env_name, action_space, observation_space) = (
                "Breakout-v4",
                ActionSpace::Discrete(4),
                ObservationSpace::Box {
                    shape: [1, 3, 210, 160],
                },
            );
            let _result: anyhow::Result<()> = Python::with_gil(|py| {
                let mut env = GymnasiumEnv3D::new(py, env_name, true)?;
                assert_eq!(env.action_space(), &action_space);
                assert_eq!(env.observation_space(), &observation_space);
                let observation = env.reset()?;
                assert_eq!(observation.len(), observation_space.shape()[1]);
                let (observation, reward, _is_done) = env.step(&Action::Discrete(0))?;
                assert_eq!(observation.len(), observation_space.shape()[1]);
                // check that experience is printed
                // because pyo3 failed silently
                println!("{}", reward);
                Ok(())
            });
        }

        Ok(())
    }
}
