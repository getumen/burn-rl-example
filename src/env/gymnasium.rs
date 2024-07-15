use anyhow::Context as _;
use pyo3::{
    types::{IntoPyDict as _, PyAnyMethods as _, PyTypeMethods as _},
    Bound, PyAny, Python,
};

use crate::{Action, ActionSpace, Env, ObservationSpace};

pub struct GymnasiumEnv<'py> {
    py: Python<'py>,
    env: Bound<'py, PyAny>,
    action_space: ActionSpace,
    observation_space: ObservationSpace,
}

impl<'py> GymnasiumEnv<'py> {
    pub fn new(py: Python<'py>, env_name: &str) -> anyhow::Result<Self> {
        let gym = py.import_bound("gymnasium")?;
        let make_func = gym.getattr("make")?;
        let kwargs = [("render_mode", "human")].into_py_dict_bound(py);
        let env = make_func
            .call((env_name,), Some(&kwargs))
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

impl<'py> Env for GymnasiumEnv<'py> {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &ObservationSpace {
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

    use crate::{env::gymnasium::GymnasiumEnv, Action, Env as _, ObservationSpace};

    use super::*;

    #[test]
    fn test_gym_env_discrete() -> anyhow::Result<()> {
        for (env_name, action_space, observation_space) in [
            (
                "Acrobot-v1",
                ActionSpace::Discrete(3),
                ObservationSpace::Box { shape: vec![6] },
            ),
            (
                "CartPole-v1",
                ActionSpace::Discrete(2),
                ObservationSpace::Box { shape: vec![4] },
            ),
            (
                "MountainCar-v0",
                ActionSpace::Discrete(3),
                ObservationSpace::Box { shape: vec![2] },
            ),
        ] {
            let _result: anyhow::Result<()> = Python::with_gil(|py| {
                let mut env = GymnasiumEnv::new(py, env_name)?;
                assert_eq!(env.action_space(), &action_space);
                assert_eq!(env.observation_space(), &observation_space);
                let observation = env.reset()?;
                assert_eq!(
                    observation.len(),
                    observation_space.shape().iter().product::<i64>() as usize
                );
                let (observation, reward, _is_done) = env.step(&Action::Discrete(0))?;
                assert_eq!(
                    observation.len(),
                    observation_space.shape().iter().product::<i64>() as usize
                );
                // check that experience is printed
                // because pyo3 failed silently
                println!("{}", reward);
                Ok(())
            });
        }

        Ok(())
    }
}
