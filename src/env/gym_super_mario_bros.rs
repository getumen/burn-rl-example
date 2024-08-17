use anyhow::Context as _;
use pyo3::{
    types::{PyAnyMethods as _, PyTypeMethods as _},
    Bound, PyAny, Python,
};

use crate::{Action, ActionSpace, Env, ObservationSpace};

pub struct GymSuperMarioBrosEnv<'py> {
    py: Python<'py>,
    env: Bound<'py, PyAny>,
    action_space: ActionSpace,
    observation_space: ObservationSpace<4>,
}

impl<'py> GymSuperMarioBrosEnv<'py> {
    pub fn new(py: Python<'py>, env_name: &str) -> anyhow::Result<Self> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (".venv/lib/python3.12/site-packages",))
            .with_context(|| "fail to append path. use rye sync")?;

        let nes_wrappers = py.import_bound("nes_py.wrappers")?;
        let joypad_space = nes_wrappers.getattr("JoypadSpace")?;
        let gym = py.import_bound("gym_super_mario_bros")?;
        let gym_actions = py.import_bound("gym_super_mario_bros.actions")?;
        let complex_movement = gym_actions.getattr("COMPLEX_MOVEMENT")?;
        let make_func = gym.getattr("make")?;
        let env = make_func
            .call((env_name,), None)
            .with_context(|| "fail to call make function")?;
        let env = joypad_space.call((env, complex_movement), None)?;

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
        let array = array.call_method("transpose", (1, 2, 0), None)?;
        let array = array.call_method("reshape", (-1,), None)?;
        let array = array.extract()?;
        Ok(array)
    }
}

impl<'py> Env<4> for GymSuperMarioBrosEnv<'py> {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &ObservationSpace<4> {
        &self.observation_space
    }

    fn reset(&mut self) -> anyhow::Result<Vec<f32>> {
        let result: Bound<'py, PyAny> = self.env.call_method("reset", (), None)?;
        Self::ndarray_to_vec(result)
    }

    fn step(&mut self, action: &Action) -> anyhow::Result<(Vec<f32>, f32, bool)> {
        let py_action = action.to_object(self.py);
        let step = self.env.call_method("step", (py_action,), None)?;
        let observation = Self::ndarray_to_vec(step.get_item(0)?)?;
        let reward: f32 = step.get_item(1)?.extract()?;
        let done: bool = step.get_item(2)?.extract()?;
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

    use super::*;

    #[test]
    fn test_gym_env_discrete() -> anyhow::Result<()> {
        for (env_name, action_space, observation_space) in [(
            "SuperMarioBros-v3",
            ActionSpace::Discrete(12),
            ObservationSpace::Box {
                shape: [1, 3, 240, 256],
            },
        )] {
            let _result: anyhow::Result<()> = Python::with_gil(|py| {
                let mut env = GymSuperMarioBrosEnv::new(py, env_name)?;
                assert_eq!(env.action_space(), &action_space);
                assert_eq!(env.observation_space(), &observation_space);
                let observation = env.reset()?;
                assert_eq!(
                    observation.len(),
                    observation_space.shape()[1..].iter().product::<usize>(),
                );
                let (observation, reward, _is_done) = env.step(&Action::Discrete(0))?;
                assert_eq!(
                    observation.len(),
                    observation_space.shape()[1..].iter().product::<usize>()
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
