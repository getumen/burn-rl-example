use pyo3::{Py, PyAny, Python, ToPyObject as _};

use crate::Action;

pub mod gym_super_mario_bros;
pub mod gymnasium;

impl Action {
    pub fn to_object(&self, py: Python) -> Py<PyAny> {
        match self {
            Action::Discrete(action) => action.to_object(py),
        }
    }
}
