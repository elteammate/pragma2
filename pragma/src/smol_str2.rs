use std::fmt::Debug;
use serde::Serialize;
use smol_str::SmolStr;

#[derive(Eq, PartialEq, Clone)]
pub struct SmolStr2(SmolStr);

impl Serialize for SmolStr2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl From<&str> for SmolStr2 {
    fn from(s: &str) -> Self {
        SmolStr2(s.into())
    }
}

impl Debug for SmolStr2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}
