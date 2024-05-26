use std::fmt::{Debug, Display};
use std::ops::Deref;
use serde::Serialize;
use smol_str::SmolStr;

#[derive(Eq, PartialEq, Clone, Hash)]
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

impl From<String> for SmolStr2 {
    fn from(s: String) -> Self {
        SmolStr2(s.into())
    }
}

impl Deref for SmolStr2 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for SmolStr2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for SmolStr2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl From<&SmolStr2> for String {
    fn from(s: &SmolStr2) -> Self {
        s.0[..].into()
    }
}
