use wasm_bindgen::prelude::*;
use pragma::parser::{parse};

#[wasm_bindgen]
pub fn compile(input: &str) -> JsValue {
    serde_wasm_bindgen::to_value(&parse(input)).unwrap()
}
