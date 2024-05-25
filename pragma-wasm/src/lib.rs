use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compile(input: &str) -> JsValue {
    let result = pragma::compile(input);
    serde_wasm_bindgen::to_value(&result).unwrap()
}
