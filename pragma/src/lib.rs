use serde::Serialize;
use std::io::Write;

mod lexer;
pub mod ast;
mod parser;
mod span;
mod compound_result;
mod smol_str2;
pub mod elaborate;
mod intrinsics;

#[derive(Serialize, Debug)]
pub enum CompileError {
    ParseError(Vec<parser::ParseError>),
    ElaborateError(Vec<elaborate::ElaborateError>),
}

#[derive(Serialize, Debug)]
pub struct Output {
    pub ast: ast::Module,
    pub ctx: String,
}

pub fn compile(input: &str) -> Result<Output, CompileError> {
    let ast = parser::parse(input);
    let ast = ast.map_err(CompileError::ParseError)?;
    
    let ctx = elaborate::elaborate(&ast).map_err(CompileError::ElaborateError)?;
    
    let mut buffer = Vec::<u8>::new();
    
    for (ident, (item, ty)) in ctx.iter() {
        writeln!(buffer, "{} = {} \n    : {}\n", &ident[..], item, ty).unwrap();
    }
    
    Ok(Output {
        ast,
        ctx: String::from_utf8(buffer).unwrap(),
    })
}

#[test]
fn test() {
    let input = r#"
x [5] = 0;
a = x[5];
    "#;
    let output = compile(input);
    println!("{:#?}", output);
}
