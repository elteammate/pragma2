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
mod c;
mod emit;

#[derive(Serialize, Debug)]
pub enum CompileError {
    ParseError(Vec<parser::ParseError>),
    ElaborateError(Vec<elaborate::ElaborateError>),
}

#[derive(Serialize, Debug)]
pub struct Output {
    pub ast: ast::Module,
    pub ctx: String,
    pub c: String,
}

pub fn compile(input: &str) -> Result<Output, CompileError> {
    let ast = parser::parse(input);
    let ast = ast.map_err(CompileError::ParseError)?;
    
    let mut ctx = elaborate::elaborate(&ast).map_err(CompileError::ElaborateError)?;
    
    let mut buffer = Vec::<u8>::new();
    
    for (ident, (item, ty)) in ctx.items.iter() {
        writeln!(buffer, "{} = {:4} \n      : {:8}\n", &ident[..], item, ty).unwrap();
    }

    let c = elaborate::extract_c(&mut ctx).map_err(CompileError::ElaborateError)?;
    let c = emit::emit(&c);

    Ok(Output {
        ast,
        ctx: String::from_utf8(buffer).unwrap(),
        c,
    })
}

#[test]
fn test() {
    let input = r#"
not [bool] = fn(x: bool) {
  if (x) {
    false
  } else {
    true
  }
};

lt [int], [int] = fn(x: int, y: int) __intrinsic_int_lt;
ge [int], [int] = fn(x: int, y: int) __intrinsic_int_lt;

main = fn() {
  x := 5;
  while (x >= 0) {
    x = x - 1;
  }
};
    "#;
    let output = compile(input);
    println!("{:#?}", output);
}
