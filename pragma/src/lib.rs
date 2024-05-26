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
add [int], [int] = __intrinsic_int_add;
sub [int], [int] = __intrinsic_int_sub;
mul [int], [int] = __intrinsic_int_mul;
f = fn(x: int, y: int) x + y;
main = fn() f(2, 2);
    "#;
    let output = compile(input);
    println!("{:#?}", output);
}
