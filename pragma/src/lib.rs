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
        writeln!(buffer, "{} = {:4} \n      : {:8}\n", &ident[..], item, ty).unwrap();
    }
    
    Ok(Output {
        ast,
        ctx: String::from_utf8(buffer).unwrap(),
    })
}

#[test]
fn test() {
    let input = r#"
id T:type, b:T -> T = b;
also_int = id[_, int];

f T:type, a:T -> T;
f [int], a:int = a;
f [string], a:string = a;
f T:type, a:T = a;
g = f[int, 5];

test T:type -> type;
test [int] = string;
test T:type = T;
test2 T:type -> test[T];

plus T:type, Q:type, a:T, b:Q -> T;
plus [int], [int], a:int, b:int = a;
plus [string], [string], a:string, b:string = a;

six = plus[_, _, "abc", "cdb"];

id [int], x:int = x;
id [string], x:string = x;
bcd = id[_, "abs"];

a -> Fn(int) -> (Fn(int) -> int) = fn(a: int) fn(b: int) b;
b = a(5)(6);
    "#;
    let output = compile(input);
    println!("{:#?}", output);
}
