mod lexer;
mod ast;
mod parser;
mod codegen;
mod stdlib;

use parser::Parser;
use codegen::CodeGenerator;
use inkwell::context::Context;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let input = if args.len() > 1 {
        // Read from file
        let filename = &args[1];
        
        // Check if file has .nerv extension
        if !filename.ends_with(".nerv") {
            eprintln!("Error: File must have .nerv extension");
            return;
        }
        
        // Check if file exists
        if !Path::new(filename).exists() {
            eprintln!("Error: File '{}' not found", filename);
            return;
        }
        
        match fs::read_to_string(filename) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading file '{}': {}", filename, e);
                return;
            }
        }
    } else {
        // Read from stdin
        let mut input = String::new();
        match io::stdin().read_to_string(&mut input) {
            Ok(_) => input,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                return;
            }
        }
    };

    // Parse the input
    let mut parser = Parser::new(&input);
    let program = match parser.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    // Generate and execute code
    let context = Context::create();
    let mut codegen = CodeGenerator::new(&context);
    
    match codegen.run(&program) {
        Ok(_) => {
            // Program executed successfully
            // All output should come from print statements
        },
        Err(e) => eprintln!("Error during execution: {}", e),
    }
}
