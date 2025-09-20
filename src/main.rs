mod lexer;
mod ast;
mod parser;
mod codegen;
mod stdlib;
mod module_resolver;
mod diagnostics;

use parser::Parser;
use codegen::CodeGenerator;
use inkwell::context::Context;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

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
    let program_raw = match parser.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            let src = diagnostics::SourceFile { name: args.get(1).map(|s| s.as_str()).unwrap_or("<stdin>"), src: &input };
            let span = parser.take_error_span();
            let msg = diagnostics::render_error(&format!("Parse error: {}", e), &src, span.as_ref());
            eprintln!("{}", msg);
            return;
        }
    };

    // Resolve imports before codegen
    let mut resolver = module_resolver::ModuleResolver::new({
        // Prefer the directory of the input file if provided, then workspace roots (current dir)
        let mut roots: Vec<PathBuf> = Vec::new();
        if args.len() > 1 {
            if let Some(parent) = Path::new(&args[1]).parent() {
                roots.push(parent.to_path_buf());
            }
        }
        roots.push(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        vec![roots].into_iter().flatten().collect()
    });
    let program = match resolver.resolve_program(&program_raw) {
        Ok(p) => p,
        Err(e) => {
            let src = diagnostics::SourceFile { name: args.get(1).map(|s| s.as_str()).unwrap_or("<stdin>"), src: &input };
            let msg = diagnostics::render_error(&format!("{}", e), &src, None);
            eprintln!("{}", msg);
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
