//! Nerv CLI entrypoint and native shims exported to the JIT runtime.
//!
//! This file hosts:
//! - The CLI glue (stdin/file input, `--help`, `--version`).
//! - Native networking and JSON helper functions exposed via C ABI and
//!   registered into the JIT (`nerv_http_get`, `nerv_ws_*`, etc.).
//! - Minimal helpers to exchange strings and structured data between Rust and
//!   the generated code.
//!
mod lexer;
mod ast;
mod parser;
mod codegen;
mod stdlib;
mod module_resolver;
mod diagnostics;
mod runtime;

use parser::Parser;
use codegen::CodeGenerator;
use inkwell::context::Context;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [OPTIONS] [file.nerv]\n\n\
         Runs a Nerv program from a file or stdin.\n\n\
         Options:\n  \
           -h, --help       Show this help\n  \
           -V, --version    Show version\n\n\
         If no file is provided, input is read from stdin.\n\n\
         Examples:\n  \
           {program} examples/showcase.nerv\n  \
           echo 'print(\"hi\");' | {program}\n",
        program = program
    );
}

fn print_version() {
    let name = env!("CARGO_PKG_NAME");
    let ver = env!("CARGO_PKG_VERSION");
    println!("{name} {ver}");
}

enum CliAction {
    Help,
    Version,
    FromFile(String),
    FromStdin,
}

struct Source {
    name: String,
    input: String,
    input_dir: Option<PathBuf>,
}

fn parse_cli(args: &[String]) -> CliAction {
    if args.len() <= 1 { return CliAction::FromStdin; }
    match args[1].as_str() {
        "-h" | "--help" => CliAction::Help,
        "-V" | "--version" => CliAction::Version,
        "-" => CliAction::FromStdin,
        other => CliAction::FromFile(other.to_string()),
    }
}

fn load_source(action: &CliAction) -> Result<Source, ()> {
    match action {
        CliAction::FromStdin => {
            let mut input = String::new();
            match io::stdin().read_to_string(&mut input) {
                Ok(_) => Ok(Source { name: "<stdin>".to_string(), input, input_dir: None }),
                Err(e) => { eprintln!("Error reading input: {}", e); Err(()) }
            }
        }
        CliAction::FromFile(filename) => {
            if !filename.ends_with(".nerv") {
                eprintln!("Error: File must have .nerv extension");
                return Err(());
            }
            if !Path::new(filename).exists() {
                eprintln!("Error: File '{}' not found", filename);
                return Err(());
            }
            match fs::read_to_string(filename) {
                Ok(content) => Ok(Source {
                    name: filename.clone(),
                    input: content,
                    input_dir: Path::new(filename).parent().map(|p| p.to_path_buf()),
                }),
                Err(e) => { eprintln!("Error reading file '{}': {}", filename, e); Err(()) }
            }
        }
        CliAction::Help | CliAction::Version => unreachable!(),
    }
}

fn compile_and_run(source: &Source) -> Result<(), ()> {
    let mut parser = Parser::new(&source.input);
    let program_raw = match parser.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            let src = diagnostics::SourceFile { name: &source.name, src: &source.input };
            let span = parser.take_error_span();
            let msg = diagnostics::render_error(&format!("Parse error: {}", e), &src, span.as_ref());
            eprintln!("{}", msg);
            return Err(());
        }
    };

    let mut resolver = module_resolver::ModuleResolver::new({
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(dir) = &source.input_dir { roots.push(dir.clone()); }
        roots.push(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        vec![roots].into_iter().flatten().collect()
    });
    let program = match resolver.resolve_program(&program_raw) {
        Ok(p) => p,
        Err(e) => {
            let src = diagnostics::SourceFile { name: &source.name, src: &source.input };
            let msg = diagnostics::render_error(&format!("{}", e), &src, None);
            eprintln!("{}", msg);
            return Err(());
        }
    };

    let context = Context::create();
    let mut codegen = CodeGenerator::new(&context);
    match codegen.run(&program) {
        Ok(_) => Ok(()),
        Err(e) => { eprintln!("Error during execution: {}", e); Err(()) }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name = args.get(0).map(|s| s.as_str()).unwrap_or("nerv");
    match parse_cli(&args) {
        CliAction::Help => { print_usage(program_name); }
        CliAction::Version => { print_version(); }
        action @ CliAction::FromFile(_) | action @ CliAction::FromStdin => {
            if let Ok(source) = load_source(&action) {
                let _ = compile_and_run(&source);
            }
        }
    }
}
