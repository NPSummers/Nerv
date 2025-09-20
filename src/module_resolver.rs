use crate::ast::*;
use crate::parser::Parser;
use crate::diagnostics;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

pub struct ModuleResolver {
    // map of module name -> fully resolved Program (flattened statements)
    cache: HashMap<String, Program>,
    // currently resolving stack to detect cycles
    resolving: HashSet<String>,
    // search roots to find modules in
    search_paths: Vec<PathBuf>,
}

impl ModuleResolver {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            cache: HashMap::new(),
            resolving: HashSet::new(),
            search_paths,
        }
    }

    pub fn resolve_program(&mut self, program: &Program) -> Result<Program, String> {
        let mut out: Vec<Stmt> = Vec::new();
        for stmt in &program.body {
            self.expand_statement(stmt, &mut out)?;
        }
        Ok(Program { body: out })
    }

    fn expand_statement(&mut self, stmt: &Stmt, out: &mut Vec<Stmt>) -> Result<(), String> {
        match stmt {
            Stmt::Import(imp) => {
                let module_prog = self.load_module(&imp.module)?;

                match &imp.items {
                    None => {
                        // import module -> include all top-level decls from module
                        out.extend(module_prog.body.clone());
                    }
                    Some(items) => {
                        // from module import a, b
                        let item_set: HashSet<String> = items.iter().cloned().collect();
                        for s in &module_prog.body {
                            if let Some(name) = Self::top_level_decl_name(s) {
                                if item_set.contains(&name) {
                                    out.push(s.clone());
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            // For any non-import statement, just pass through
            other => {
                out.push(other.clone());
                Ok(())
            }
        }
    }

    fn top_level_decl_name(stmt: &Stmt) -> Option<String> {
        match stmt {
            Stmt::FunctionDecl(f) => Some(f.name.clone()),
            Stmt::ClassDecl(c) => Some(c.name.clone()),
            Stmt::VarDecl(v) => Some(v.name.clone()),
            _ => None,
        }
    }

    fn load_module(&mut self, name: &str) -> Result<Program, String> {
        if let Some(p) = self.cache.get(name) {
            return Ok(p.clone());
        }
        if self.resolving.contains(name) {
            return Err(format!("Circular import detected for module '{}'", name));
        }
        self.resolving.insert(name.to_string());

        let path = self.find_module_path(name).ok_or_else(|| {
            format!("Could not find module '{}' in search paths", name)
        })?;

        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read module '{}': {}", name, e))?;

        let mut parser = Parser::new(&content);
        let parsed = match parser.parse_program() {
            Ok(p) => p,
            Err(e) => {
                let src = diagnostics::SourceFile { name, src: &content };
                let span = parser.take_error_span();
                let msg = diagnostics::render_error(&format!("Parse error in module '{}': {}", name, e), &src, span.as_ref());
                return Err(msg);
            }
        };

        // Recursively resolve imports within the module
        let resolved = self.resolve_program(&parsed)?;

        self.resolving.remove(name);
        self.cache.insert(name.to_string(), resolved.clone());
        Ok(resolved)
    }

    fn find_module_path(&self, name: &str) -> Option<PathBuf> {
        // For now, modules map to '<name>.nerv' under any search path
        let filename = format!("{}.nerv", name);
        for root in &self.search_paths {
            let candidate = root.join(&filename);
            if candidate.exists() && candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }
}


