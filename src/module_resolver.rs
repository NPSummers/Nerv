//! Import resolver that expands `import` statements by inlining module contents.
//!
//! Design:
//! - Modules map to files named `<name>.nerv` found under configured search paths.
//! - `import foo;` inlines all top-level declarations from `foo`.
//! - `from foo import a, b;` inlines only the selected declarations.
//! - Basic cycle detection prevents infinite recursion on circular imports.
//!
use crate::ast::*;
use crate::parser::Parser;
use crate::diagnostics;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub struct ModuleResolver {
    // map of module name -> fully resolved Program (flattened statements)
    cache: HashMap<String, Program>,
    // currently resolving stack to detect cycles
    resolving: HashSet<String>,
    // search roots to find modules in
    search_paths: Vec<PathBuf>,
    // alias map for imports: alias -> module path
    aliases: HashMap<String, String>,
}

impl ModuleResolver {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            cache: HashMap::new(),
            resolving: HashSet::new(),
            search_paths,
            aliases: HashMap::new(),
        }
    }

    pub fn resolve_program(&mut self, program: &Program) -> Result<Program, String> {
        let mut out: Vec<Stmt> = Vec::new();
        for stmt in &program.body {
            self.expand_statement(stmt, &mut out)?;
        }
        // After expansion, rewrite any alias-qualified names into full module paths
        let mut rewrited = Program { body: out };
        self.rewrite_aliases_in_program(&mut rewrited);
        Ok(rewrited)
    }

    fn expand_statement(&mut self, stmt: &Stmt, out: &mut Vec<Stmt>) -> Result<(), String> {
        match stmt {
            Stmt::Import(imp) => {
                // Record alias if provided (only for bare imports)
                if imp.items.is_none() {
                    if let Some(alias) = &imp.alias {
                        self.aliases.insert(alias.clone(), imp.module.clone());
                    }
                }

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
        // Built-in virtual namespaces like nerv::std are resolved to empty modules here;
        // their symbols are provided by stdlib/codegen without file IO.
        if name.starts_with("nerv::") {
            return Ok(Program { body: vec![] });
        }
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
        // Strategy:
        // - If absolute path or starts with '/' -> use as path; append .nerv if needed
        // - If starts with './' or '../' -> resolve relative to each search root
        // - If contains '::' -> treat as package path with separators
        // - Else -> '<name>.nerv' under search roots
        let as_path = if name.starts_with('/') {
            let mut p = PathBuf::from(name);
            if p.extension().is_none() { p.set_extension("nerv"); }
            return if p.exists() && p.is_file() { Some(p) } else { None };
        } else if name.starts_with("./") || name.starts_with("../") {
            let rel = Path::new(name);
            for root in &self.search_paths {
                let mut candidate = root.join(rel);
                if candidate.extension().is_none() { candidate.set_extension("nerv"); }
                if candidate.exists() && candidate.is_file() { return Some(candidate); }
            }
            return None;
        } else {
            // package-style path using '::'
            let path_like = name.replace("::", "/");
            Some(PathBuf::from(format!("{}.nerv", path_like)))
        };
        if let Some(p) = as_path {
            if p.is_absolute() { return if p.exists() && p.is_file() { Some(p) } else { None }; }
            for root in &self.search_paths {
                let candidate = root.join(&p);
                if candidate.exists() && candidate.is_file() { return Some(candidate); }
            }
        }
        None
    }

    fn rewrite_aliases_in_program(&self, program: &mut Program) {
        for stmt in &mut program.body {
            self.rewrite_aliases_in_stmt(stmt);
        }
    }

    fn rewrite_aliases_in_stmt(&self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Expr(e) => self.rewrite_aliases_in_expr(e),
            Stmt::VarDecl(v) => {
                if let Some(init) = &mut v.initializer { self.rewrite_aliases_in_expr(init); }
            }
            Stmt::FunctionDecl(f) => {
                for s in &mut f.body { self.rewrite_aliases_in_stmt(s); }
            }
            Stmt::ClassDecl(c) => {
                for m in &mut c.members {
                    if let Some(init) = &mut m.initializer { self.rewrite_aliases_in_expr(init); }
                }
                for m in &mut c.methods {
                    for s in &mut m.body { self.rewrite_aliases_in_stmt(s); }
                }
            }
            Stmt::Return(opt) => { if let Some(e) = opt { self.rewrite_aliases_in_expr(e); } }
            Stmt::Print(e) => self.rewrite_aliases_in_expr(e),
            Stmt::If(i) => {
                self.rewrite_aliases_in_expr(&mut i.condition);
                for s in &mut i.then_branch { self.rewrite_aliases_in_stmt(s); }
                if let Some(else_b) = &mut i.else_branch { for s in else_b { self.rewrite_aliases_in_stmt(s); } }
            }
            Stmt::While(w) => {
                self.rewrite_aliases_in_expr(&mut w.condition);
                for s in &mut w.body { self.rewrite_aliases_in_stmt(s); }
            }
            Stmt::For(f) => {
                self.rewrite_aliases_in_expr(&mut f.iterable);
                for s in &mut f.body { self.rewrite_aliases_in_stmt(s); }
            }
            Stmt::Import(_) => {}
        }
    }

    fn rewrite_aliases_in_expr(&self, expr: &mut Expr) {
        match expr {
            Expr::Identifier(name) => {
                if let Some(new_name) = self.expand_alias_path(name) { *name = new_name; }
            }
            Expr::FunctionCall(fc) => {
                if let Some(new_name) = self.expand_alias_path(&fc.name) { fc.name = new_name; }
                for a in &mut fc.args { self.rewrite_aliases_in_expr(a); }
            }
            Expr::Binary(b) => { self.rewrite_aliases_in_expr(&mut b.left); self.rewrite_aliases_in_expr(&mut b.right); }
            Expr::Unary(u) => self.rewrite_aliases_in_expr(&mut u.operand),
            Expr::Assignment(a) => self.rewrite_aliases_in_expr(&mut a.value),
            Expr::MemberAccess(m) => self.rewrite_aliases_in_expr(&mut m.object),
            Expr::ObjectInstantiation(o) => { for a in &mut o.args { self.rewrite_aliases_in_expr(a); } }
            Expr::IndexAccess(i) => { self.rewrite_aliases_in_expr(&mut i.object); self.rewrite_aliases_in_expr(&mut i.index); }
            Expr::InterpolatedString(s) => {
                for seg in &mut s.segments {
                    if let crate::ast::InterpolatedSegment::Expr(e) = seg { self.rewrite_aliases_in_expr(e); }
                }
            }
            Expr::Literal(crate::ast::LiteralExpr::Array(items)) => { for e in items { self.rewrite_aliases_in_expr(e); } }
            Expr::Literal(crate::ast::LiteralExpr::Dict(pairs)) => { for (k,v) in pairs { self.rewrite_aliases_in_expr(k); self.rewrite_aliases_in_expr(v); } }
            _ => {}
        }
    }

    fn expand_alias_path(&self, name: &str) -> Option<String> {
        // If name starts with '<alias>::', replace with recorded module path
        if let Some((head, tail)) = name.split_once("::") {
            if let Some(full) = self.aliases.get(head) {
                let mut s = String::with_capacity(full.len() + 2 + tail.len());
                s.push_str(full);
                s.push_str("::");
                s.push_str(tail);
                return Some(s);
            }
        }
        None
    }
}


