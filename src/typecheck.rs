use std::collections::HashMap;

use crate::ast::*;
use crate::stdlib::{get_function_arity, get_function_signature, resolve_std_alias};

#[derive(Default)]
struct Env {
    vars: HashMap<String, Type>,
}

pub struct TypeChecker {
    functions: HashMap<String, (Vec<Type>, Option<Type>)>,
    classes: HashMap<String, ClassDeclStmt>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            classes: HashMap::new(),
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<(), String> {
        // Collect class and function signatures first
        for stmt in &program.body {
            match stmt {
                Stmt::ClassDecl(c) => {
                    self.classes.insert(c.name.clone(), c.clone());
                }
                Stmt::FunctionDecl(f) => {
                    let params: Vec<Type> = f.params.iter().map(|(_, t)| t.clone()).collect();
                    self.functions.insert(f.name.clone(), (params, f.return_type.clone()));
                }
                _ => {}
            }
        }

        // Check top-level statements and functions
        let mut env = Env::default();
        for stmt in &program.body {
            self.check_stmt(stmt, &mut env, None)?;
        }
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Stmt, env: &mut Env, current_fn: Option<&FunctionDeclStmt>) -> Result<(), String> {
        match stmt {
            Stmt::VarDecl(v) => {
                if let Some(init) = &v.initializer {
                    let init_ty = self.infer_expr_type(init, env)?;
                    if let Some(ann) = &v.type_annotation {
                        if !self.is_assignable(ann, &init_ty) {
                            return Err(format!("Type mismatch in variable '{}': expected {:?}, got {:?}", v.name, ann, init_ty));
                        }
                        env.vars.insert(v.name.clone(), ann.clone());
                    } else {
                        env.vars.insert(v.name.clone(), init_ty);
                    }
                } else {
                    // No initializer; require annotation or default to Int
                    let ty = v.type_annotation.clone().unwrap_or(Type::Int);
                    env.vars.insert(v.name.clone(), ty);
                }
                Ok(())
            }
            Stmt::Expr(e) => { self.infer_expr_type(e, env).map(|_| ()) }
            Stmt::Return(opt_e) => {
                if let Some(f) = current_fn {
                    if let Some(rt) = &f.return_type {
                        let got = if let Some(e) = opt_e { self.infer_expr_type(e, env)? } else { Type::Int };
                        if !self.is_assignable(rt, &got) {
                            return Err(format!("Return type mismatch in function '{}': expected {:?}, got {:?}", f.name, rt, got));
                        }
                    }
                }
                Ok(())
            }
            Stmt::FunctionDecl(f) => {
                // New scope for params
                let mut local = Env { vars: HashMap::new() };
                for (name, ty) in &f.params { local.vars.insert(name.clone(), ty.clone()); }
                for s in &f.body { self.check_stmt(s, &mut local, Some(f))?; }
                Ok(())
            }
            Stmt::ClassDecl(c) => {
                // Check methods
                for m in &c.methods {
                    let mut local = Env { vars: HashMap::new() };
                    // Methods receive declared params (no implicit self in current AST)
                    for (name, ty) in &m.params { local.vars.insert(name.clone(), ty.clone()); }
                    for s in &m.body { self.check_stmt(s, &mut local, Some(m))?; }
                }
                Ok(())
            }
            Stmt::If(i) => {
                let _ = self.infer_expr_type(&i.condition, env)?;
                for s in &i.then_branch { self.check_stmt(s, env, current_fn)?; }
                if let Some(eb) = &i.else_branch { for s in eb { self.check_stmt(s, env, current_fn)?; } }
                Ok(())
            }
            Stmt::While(w) => {
                let _ = self.infer_expr_type(&w.condition, env)?;
                for s in &w.body { self.check_stmt(s, env, current_fn)?; }
                Ok(())
            }
            Stmt::For(f) => {
                // iterable must be list or range(...)
                let iter_ty = self.infer_expr_type(&f.iterable, env)?;
                match iter_ty {
                    Type::List(elem) => {
                        env.vars.insert(f.variable.clone(), *elem);
                        for s in &f.body { self.check_stmt(s, env, current_fn)?; }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            Stmt::Import(_) | Stmt::Print(_) => Ok(()),
        }
    }

    fn infer_expr_type(&mut self, expr: &Expr, env: &mut Env) -> Result<Type, String> {
        match expr {
            Expr::Literal(l) => Ok(match l {
                LiteralExpr::Int(_) => Type::Int,
                LiteralExpr::Float(_) => Type::Float,
                LiteralExpr::String(_) => Type::String,
                LiteralExpr::Bool(_) => Type::Bool,
                LiteralExpr::Char(_) => Type::Char,
                LiteralExpr::Array(_) => Type::List(Box::new(Type::Int)),
                LiteralExpr::Dict(_) => Type::Dict(Box::new(Type::String), Box::new(Type::Int)),
            }),
            Expr::Identifier(name) => env.vars.get(name).cloned().ok_or_else(|| format!("Unknown identifier '{}'", name)),
            Expr::Assignment(a) => {
                let vty = env.vars.get(&a.target).cloned().ok_or_else(|| format!("Assignment to unknown variable '{}'", a.target))?;
                let rty = self.infer_expr_type(&a.value, env)?;
                if !self.is_assignable(&vty, &rty) { return Err(format!("Cannot assign {:?} to variable '{}' of type {:?}", rty, a.target, vty)); }
                Ok(vty)
            }
            Expr::Binary(b) => {
                let lt = self.infer_expr_type(&b.left, env)?;
                let rt = self.infer_expr_type(&b.right, env)?;
                use BinaryOp::*;
                match b.op {
                    Add => {
                        // String concatenation if either side is string
                        if matches!(lt, Type::String) || matches!(rt, Type::String) {
                            Ok(Type::String)
                        } else {
                            if !self.is_numeric(&lt) || !self.is_numeric(&rt) { return Err("Arithmetic on non-numeric types".to_string()); }
                            if matches!(lt, Type::Float) || matches!(rt, Type::Float) { Ok(Type::Float) } else { Ok(Type::Int) }
                        }
                    }
                    Multiply => {
                        // Special-case: list repetition like [x] * N or N * [x]
                        let is_array_left = matches!(&*b.left, Expr::Literal(LiteralExpr::Array(_)));
                        let is_array_right = matches!(&*b.right, Expr::Literal(LiteralExpr::Array(_)));
                        if (is_array_left && matches!(rt, Type::Int)) || (is_array_right && matches!(lt, Type::Int)) {
                            // Resulting type is the list type from the array literal side
                            if let Type::List(elem) = lt.clone() { return Ok(Type::List(elem)); }
                            if let Type::List(elem) = rt.clone() { return Ok(Type::List(elem)); }
                        }
                        if !self.is_numeric(&lt) || !self.is_numeric(&rt) { return Err("Arithmetic on non-numeric types".to_string()); }
                        if matches!(lt, Type::Float) || matches!(rt, Type::Float) { Ok(Type::Float) } else { Ok(Type::Int) }
                    }
                    Subtract | Divide => {
                        if !self.is_numeric(&lt) || !self.is_numeric(&rt) { return Err("Arithmetic on non-numeric types".to_string()); }
                        if matches!(lt, Type::Float) || matches!(rt, Type::Float) { Ok(Type::Float) } else { Ok(Type::Int) }
                    }
                    Modulo => {
                        if lt != Type::Int || rt != Type::Int { return Err("Modulo expects int operands".to_string()); }
                        Ok(Type::Int)
                    }
                    Equal | NotEqual | LessThan | GreaterThan | LessThanOrEqual | GreaterThanOrEqual => Ok(Type::Int),
                    And | Or => Ok(Type::Int),
                    BitwiseAnd | BitwiseOr | BitwiseXor | LeftShift | RightShift => {
                        if lt != Type::Int || rt != Type::Int { return Err("Bitwise ops expect int operands".to_string()); }
                        Ok(Type::Int)
                    }
                }
            }
            Expr::Unary(u) => {
                let _t = self.infer_expr_type(&u.operand, env)?;
                Ok(Type::Int)
            }
            Expr::IndexAccess(ix) => {
                let obj_ty = self.infer_expr_type(&ix.object, env)?;
                let idx_ty = self.infer_expr_type(&ix.index, env)?;
                match obj_ty {
                    Type::List(elem) => {
                        if idx_ty != Type::Int { return Err("List index must be int".to_string()); }
                        Ok(*elem)
                    }
                    Type::Dict(key, val) => {
                        if !self.is_assignable(&key, &idx_ty) { return Err(format!("Dict index type mismatch: expected {:?}, got {:?}", key, idx_ty)); }
                        Ok(*val)
                    }
                    Type::String => Err("Indexing strings is not supported".to_string()),
                    _ => Err("Indexing is only supported on list or dict".to_string()),
                }
            }
            Expr::FunctionCall(fc) => {
                // Method call sugar: METHOD_CALL::name(obj, ...)
                if fc.name.starts_with("METHOD_CALL::") {
                    let mname = fc.name.trim_start_matches("METHOD_CALL::");
                    if fc.args.is_empty() { return Err("Method call requires object".to_string()); }
                    let obj_ty = self.infer_expr_type(&fc.args[0], env)?;
                    match obj_ty {
                        Type::Custom(cls) => {
                            let class_name = cls.clone();
                            let (params, ret_ty) = {
                                let c = self.classes.get(&class_name).ok_or_else(|| format!("Unknown class '{}'", class_name))?;
                                let method = c.methods.iter().find(|m| m.name == mname).ok_or_else(|| format!("Unknown method '{}' on {}", mname, class_name))?;
                                (method.params.clone(), method.return_type.clone())
                            };
                            if fc.args.len() - 1 != params.len() { return Err(format!("Method '{}' expects {} args, got {}", mname, params.len(), fc.args.len()-1)); }
                            for (i, (_pn, pty)) in params.iter().enumerate() {
                                let aty = self.infer_expr_type(&fc.args[i + 1], env)?;
                                if !self.is_assignable(pty, &aty) { return Err(format!("Argument {} type mismatch for method '{}': expected {:?}, got {:?}", i+1, mname, pty, aty)); }
                            }
                            Ok(ret_ty.unwrap_or(Type::Int))
                        }
                        _ => Err("Method call on non-object".to_string()),
                    }
                } else {
                    // stdlib or user function
                    if let Some((params, ret)) = self.functions.get(&fc.name).cloned() {
                        if fc.args.len() != params.len() { return Err(format!("Function '{}' expects {} args, got {}", fc.name, params.len(), fc.args.len())); }
                        for (i, pty) in params.iter().enumerate() {
                            let aty = self.infer_expr_type(&fc.args[i], env)?;
                            if !self.is_assignable(pty, &aty) { return Err(format!("Argument {} type mismatch for function '{}': expected {:?}, got {:?}", i+1, fc.name, pty, aty)); }
                        }
                        Ok(ret.unwrap_or(Type::Int))
                    } else {
                        // stdlib by signature
                        let name = resolve_std_alias(&fc.name).unwrap_or(&fc.name);
                        if let Some((param_kinds, ret_kind)) = get_function_signature(name) {
                            let (min_args, max_args_opt) = get_function_arity(name)
                                .unwrap_or((param_kinds.len(), Some(param_kinds.len())));
                            let argc = fc.args.len();
                            let arity_ok = match max_args_opt {
                                Some(max_args) => argc >= min_args && argc <= max_args,
                                None => argc >= min_args,
                            };
                            if !arity_ok {
                                return Err(match max_args_opt {
                                    Some(max_args) => format!(
                                        "Function '{}' expects between {} and {} args, got {}",
                                        fc.name, min_args, max_args, argc
                                    ),
                                    None => format!(
                                        "Function '{}' expects at least {} args, got {}",
                                        fc.name, min_args, argc
                                    ),
                                });
                            }
                            // Type-check only the fixed prefix declared in the signature
                            for (i, kind) in param_kinds.iter().enumerate() {
                                let aty = self.infer_expr_type(&fc.args[i], env)?;
                                if !self.match_std_kind(kind, &aty) { return Err(format!("Argument {} type mismatch for function '{}': expected {}, got {:?}", i+1, fc.name, kind, aty)); }
                            }
                            Ok(self.std_ret_type(ret_kind))
                        } else if fc.name == "len" {
                            if fc.args.len() != 1 { return Err("len expects 1 argument".to_string()); }
                            let aty = self.infer_expr_type(&fc.args[0], env)?;
                            match aty {
                                Type::String | Type::List(_) | Type::Dict(_, _) => Ok(Type::Int),
                                _ => Err("len expects string, list, or dict".to_string()),
                            }
                        } else if fc.name == "list_get" {
                            if fc.args.len() != 2 { return Err("list_get expects 2 arguments".to_string()); }
                            let lty = self.infer_expr_type(&fc.args[0], env)?;
                            if let Type::List(elem) = lty { Ok(*elem) } else { Err("list_get expects list as first argument".to_string()) }
                        } else if fc.name == "dict_get" {
                            if fc.args.len() != 2 { return Err("dict_get expects 2 arguments".to_string()); }
                            let dty = self.infer_expr_type(&fc.args[0], env)?;
                            if let Type::Dict(_, val) = dty { Ok(*val) } else { Err("dict_get expects dict as first argument".to_string()) }
                        } else {
                            // Unknown function: default int return
                            Ok(Type::Int)
                        }
                    }
                }
            }
            Expr::MemberAccess(ma) => {
                // Validate member exists
                if let Expr::Identifier(obj_name) = &*ma.object {
                    if let Some(Type::Custom(cls)) = env.vars.get(obj_name) {
                        if let Some(c) = self.classes.get(cls) {
                            for m in &c.members {
                                if m.name == ma.member {
                                    return Ok(m.type_annotation.clone().unwrap_or(Type::Int));
                                }
                            }
                            return Err(format!("Unknown member '{}' on class '{}'", ma.member, cls));
                        }
                    }
                }
                Ok(Type::Int)
            }
            Expr::ObjectInstantiation(oi) => {
                // Treat as pointer-like custom type
                Ok(Type::Custom(oi.class_name.clone()))
            }
            Expr::InterpolatedString(_) => Ok(Type::String),
        }
    }

    fn is_numeric(&self, t: &Type) -> bool {
        matches!(t, Type::Int | Type::Float | Type::Bool | Type::Char)
    }

    fn is_assignable(&self, to: &Type, from: &Type) -> bool {
        to == from || matches!((to, from),
            (Type::Float, Type::Int) |
            (Type::Int, Type::Char) |
            (Type::Int, Type::Bool) |
            (Type::Bool, Type::Int)
        )
    }

    fn match_std_kind(&self, kind: &str, t: &Type) -> bool {
        match kind {
            "i32" | "i64" => matches!(t, Type::Int | Type::Bool | Type::Char | Type::Float),
            "f64" => matches!(t, Type::Float | Type::Int),
            "bool" => matches!(t, Type::Bool | Type::Int),
            "i8_ptr" => matches!(t, Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_)),
            _ => true,
        }
    }

    fn std_ret_type(&self, kind: &str) -> Type {
        match kind {
            "i8_ptr" => Type::String,
            "f64" => Type::Float,
            "bool" => Type::Bool,
            _ => Type::Int,
        }
    }
}

pub fn check_program(program: &Program) -> Result<(), String> {
    let mut tc = TypeChecker::new();
    tc.check_program(program)
}


