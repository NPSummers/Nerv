use crate::ast::*;
use crate::stdlib::ExternalFunctions;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{FunctionValue, PointerValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    named_values: HashMap<String, PointerValue<'ctx>>,
    external_functions: ExternalFunctions<'ctx>,
    user_functions: HashMap<String, FunctionValue<'ctx>>,
}

type MainFunc = unsafe extern "C" fn() -> i32;

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("nerv_module");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let builder = context.create_builder();

        // Declare all external functions using our macro
        let external_functions = ExternalFunctions::declare_all(&module, context);

        CodeGenerator {
            context,
            module,
            builder,
            execution_engine,
            named_values: HashMap::new(),
            external_functions,
            user_functions: HashMap::new(),
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<i32, String> {
        let main_fn = self.gen_program(program)?;
        unsafe { Ok(main_fn.call()) }
    }

    fn gen_program(&mut self, program: &Program) -> Result<JitFunction<'_, MainFunc>, String> {
        // First pass: declare all functions
        for stmt in &program.body {
            if let Stmt::FunctionDecl(func_decl) = stmt {
                self.declare_function(func_decl)?;
            }
        }

        // Second pass: generate function bodies
        for stmt in &program.body {
            if let Stmt::FunctionDecl(func_decl) = stmt {
                self.gen_function_body(func_decl)?;
            }
        }

        // Generate main function
        let i32_type = self.context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let main_function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(main_function, "entry");
        self.builder.position_at_end(basic_block);

        // Execute non-function statements in main
        for stmt in &program.body {
            if !matches!(stmt, Stmt::FunctionDecl(_)) {
                self.gen_stmt(stmt, main_function)?;
            }
        }
        
        // Always return 0 from main
        let zero = i32_type.const_int(0, false);
        self.builder.build_return(Some(&zero));

        unsafe { self.execution_engine.get_function("main") }.map_err(|e| e.to_string())
    }

    fn gen_stmt(&mut self, stmt: &Stmt, function: FunctionValue) -> Result<(), String> {
        match stmt {
            Stmt::Expr(expr) => {
                // Just evaluate the expression but don't store result anywhere
                self.gen_expr(expr, function)?;
            }
            Stmt::Print(expr) => {
                let val = self.gen_expr(expr, function)?;
                let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
            }
            Stmt::VarDecl(var_decl) => {
                let i32_type = self.context.i32_type();
                let alloca = self.builder.build_alloca(i32_type, &var_decl.name);
                self.named_values.insert(var_decl.name.clone(), alloca);

                if let Some(initializer) = &var_decl.initializer {
                    let init_val = self.gen_expr(initializer, function)?;
                    self.builder.build_store(alloca, init_val);
                }
            }
            Stmt::FunctionDecl(_) => {
                // Function declarations are handled separately in a pre-pass
                // For now, we'll skip them in the main statement generation
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    let return_val = self.gen_expr(expr, function)?;
                    self.builder.build_return(Some(&return_val));
                } else {
                    // Return void - for now return 0
                    let zero = self.context.i32_type().const_int(0, false);
                    self.builder.build_return(Some(&zero));
                }
            }
            Stmt::If(if_stmt) => {
                self.gen_if_stmt(if_stmt, function)?;
            }
            Stmt::While(while_stmt) => {
                self.gen_while_stmt(while_stmt, function)?;
            }
            _ => return Err("Unsupported statement type".to_string()),
        }
        Ok(())
    }

    fn gen_expr(&mut self, expr: &Expr, function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        match expr {
            Expr::Literal(lit) => match lit {
                LiteralExpr::Int(val) => Ok(self.context.i32_type().const_int(*val as u64, true)),
                LiteralExpr::Bool(val) => Ok(self.context.i32_type().const_int(if *val { 1 } else { 0 }, false)),
                LiteralExpr::Char(val) => Ok(self.context.i32_type().const_int(*val as u64, false)),
                LiteralExpr::Float(_) => Err("Float literals not yet fully supported in codegen".to_string()),
                LiteralExpr::String(_) => Err("String literals not yet fully supported in codegen".to_string()),
                LiteralExpr::Array(_) => Err("Array literals not yet implemented".to_string()),
                LiteralExpr::Dict(_) => Err("Dictionary literals not yet implemented".to_string()),
            },
            Expr::Binary(bin_expr) => {
                let left = self.gen_expr(&bin_expr.left, function)?;
                let right = self.gen_expr(&bin_expr.right, function)?;
                match bin_expr.op {
                    // Arithmetic
                    BinaryOp::Add => Ok(self.builder.build_int_add(left, right, "tmpadd")),
                    BinaryOp::Subtract => Ok(self.builder.build_int_sub(left, right, "tmpsub")),
                    BinaryOp::Multiply => Ok(self.builder.build_int_mul(left, right, "tmpmul")),
                    BinaryOp::Divide => Ok(self.builder.build_int_signed_div(left, right, "tmpdiv")),
                    BinaryOp::Modulo => Ok(self.builder.build_int_signed_rem(left, right, "tmpmod")),
                    
                    // Comparison
                    BinaryOp::Equal => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::EQ, left, right, "tmpeq");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "eq_result"))
                    },
                    BinaryOp::NotEqual => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::NE, left, right, "tmpne");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "ne_result"))
                    },
                    BinaryOp::LessThan => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::SLT, left, right, "tmplt");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "lt_result"))
                    },
                    BinaryOp::GreaterThan => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::SGT, left, right, "tmpgt");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "gt_result"))
                    },
                    BinaryOp::LessThanOrEqual => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::SLE, left, right, "tmple");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "le_result"))
                    },
                    BinaryOp::GreaterThanOrEqual => {
                        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::SGE, left, right, "tmpge");
                        Ok(self.builder.build_int_cast(cmp, self.context.i32_type(), "ge_result"))
                    },
                    
                    // Logical (treat as boolean)
                    BinaryOp::And => {
                        let zero = self.context.i32_type().const_int(0, false);
                        let left_bool = self.builder.build_int_compare(inkwell::IntPredicate::NE, left, zero, "left_bool");
                        let right_bool = self.builder.build_int_compare(inkwell::IntPredicate::NE, right, zero, "right_bool");
                        let and_result = self.builder.build_and(left_bool, right_bool, "and_result");
                        Ok(self.builder.build_int_cast(and_result, self.context.i32_type(), "and_int"))
                    },
                    BinaryOp::Or => {
                        let zero = self.context.i32_type().const_int(0, false);
                        let left_bool = self.builder.build_int_compare(inkwell::IntPredicate::NE, left, zero, "left_bool");
                        let right_bool = self.builder.build_int_compare(inkwell::IntPredicate::NE, right, zero, "right_bool");
                        let or_result = self.builder.build_or(left_bool, right_bool, "or_result");
                        Ok(self.builder.build_int_cast(or_result, self.context.i32_type(), "or_int"))
                    },
                    
                    // Bitwise
                    BinaryOp::BitwiseAnd => Ok(self.builder.build_and(left, right, "bitand")),
                    BinaryOp::BitwiseOr => Ok(self.builder.build_or(left, right, "bitor")),
                    BinaryOp::BitwiseXor => Ok(self.builder.build_xor(left, right, "bitxor")),
                    BinaryOp::LeftShift => Ok(self.builder.build_left_shift(left, right, "lshift")),
                    BinaryOp::RightShift => Ok(self.builder.build_right_shift(left, right, true, "rshift")),
                }
            }
            Expr::Identifier(name) => {
                let var = self.named_values.get(name).ok_or("Unknown variable")?;
                Ok(self.builder.build_load(*var, name).into_int_value())
            }
            Expr::FunctionCall(func_call) => {
                self.gen_function_call(func_call, function)
            }
            Expr::Unary(unary_expr) => {
                let operand = self.gen_expr(&unary_expr.operand, function)?;
                match unary_expr.op {
                    UnaryOp::Minus => Ok(self.builder.build_int_neg(operand, "tmpneg")),
                    UnaryOp::Plus => Ok(operand), // Unary plus is a no-op
                    UnaryOp::Not => {
                        // For logical not, treat as boolean operation
                        let zero = self.context.i32_type().const_int(0, false);
                        let is_zero = self.builder.build_int_compare(inkwell::IntPredicate::EQ, operand, zero, "is_zero");
                        Ok(self.builder.build_int_cast(is_zero, self.context.i32_type(), "not_result"))
                    }
                }
            }
// Index and Assignment expressions removed to eliminate warnings
        }
    }

    fn gen_function_call(&mut self, func_call: &FunctionCallExpr, _function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        // Check if this is a user-defined function first
        let user_func = self.user_functions.get(&func_call.name).copied();
        if let Some(user_func) = user_func {
            // Generate arguments for user function
            let mut args = Vec::new();
            for arg_expr in &func_call.args {
                let arg_val = self.gen_expr(arg_expr, _function)?;
                args.push(arg_val.into());
            }
            
            // Call the user function
            let call_result = self.builder.build_call(user_func, &args, "user_call");
            return Ok(call_result.try_as_basic_value().left().unwrap().into_int_value());
        }

        // Get the function from our stdlib functions
        let stdlib_func = self.external_functions.get_function_by_name(&func_call.name)
            .ok_or_else(|| format!("Unknown function: {}", func_call.name))?;

        // Generate arguments
        let mut args = Vec::new();
        for arg_expr in &func_call.args {
            match arg_expr {
                Expr::Literal(LiteralExpr::Int(val)) => {
                    args.push(self.context.i32_type().const_int(*val as u64, true).into());
                }
                Expr::Literal(LiteralExpr::String(s)) => {
                    let global_str = self.builder.build_global_string_ptr(s, ".str").as_pointer_value();
                    args.push(global_str.into());
                }
                _ => {
                    // For other expressions, generate them as integers for now
                    let expr_val = self.gen_expr(arg_expr, _function)?;
                    args.push(expr_val.into());
                }
            }
        }

        // Call the function
        let call_result = self.builder.build_call(stdlib_func, &args, "call");
        
        // Handle return value based on function signature
        match func_call.name.as_str() {
            "printf" | "puts" | "abs" => {
                Ok(call_result.try_as_basic_value().left().unwrap().into_int_value())
            }
            "malloc" | "free" => {
                // These return pointers or void, return 0 for now
                Ok(self.context.i32_type().const_int(0, false))
            }
            "sqrt" | "pow" => {
                // These return floats, but for now cast to int
                let float_val = call_result.try_as_basic_value().left().unwrap().into_float_value();
                Ok(self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int"))
            }
            _ => Err(format!("Unsupported function: {}", func_call.name))
        }
    }

    fn declare_function(&mut self, func_decl: &FunctionDeclStmt) -> Result<(), String> {
        let i32_type = self.context.i32_type();
        
        // For now, all functions take and return i32
        let param_types: Vec<_> = func_decl.params.iter().map(|_| i32_type.into()).collect();
        let fn_type = i32_type.fn_type(&param_types, false);
        
        let function = self.module.add_function(&func_decl.name, fn_type, None);
        self.user_functions.insert(func_decl.name.clone(), function);
        
        Ok(())
    }

    fn gen_function_body(&mut self, func_decl: &FunctionDeclStmt) -> Result<(), String> {
        let function = *self.user_functions.get(&func_decl.name).unwrap();
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // Save current named values (for nested scope)
        let saved_values = self.named_values.clone();
        
        // Create allocas for parameters
        let i32_type = self.context.i32_type();
        for (i, (param_name, _param_type)) in func_decl.params.iter().enumerate() {
            let alloca = self.builder.build_alloca(i32_type, param_name);
            let param = function.get_nth_param(i as u32).unwrap().into_int_value();
            self.builder.build_store(alloca, param);
            self.named_values.insert(param_name.clone(), alloca);
        }

        // Generate function body
        for stmt in &func_decl.body {
            self.gen_stmt(stmt, function)?;
        }

        // If no explicit return, return 0
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            let zero = self.context.i32_type().const_int(0, false);
            self.builder.build_return(Some(&zero));
        }

        // Restore previous named values
        self.named_values = saved_values;
        
        Ok(())
    }

    fn gen_if_stmt(&mut self, if_stmt: &IfStmt, function: FunctionValue) -> Result<(), String> {
        let condition_val = self.gen_expr(&if_stmt.condition, function)?;
        
        // Convert condition to boolean (non-zero = true)
        let zero = self.context.i32_type().const_int(0, false);
        let cond_bool = self.builder.build_int_compare(
            inkwell::IntPredicate::NE, 
            condition_val, 
            zero, 
            "if_cond"
        );

        let then_block = self.context.append_basic_block(function, "then");
        let else_block = self.context.append_basic_block(function, "else");
        let merge_block = self.context.append_basic_block(function, "merge");

        // Branch based on condition
        self.builder.build_conditional_branch(cond_bool, then_block, else_block);

        // Generate then block
        self.builder.position_at_end(then_block);
        for stmt in &if_stmt.then_branch {
            self.gen_stmt(stmt, function)?;
        }
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_unconditional_branch(merge_block);
        }

        // Generate else block
        self.builder.position_at_end(else_block);
        if let Some(else_branch) = &if_stmt.else_branch {
            for stmt in else_branch {
                self.gen_stmt(stmt, function)?;
            }
        }
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_unconditional_branch(merge_block);
        }

        // Continue at merge block
        self.builder.position_at_end(merge_block);
        Ok(())
    }

    fn gen_while_stmt(&mut self, while_stmt: &WhileStmt, function: FunctionValue) -> Result<(), String> {
        let loop_header = self.context.append_basic_block(function, "loop_header");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let loop_exit = self.context.append_basic_block(function, "loop_exit");

        // Jump to loop header
        self.builder.build_unconditional_branch(loop_header);

        // Generate loop header (condition check)
        self.builder.position_at_end(loop_header);
        let condition_val = self.gen_expr(&while_stmt.condition, function)?;
        let zero = self.context.i32_type().const_int(0, false);
        let cond_bool = self.builder.build_int_compare(
            inkwell::IntPredicate::NE, 
            condition_val, 
            zero, 
            "while_cond"
        );
        self.builder.build_conditional_branch(cond_bool, loop_body, loop_exit);

        // Generate loop body
        self.builder.position_at_end(loop_body);
        for stmt in &while_stmt.body {
            self.gen_stmt(stmt, function)?;
        }
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_unconditional_branch(loop_header);
        }

        // Continue at loop exit
        self.builder.position_at_end(loop_exit);
        Ok(())
    }
}
