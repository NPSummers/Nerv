use crate::ast::*;
use crate::stdlib::ExternalFunctions;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{FunctionValue, PointerValue, BasicValueEnum};
use inkwell::types::{BasicType, BasicMetadataTypeEnum};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    named_values: HashMap<String, (PointerValue<'ctx>, Type)>,
    external_functions: ExternalFunctions<'ctx>,
    user_functions: HashMap<String, FunctionValue<'ctx>>,
    class_definitions: HashMap<String, ClassDeclStmt>,
    class_types: HashMap<String, inkwell::types::StructType<'ctx>>,
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
            class_definitions: HashMap::new(),
            class_types: HashMap::new(),
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<i32, String> {
        let main_fn = self.gen_program(program)?;
        unsafe { Ok(main_fn.call()) }
    }

    fn gen_program(&mut self, program: &Program) -> Result<JitFunction<'_, MainFunc>, String> {
        // First pass: declare all classes and functions
        for stmt in &program.body {
            match stmt {
                Stmt::ClassDecl(class_decl) => {
                    self.declare_class(class_decl)?;
                }
                Stmt::FunctionDecl(func_decl) => {
                    self.declare_function(func_decl)?;
                }
                _ => {}
            }
        }

        // Second pass: generate function bodies and class methods
        for stmt in &program.body {
            match stmt {
                Stmt::ClassDecl(class_decl) => {
                    self.gen_class_methods(class_decl)?;
                }
                Stmt::FunctionDecl(func_decl) => {
                    self.gen_function_body(func_decl)?;
                }
                _ => {}
            }
        }

        // Generate main function
        let i32_type = self.context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let main_function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(main_function, "entry");
        self.builder.position_at_end(basic_block);

        // Execute non-function and non-class statements in main
        for stmt in &program.body {
            if !matches!(stmt, Stmt::FunctionDecl(_) | Stmt::ClassDecl(_)) {
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
                // Determine the type of expression and print accordingly
                match expr {
                    Expr::Literal(LiteralExpr::String(_)) => {
                        // For string literals, print as string
                        if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                            let format_str = self.builder.build_global_string_ptr("%s\n", ".str").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                        }
                    },
                    Expr::Literal(LiteralExpr::Float(_)) => {
                        // For float literals, print as float
                        if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                            let format_str = self.builder.build_global_string_ptr("%.2f\n", ".str").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                        }
                    },
                    Expr::Literal(LiteralExpr::Char(_)) => {
                        // For char literals, print as char
                        if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                            let format_str = self.builder.build_global_string_ptr("%c\n", ".str").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                        }
                    },
                    Expr::Identifier(name) => {
                        // For variables, check their type
                        if let Some((_ptr, var_type)) = self.named_values.get(name) {
                            match var_type {
                                Type::String => {
                                    if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                                        let format_str = self.builder.build_global_string_ptr("%s\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                Type::Float => {
                                    if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                                        let format_str = self.builder.build_global_string_ptr("%.2f\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                Type::Char => {
                                    if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                                        let format_str = self.builder.build_global_string_ptr("%c\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                _ => {
                                    // Default to integer printing
                                    let val = self.gen_expr(expr, function)?;
                                    let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
                                }
                            }
                        }
                    },
                    _ => {
                        // Default case: print as integer
                        let val = self.gen_expr(expr, function)?;
                        let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
                    }
                }
            }
            Stmt::VarDecl(var_decl) => {
                let var_type = var_decl.type_annotation.clone().unwrap_or_else(|| {
                    // Infer type from initializer if no explicit type
                    if let Some(initializer) = &var_decl.initializer {
                        self.infer_type_from_expr(initializer)
                    } else {
                        Type::Int // Default to int
                    }
                });

                let llvm_type = self.get_llvm_type(&var_type)?;
                let alloca = self.builder.build_alloca(llvm_type, &var_decl.name);
                self.named_values.insert(var_decl.name.clone(), (alloca, var_type.clone()));

                if let Some(initializer) = &var_decl.initializer {
                    // Check if this is a class type being initialized
                    if let Type::Custom(class_name) = &var_type {
                        if self.class_definitions.contains_key(class_name) {
                            // This is an object variable - get the pointer directly
                            if let Ok(ptr_val) = self.gen_expr_as_pointer(initializer, function) {
                                self.builder.build_store(alloca, ptr_val);
                            } else {
                                return Err("Failed to initialize object variable".to_string());
                            }
                        } else {
                            let init_val = self.gen_expr_typed(initializer, function, &var_type)?;
                            self.builder.build_store(alloca, init_val);
                        }
                    } else {
                        let init_val = self.gen_expr_typed(initializer, function, &var_type)?;
                        self.builder.build_store(alloca, init_val);
                    }
                }
            }
            Stmt::FunctionDecl(_) => {
                // Function declarations are handled separately in a pre-pass
                // For now, we'll skip them in the main statement generation
            }
            Stmt::ClassDecl(_) => {
                // Class declarations are handled separately in a pre-pass
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
            Stmt::Import(import_stmt) => {
                // For now, just ignore imports (placeholder for future module system)
                println!("Import: {} (not yet implemented)", import_stmt.module);
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
                LiteralExpr::Float(val) => {
                    let float_val = self.context.f64_type().const_float(*val);
                    // Convert float to int for now (temporary solution)
                    let int_val = self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int");
                    Ok(int_val)
                },
                LiteralExpr::String(val) => {
                    let _string_ptr = self.builder.build_global_string_ptr(val, "str_literal").as_pointer_value();
                    // For now, return the length of the string as an integer
                    Ok(self.context.i32_type().const_int(val.len() as u64, false))
                },
                LiteralExpr::Array(elements) => {
                    // For now, create a simple array of integers
                    let array_type = self.context.i32_type().array_type(elements.len() as u32);
                    let array_alloca = self.builder.build_alloca(array_type, "array");
                    
                    for (i, element) in elements.iter().enumerate() {
                        let element_value = self.gen_expr(element, function)?;
                        let element_ptr = unsafe {
                            self.builder.build_gep(
                                array_alloca,
                                &[self.context.i32_type().const_int(0, false), 
                                  self.context.i32_type().const_int(i as u64, false)],
                                "element_ptr"
                            )
                        };
                        self.builder.build_store(element_ptr, element_value);
                    }
                    
                    // Return the array pointer as an int (simplified)
                    Ok(self.context.i32_type().const_int(elements.len() as u64, false))
                },
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
                let (var_ptr, var_type) = self.named_values.get(name).ok_or("Unknown variable")?;
                let loaded_val = self.builder.build_load(*var_ptr, name);
                // For now, convert everything to int for compatibility
                match var_type {
                    Type::Float => {
                        let float_val = loaded_val.into_float_value();
                        Ok(self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int"))
                    },
                    Type::Custom(_) => {
                        // For object types, convert pointer to int for expressions
                        let ptr_val = loaded_val.into_pointer_value();
                        Ok(self.builder.build_ptr_to_int(ptr_val, self.context.i32_type(), "obj_to_int"))
                    },
                    _ => Ok(loaded_val.into_int_value())
                }
            }
            Expr::FunctionCall(func_call) => {
                self.gen_function_call(func_call, function)
            }
            Expr::Assignment(assignment) => {
                let value = self.gen_expr(&assignment.value, function)?;
                if let Some((var_ptr, _var_type)) = self.named_values.get(&assignment.target) {
                    self.builder.build_store(*var_ptr, value);
                    Ok(value) // Return the assigned value
                } else {
                    Err(format!("Undefined variable: {}", assignment.target))
                }
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
            Expr::MemberAccess(member_access) => {
                self.gen_member_access(member_access, function)
            }
            Expr::ObjectInstantiation(obj_inst) => {
                self.gen_object_instantiation(obj_inst, function)
            }
        }
    }

    fn gen_expr_as_pointer(&mut self, expr: &Expr, function: FunctionValue) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        match expr {
            Expr::FunctionCall(func_call) => {
                // Check if this is a class instantiation
                if self.class_definitions.contains_key(&func_call.name) {
                    let constructor_name = format!("{}_new", func_call.name);
                    if let Some(&constructor_func) = self.user_functions.get(&constructor_name) {
                        // Generate arguments
                        let mut args = Vec::new();
                        for arg_expr in &func_call.args {
                            let arg_val = self.gen_expr(arg_expr, function)?;
                            args.push(arg_val.into());
                        }
                        
                        // Call constructor and return the actual pointer
                        let call_result = self.builder.build_call(constructor_func, &args, "new_instance");
                        let instance_ptr = call_result.try_as_basic_value().left().unwrap().into_pointer_value();
                        return Ok(instance_ptr);
                    }
                }
                Err("Cannot get pointer from non-object expression".to_string())
            }
            _ => Err("Cannot get pointer from this expression type".to_string())
        }
    }

    fn gen_function_call(&mut self, func_call: &FunctionCallExpr, _function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        // Check if this is a member assignment first
        if func_call.name == "MEMBER_ASSIGN" {
            return self.gen_member_assignment(&func_call.args, _function);
        }
        
        // Check if this is a method call
        if func_call.name.starts_with("METHOD_CALL::") {
            let method_name = func_call.name.strip_prefix("METHOD_CALL::").unwrap();
            return self.gen_method_call(method_name, &func_call.args, _function);
        }
        
        // Check if this is a class instantiation
        if self.class_definitions.contains_key(&func_call.name) {
            let constructor_name = format!("{}_new", func_call.name);
            if let Some(&constructor_func) = self.user_functions.get(&constructor_name) {
                // Generate arguments
                let mut args = Vec::new();
                for arg_expr in &func_call.args {
                    // For string literals, we need to handle them specially
                    match arg_expr {
                        Expr::Literal(LiteralExpr::String(s)) => {
                            let global_str = self.builder.build_global_string_ptr(s, "str_literal");
                            args.push(global_str.as_pointer_value().into());
                        }
                        _ => {
                            let arg_val = self.gen_expr(arg_expr, _function)?;
                            args.push(arg_val.into());
                        }
                    }
                }
                
                // Call constructor
                let call_result = self.builder.build_call(constructor_func, &args, "new_instance");
                let instance_ptr = call_result.try_as_basic_value().left().unwrap().into_pointer_value();
                
                // Return the pointer as an int for now (we'll store it properly in variables)
                return Ok(self.builder.build_ptr_to_int(instance_ptr, self.context.i32_type(), "instance_int"));
            }
        }
        
        // Check if this is a user-defined function
        let user_func = self.user_functions.get(&func_call.name).copied();
        if let Some(user_func) = user_func {
            // Generate arguments for user function
            let mut args = Vec::new();
            for arg_expr in &func_call.args {
                // For string literals, we need to handle them specially
                match arg_expr {
                    Expr::Literal(LiteralExpr::String(s)) => {
                        let global_str = self.builder.build_global_string_ptr(s, "str_literal");
                        args.push(global_str.as_pointer_value().into());
                    }
                    _ => {
                        let arg_val = self.gen_expr(arg_expr, _function)?;
                        args.push(arg_val.into());
                    }
                }
            }
            
            // Call the user function
            let call_result = self.builder.build_call(user_func, &args, "user_call");
            let result_value = call_result.try_as_basic_value().left().unwrap();
            
            // Handle different return types
            match result_value {
                inkwell::values::BasicValueEnum::IntValue(int_val) => return Ok(int_val),
                inkwell::values::BasicValueEnum::PointerValue(ptr_val) => {
                    // Convert pointer to int for now
                    return Ok(self.builder.build_ptr_to_int(ptr_val, self.context.i32_type(), "ptr_to_int"));
                },
                inkwell::values::BasicValueEnum::FloatValue(float_val) => {
                    // Convert float to int for now
                    return Ok(self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int"));
                },
                _ => return Err("Unsupported function return type".to_string())
            }
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
        for (i, (param_name, param_type)) in func_decl.params.iter().enumerate() {
            let llvm_type = self.get_llvm_type(param_type)?;
            let alloca = self.builder.build_alloca(llvm_type, param_name);
            let param = function.get_nth_param(i as u32).unwrap();
            self.builder.build_store(alloca, param);
            self.named_values.insert(param_name.clone(), (alloca, param_type.clone()));
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

    fn get_llvm_type(&self, nerv_type: &Type) -> Result<inkwell::types::BasicTypeEnum<'ctx>, String> {
        match nerv_type {
            Type::Int => Ok(self.context.i32_type().into()),
            Type::Float => Ok(self.context.f64_type().into()),
            Type::Bool => Ok(self.context.i32_type().into()), // Use i32 for bool
            Type::Char => Ok(self.context.i8_type().into()),
            Type::String => Ok(self.context.i8_type().ptr_type(inkwell::AddressSpace::default()).into()),
            Type::List(_element_type) => {
                // For now, represent lists as i8* (pointer)
                Ok(self.context.i8_type().ptr_type(inkwell::AddressSpace::default()).into())
            },
            Type::Dict(_key_type, _value_type) => {
                // For now, represent dicts as i8* (pointer)
                Ok(self.context.i8_type().ptr_type(inkwell::AddressSpace::default()).into())
            },
            Type::Custom(class_name) => {
                // Check if this is a class type
                if let Some(class_type) = self.class_types.get(class_name) {
                    Ok(class_type.ptr_type(inkwell::AddressSpace::default()).into())
                } else {
                    // Default to i32 for unknown custom types
                    Ok(self.context.i32_type().into())
                }
            }
        }
    }

    fn infer_type_from_expr(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Literal(lit) => match lit {
                LiteralExpr::Int(_) => Type::Int,
                LiteralExpr::Float(_) => Type::Float,
                LiteralExpr::String(_) => Type::String,
                LiteralExpr::Bool(_) => Type::Bool,
                LiteralExpr::Char(_) => Type::Char,
                LiteralExpr::Array(_) => Type::List(Box::new(Type::Int)), // Default to int list
                LiteralExpr::Dict(_) => Type::Dict(Box::new(Type::String), Box::new(Type::Int)), // Default types
            },
            Expr::FunctionCall(func_call) => {
                // Check if this is a class constructor call
                if self.class_definitions.contains_key(&func_call.name) {
                    Type::Custom(func_call.name.clone())
                } else {
                    Type::Int // Default for regular function calls
                }
            },
            _ => Type::Int, // Default fallback
        }
    }

    fn gen_expr_for_print(&mut self, expr: &Expr, function: FunctionValue) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::Literal(lit) => match lit {
                LiteralExpr::String(val) => {
                    let global_str = self.builder.build_global_string_ptr(val, "str_literal");
                    Ok(global_str.as_pointer_value().into())
                },
                LiteralExpr::Float(val) => Ok(self.context.f64_type().const_float(*val).into()),
                LiteralExpr::Char(val) => Ok(self.context.i32_type().const_int(*val as u64, false).into()),
                _ => {
                    let int_val = self.gen_expr(expr, function)?;
                    Ok(int_val.into())
                }
            },
            Expr::Identifier(name) => {
                let (var_ptr, _var_type) = self.named_values.get(name).ok_or("Unknown variable")?;
                let loaded_val = self.builder.build_load(*var_ptr, name);
                Ok(loaded_val)
            },
            _ => {
                let int_val = self.gen_expr(expr, function)?;
                Ok(int_val.into())
            }
        }
    }

    fn gen_expr_typed(&mut self, expr: &Expr, function: FunctionValue, target_type: &Type) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::Literal(lit) => match lit {
                LiteralExpr::Int(val) => Ok(self.context.i32_type().const_int(*val as u64, true).into()),
                LiteralExpr::Float(val) => Ok(self.context.f64_type().const_float(*val).into()),
                LiteralExpr::String(val) => {
                    let global_str = self.builder.build_global_string_ptr(val, "str_literal");
                    Ok(global_str.as_pointer_value().into())
                },
                LiteralExpr::Bool(val) => Ok(self.context.i32_type().const_int(if *val { 1 } else { 0 }, false).into()),
                LiteralExpr::Char(val) => Ok(self.context.i8_type().const_int(*val as u64, false).into()),
                LiteralExpr::Array(elements) => {
                    // For now, create a simple array and return pointer
                    let array_type = self.context.i32_type().array_type(elements.len() as u32);
                    let array_alloca = self.builder.build_alloca(array_type, "array");
                    
                    for (i, element) in elements.iter().enumerate() {
                        let element_value = self.gen_expr(element, function)?;
                        let element_ptr = unsafe {
                            self.builder.build_gep(
                                array_alloca,
                                &[self.context.i32_type().const_int(0, false), 
                                  self.context.i32_type().const_int(i as u64, false)],
                                "element_ptr"
                            )
                        };
                        self.builder.build_store(element_ptr, element_value);
                    }
                    
                    Ok(array_alloca.into())
                },
                LiteralExpr::Dict(_pairs) => {
                    // For now, just return a null pointer
                    let ptr_type = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                    Ok(ptr_type.const_null().into())
                },
            },
            _ => {
                // For non-literal expressions, use the existing gen_expr and convert if needed
                let int_val = self.gen_expr(expr, function)?;
                match target_type {
                    Type::Float => {
                        let float_val = self.builder.build_signed_int_to_float(int_val, self.context.f64_type(), "int_to_float");
                        Ok(float_val.into())
                    },
                    _ => Ok(int_val.into())
                }
            }
        }
    }

    fn declare_class(&mut self, class_decl: &ClassDeclStmt) -> Result<(), String> {
        // Create LLVM struct type for the class
        let mut member_types = Vec::new();
        
        for member in &class_decl.members {
            let member_type = if let Some(ref t) = member.type_annotation {
                self.get_llvm_type(t)?
            } else {
                // Default to i32 if no type specified
                self.context.i32_type().into()
            };
            member_types.push(member_type);
        }
        
        let struct_type = self.context.struct_type(&member_types, false);
        self.class_types.insert(class_decl.name.clone(), struct_type);
        self.class_definitions.insert(class_decl.name.clone(), class_decl.clone());
        
        // Declare constructor (if exists)
        for method in &class_decl.methods {
            if method.name == class_decl.name {
                // This is a constructor
                self.declare_constructor(&class_decl.name, method)?;
            } else {
                // Regular method
                self.declare_method(&class_decl.name, method)?;
            }
        }
        
        Ok(())
    }

    fn declare_constructor(&mut self, class_name: &str, method: &FunctionDeclStmt) -> Result<(), String> {
        // Constructor returns a pointer to the class instance
        let class_type = *self.class_types.get(class_name).unwrap();
        let ptr_type = class_type.ptr_type(inkwell::AddressSpace::default());
        
        let param_types: Vec<BasicMetadataTypeEnum> = method.params.iter()
            .map(|(_, param_type)| self.get_llvm_type(param_type).unwrap().into())
            .collect();
        
        let fn_type = ptr_type.fn_type(&param_types, false);
        let function = self.module.add_function(&format!("{}_new", class_name), fn_type, None);
        self.user_functions.insert(format!("{}_new", class_name), function);
        
        Ok(())
    }

    fn declare_method(&mut self, class_name: &str, method: &FunctionDeclStmt) -> Result<(), String> {
        // Methods take a pointer to the class instance as first parameter
        let class_type = *self.class_types.get(class_name).unwrap();
        let ptr_type = class_type.ptr_type(inkwell::AddressSpace::default());
        
        let mut param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into()]; // 'this' parameter
        param_types.extend(method.params.iter()
            .map(|(_, param_type)| -> BasicMetadataTypeEnum { self.get_llvm_type(param_type).unwrap().into() }));
        
        let return_type = if let Some(ref ret_type) = method.return_type {
            self.get_llvm_type(ret_type)?
        } else {
            self.context.i32_type().into()
        };
        
        let fn_type = return_type.fn_type(&param_types, false);
        let function = self.module.add_function(&format!("{}_{}", class_name, method.name), fn_type, None);
        self.user_functions.insert(format!("{}_{}", class_name, method.name), function);
        
        Ok(())
    }

    fn gen_class_methods(&mut self, class_decl: &ClassDeclStmt) -> Result<(), String> {
        for method in &class_decl.methods {
            if method.name == class_decl.name {
                // Constructor
                self.gen_constructor_body(&class_decl.name, method)?;
            } else {
                // Regular method
                self.gen_method_body(&class_decl.name, method)?;
            }
        }
        Ok(())
    }

    fn gen_constructor_body(&mut self, class_name: &str, method: &FunctionDeclStmt) -> Result<(), String> {
        let function = *self.user_functions.get(&format!("{}_new", class_name)).unwrap();
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let saved_values = self.named_values.clone();
        
        // Allocate memory for the class instance using malloc
        let class_type = *self.class_types.get(class_name).unwrap();
        let size = class_type.size_of().unwrap();
        let malloc_call = self.builder.build_call(
            self.external_functions.malloc, 
            &[size.into()], 
            "malloc_call"
        );
        let instance_ptr = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();
        
        // Cast to the correct type
        let typed_ptr = self.builder.build_bitcast(
            instance_ptr, 
            class_type.ptr_type(inkwell::AddressSpace::default()), 
            "typed_instance"
        ).into_pointer_value();
        
        // Initialize parameters
        for (i, (param_name, param_type)) in method.params.iter().enumerate() {
            let llvm_type = self.get_llvm_type(param_type)?;
            let param_alloca = self.builder.build_alloca(llvm_type, param_name);
            let param = function.get_nth_param(i as u32).unwrap();
            self.builder.build_store(param_alloca, param);
            self.named_values.insert(param_name.clone(), (param_alloca, param_type.clone()));
        }
        
        // Set up 'this' to point to the instance
        self.named_values.insert("this".to_string(), (typed_ptr, Type::Custom(class_name.to_string())));
        
        // Generate constructor body
        for stmt in &method.body {
            self.gen_stmt(stmt, function)?;
        }
        
        // Return the instance pointer
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_return(Some(&typed_ptr));
        }
        
        self.named_values = saved_values;
        Ok(())
    }

    fn gen_method_body(&mut self, class_name: &str, method: &FunctionDeclStmt) -> Result<(), String> {
        let function = *self.user_functions.get(&format!("{}_{}", class_name, method.name)).unwrap();
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let saved_values = self.named_values.clone();
        
        // First parameter is 'this'
        let this_param = function.get_nth_param(0).unwrap().into_pointer_value();
        self.named_values.insert("this".to_string(), (this_param, Type::Custom(class_name.to_string())));
        
        // Initialize other parameters
        for (i, (param_name, param_type)) in method.params.iter().enumerate() {
            let llvm_type = self.get_llvm_type(param_type)?;
            let param_alloca = self.builder.build_alloca(llvm_type, param_name);
            let param = function.get_nth_param((i + 1) as u32).unwrap(); // +1 because 'this' is first
            self.builder.build_store(param_alloca, param);
            self.named_values.insert(param_name.clone(), (param_alloca, param_type.clone()));
        }
        
        // Generate method body
        for stmt in &method.body {
            self.gen_stmt(stmt, function)?;
        }
        
        // Default return if no explicit return
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            let zero = self.context.i32_type().const_int(0, false);
            self.builder.build_return(Some(&zero));
        }
        
        self.named_values = saved_values;
        Ok(())
    }

    fn gen_member_access(&mut self, member_access: &MemberAccessExpr, _function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        match &*member_access.object {
            Expr::Identifier(obj_name) => {
                if let Some((obj_var_ptr, obj_type)) = self.named_values.get(obj_name) {
                    if let Type::Custom(class_name) = obj_type {
                        if let Some(class_def) = self.class_definitions.get(class_name) {
                            let obj_ptr = if obj_name == "this" {
                                // 'this' is already a pointer to the object
                                *obj_var_ptr
                            } else {
                                // Regular object variables store a pointer, so load it
                                self.builder.build_load(*obj_var_ptr, obj_name).into_pointer_value()
                            };
                            
                            // Find the member index
                            for (i, member) in class_def.members.iter().enumerate() {
                                if member.name == member_access.member {
                                    // Access the member using GEP on the object pointer
                                    let member_ptr = self.builder.build_struct_gep(obj_ptr, i as u32, &format!("{}_member", member.name)).unwrap();
                                    let loaded_val = self.builder.build_load(member_ptr, &member.name);
                                    return Ok(loaded_val.into_int_value());
                                }
                            }
                            return Err(format!("Member {} not found in class {}", member_access.member, class_name));
                        }
                    }
                }
                Err(format!("Cannot access member {} of {} - not an object", member_access.member, obj_name))
            }
            _ => Err("Complex member access not yet implemented".to_string())
        }
    }

    fn gen_object_instantiation(&mut self, obj_inst: &ObjectInstantiationExpr, function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        // Check if this is a class constructor call
        if self.class_definitions.contains_key(&obj_inst.class_name) {
            let constructor_name = format!("{}_new", obj_inst.class_name);
            if let Some(&constructor_func) = self.user_functions.get(&constructor_name) {
                // Generate arguments
                let mut args = Vec::new();
                for arg_expr in &obj_inst.args {
                    let arg_val = self.gen_expr(arg_expr, function)?;
                    args.push(arg_val.into());
                }
                
                // Call constructor
                let call_result = self.builder.build_call(constructor_func, &args, "new_instance");
                let instance_ptr = call_result.try_as_basic_value().left().unwrap().into_pointer_value();
                
                // For now, return the pointer as an int (simplified)
                Ok(self.builder.build_ptr_to_int(instance_ptr, self.context.i32_type(), "instance_int"))
            } else {
                Err(format!("No constructor found for class {}", obj_inst.class_name))
            }
        } else {
            Err(format!("Unknown class: {}", obj_inst.class_name))
        }
    }

    fn gen_method_call(&mut self, method_name: &str, args: &[Expr], function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        // First argument is the object
        if args.is_empty() {
            return Err("Method call requires an object".to_string());
        }
        
        let obj_expr = &args[0];
        let method_args = &args[1..]; // Rest are method arguments
        
        // Get the object type
        match obj_expr {
            Expr::Identifier(obj_name) => {
                if let Some((obj_var_ptr, obj_type)) = self.named_values.get(obj_name) {
                    if let Type::Custom(class_name) = obj_type {
                        // Look up the method
                        let method_func_name = format!("{}_{}", class_name, method_name);
                        if let Some(&method_func) = self.user_functions.get(&method_func_name) {
                            // Load the object pointer
                            let obj_ptr = if obj_name == "this" {
                                // 'this' is already a pointer
                                (*obj_var_ptr).into()
                            } else {
                                // Regular object variables store a pointer, so load it
                                self.builder.build_load(*obj_var_ptr, obj_name)
                            };
                            
                            // Generate method arguments
                            let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![obj_ptr.into()]; // First arg is 'this'
                            for arg_expr in method_args {
                                // For string literals, we need to handle them specially
                                match arg_expr {
                                    Expr::Literal(LiteralExpr::String(s)) => {
                                        let global_str = self.builder.build_global_string_ptr(s, "str_literal");
                                        call_args.push(global_str.as_pointer_value().into());
                                    }
                                    _ => {
                                        let arg_val = self.gen_expr(arg_expr, function)?;
                                        call_args.push(arg_val.into());
                                    }
                                }
                            }
                            
                            // Call the method
                            let call_result = self.builder.build_call(method_func, &call_args, "method_call");
                            let result_value = call_result.try_as_basic_value().left().unwrap();
                            
                            // Handle different return types
                            return match result_value {
                                inkwell::values::BasicValueEnum::IntValue(int_val) => Ok(int_val),
                                inkwell::values::BasicValueEnum::PointerValue(ptr_val) => {
                                    // Convert pointer to int for now
                                    Ok(self.builder.build_ptr_to_int(ptr_val, self.context.i32_type(), "ptr_to_int"))
                                },
                                inkwell::values::BasicValueEnum::FloatValue(float_val) => {
                                    // Convert float to int for now
                                    Ok(self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int"))
                                },
                                _ => Err("Unsupported method return type".to_string())
                            }
                        } else {
                            return Err(format!("Method {} not found in class {}", method_name, class_name));
                        }
                    }
                }
                Err(format!("Object {} not found or not a class instance", obj_name))
            }
            _ => Err("Complex method calls not yet supported".to_string())
        }
    }

    fn gen_member_assignment(&mut self, args: &[Expr], function: FunctionValue) -> Result<inkwell::values::IntValue<'ctx>, String> {
        if args.len() != 2 {
            return Err("Member assignment requires exactly 2 arguments".to_string());
        }
        
        let member_access = &args[0];
        let value_expr = &args[1];
        
        // Generate the value to assign
        let value = self.gen_expr(value_expr, function)?;
        
        // Handle the member access for assignment
        if let Expr::MemberAccess(member_access_expr) = member_access {
            match &*member_access_expr.object {
                Expr::Identifier(obj_name) => {
                    if let Some((obj_var_ptr, obj_type)) = self.named_values.get(obj_name) {
                        if let Type::Custom(class_name) = obj_type {
                            if let Some(class_def) = self.class_definitions.get(class_name) {
                                let obj_ptr = if obj_name == "this" {
                                    // 'this' is already a pointer to the object
                                    *obj_var_ptr
                                } else {
                                    // Regular object variables store a pointer, so load it
                                    self.builder.build_load(*obj_var_ptr, obj_name).into_pointer_value()
                                };
                                
                                // Find the member index
                                for (i, member) in class_def.members.iter().enumerate() {
                                    if member.name == member_access_expr.member {
                                        // Get the member pointer and store the value
                                        let member_ptr = self.builder.build_struct_gep(obj_ptr, i as u32, &format!("{}_member", member.name)).unwrap();
                                        self.builder.build_store(member_ptr, value);
                                        return Ok(value); // Return the assigned value
                                    }
                                }
                                return Err(format!("Member {} not found in class {}", member_access_expr.member, class_name));
                            }
                        }
                    }
                    Err(format!("Cannot assign to member {} of {} - not an object", member_access_expr.member, obj_name))
                }
                _ => Err("Complex member assignment not yet implemented".to_string())
            }
        } else {
            Err("Invalid member assignment".to_string())
        }
    }
}
