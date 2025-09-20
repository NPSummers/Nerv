use crate::ast::*;
use crate::stdlib::{ExternalFunctions, builtin_lookup, BuiltinOp, get_function_signature};
// use inkwell::values::BasicValue; // not needed currently
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{FunctionValue, PointerValue, BasicValueEnum};
use inkwell::types::{BasicType, BasicMetadataTypeEnum};
use inkwell::OptimizationLevel;
use std::collections::HashMap;
use std::env;
use std::ffi::CString;

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
    user_function_signatures: HashMap<String, (Vec<Type>, Option<Type>)>,
    current_function_return_type: Option<Type>,
    argv_globals: Vec<inkwell::values::GlobalValue<'ctx>>,
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

        let mut cg = CodeGenerator {
            context,
            module,
            builder,
            execution_engine,
            named_values: HashMap::new(),
            external_functions,
            user_functions: HashMap::new(),
            class_definitions: HashMap::new(),
            class_types: HashMap::new(),
            user_function_signatures: HashMap::new(),
            current_function_return_type: None,
            argv_globals: Vec::new(),
        };
        // Register native symbols for networking shims
        unsafe {
            extern "C" {
                fn nerv_http_get(url: *const i8) -> *mut i8;
                fn nerv_http_post(url: *const i8, body: *const i8) -> *mut i8;
                fn nerv_ws_connect(url: *const i8) -> i32;
                fn nerv_ws_send(handle: i32, msg: *const i8) -> i32;
                fn nerv_ws_recv(handle: i32) -> *mut i8;
                fn nerv_ws_close(handle: i32) -> i32;
                fn nerv_json_pretty(s: *const i8) -> *mut i8;
                fn nerv_json_to_dict_ss(s: *const i8) -> *mut i8;
            }
            cg.execution_engine.add_global_mapping(&cg.external_functions.http_get, nerv_http_get as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.http_post, nerv_http_post as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.ws_connect, nerv_ws_connect as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.ws_send, nerv_ws_send as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.ws_recv, nerv_ws_recv as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.ws_close, nerv_ws_close as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.json_pretty, nerv_json_pretty as usize);
            cg.execution_engine.add_global_mapping(&cg.external_functions.json_to_dict_ss, nerv_json_to_dict_ss as usize);
        }
        cg
    }

    // Use better type inference for string interpolation to treat identifiers by their actual types
    fn infer_type_from_expr_for_interpolation(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Identifier(name) => {
                if let Some((_ptr, var_ty)) = self.named_values.get(name) {
                    return var_ty.clone();
                }
                self.infer_type_from_expr(expr)
            }
            Expr::MemberAccess(ma) => {
                if let Some((_class, member_ty)) = self.get_object_and_member_type(&ma.object, &ma.member).ok().flatten() {
                    return member_ty;
                }
                self.infer_type_from_expr(expr)
            }
            _ => self.infer_type_from_expr(expr)
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<i32, String> {
        self.gen_program(program)?;
        if env::var("NERV_DUMP_IR").is_ok() {
            let ir = self.module.print_to_string();
            eprintln!("===== LLVM IR BEGIN =====\n{}\n===== LLVM IR END =====", ir.to_string());
        }
        let main_fn: JitFunction<'_, MainFunc> = unsafe { self.execution_engine.get_function("main") }.map_err(|e| e.to_string())?;
        unsafe { Ok(main_fn.call()) }
    }

    fn build_entry_alloca(&self, function: FunctionValue<'ctx>, ty: inkwell::types::BasicTypeEnum<'ctx>, name: &str) -> PointerValue<'ctx> {
        let entry = function.get_first_basic_block().expect("function has no entry block");
        let tmp_builder = self.context.create_builder();
        if let Some(first_instr) = entry.get_first_instruction() {
            tmp_builder.position_before(&first_instr);
        } else {
            tmp_builder.position_at_end(entry);
        }
        tmp_builder.build_alloca(ty, name)
    }

    fn gen_program(&mut self, program: &Program) -> Result<(), String> {
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

        // Prepare argv as globals for builtin access
        self.materialize_process_args();

        // Execute non-function and non-class statements in main
        for stmt in &program.body {
            if !matches!(stmt, Stmt::FunctionDecl(_) | Stmt::ClassDecl(_)) {
                self.gen_stmt(stmt, main_function)?;
            }
        }
        
        // Always return 0 from main
        let zero = i32_type.const_int(0, false);
        self.builder.build_return(Some(&zero));

        Ok(())
    }

    fn materialize_process_args(&mut self) {
        if !self.argv_globals.is_empty() { return; }
        let args: Vec<String> = std::env::args().collect();
        for (idx, s) in args.iter().enumerate() {
            let gv = self.module.add_global(self.context.i8_type().array_type((s.len() + 1) as u32), None, &format!(".arg_{}", idx));
            let cstr = CString::new(s.as_str()).unwrap();
            let bytes = cstr.as_bytes_with_nul();
            let init = self.context.i8_type().const_array(&bytes.iter().map(|b| self.context.i8_type().const_int(*b as u64, false)).collect::<Vec<_>>());
            gv.set_initializer(&init);
            gv.set_constant(true);
            self.argv_globals.push(gv);
        }
    }

    fn gen_stmt(&mut self, stmt: &Stmt, function: inkwell::values::FunctionValue<'ctx>) -> Result<(), String> {
        match stmt {
            Stmt::Expr(expr) => {
                // Just evaluate the expression but don't store result anywhere
                self.gen_expr(expr, function)?;
            }
            Stmt::For(for_stmt) => {
                self.gen_for_stmt(for_stmt, function)?;
            }
            Stmt::Print(expr) => {
                // Determine the type of expression and print accordingly
                match expr {
                    Expr::FunctionCall(func_call) => {
                        // Pretty print dicts returned by http/json helpers
                        let is_dict_like = func_call.name == "http_get" || func_call.name == "http_post" || func_call.name == "json_to_dict_ss";
                        if is_dict_like {
                            // Get pointer to dict base
                            let as_int = self.gen_function_call(func_call, function)?;
                            let base_i8 = self.builder.build_int_to_ptr(as_int, self.context.i8_type().ptr_type(inkwell::AddressSpace::default()), "dict_ptr");

                            // Compute sizes for key/value (both string pointers)
                            let llvm_key_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                            let llvm_val_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                            let key_sz = llvm_key_ty.size_of(); // i64
                            let val_sz = llvm_val_ty.size_of(); // i64
                            let pair_sz = self.builder.build_int_add(key_sz, val_sz, "pair_sz");

                            // Load length and pairs base
                            let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "d_len_ptr_i8") };
                            let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "d_len_ptr").into_pointer_value();
                            let len = self.builder.build_load(len_ptr, "d_len").into_int_value();
                            let pairs_off = self.context.i32_type().const_int(8, false);
                            let pairs_base_i8 = unsafe { self.builder.build_gep(base_i8, &[pairs_off], "pairs_base_i8") };

                            // Print opening brace
                            let obr = self.builder.build_global_string_ptr("{", ".obr").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[obr.into()], "printf");

                            // Loop over entries
                            let i_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
                            self.builder.build_store(i_alloca, self.context.i32_type().const_int(0, false));
                            let header = self.context.append_basic_block(function, "pfd_header");
                            let body = self.context.append_basic_block(function, "pfd_body");
                            let exit = self.context.append_basic_block(function, "pfd_exit");
                            self.builder.build_unconditional_branch(header);
                            self.builder.position_at_end(header);
                            let i_val = self.builder.build_load(i_alloca, "i").into_int_value();
                            let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i_lt_len");
                            self.builder.build_conditional_branch(cond, body, exit);

                            self.builder.position_at_end(body);
                            let i_i64 = self.builder.build_int_cast(i_val, self.context.i64_type(), "i_i64");
                            let off = self.builder.build_int_mul(pair_sz, i_i64, "pair_off");
                            let off_i32 = self.builder.build_int_cast(off, self.context.i32_type(), "pair_off_i32");
                            let kv_base_i8 = unsafe { self.builder.build_gep(pairs_base_i8, &[off_i32], "kv_base_i8") };
                            let key_ptr = self.builder.build_bitcast(kv_base_i8, llvm_key_ty.ptr_type(inkwell::AddressSpace::default()), "key_ptr").into_pointer_value();
                            let key_loaded = self.builder.build_load(key_ptr, "k");
                            let val_off_i32 = self.builder.build_int_cast(key_sz, self.context.i32_type(), "val_off_i32");
                            let val_ptr_i8 = unsafe { self.builder.build_gep(kv_base_i8, &[val_off_i32], "val_ptr_i8") };
                            let val_ptr = self.builder.build_bitcast(val_ptr_i8, llvm_val_ty.ptr_type(inkwell::AddressSpace::default()), "val_ptr").into_pointer_value();
                            let val_loaded = self.builder.build_load(val_ptr, "v");
                            // print "key: value"
                            let fmt = self.builder.build_global_string_ptr("%s: %s", ".fmt").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), key_loaded.into(), val_loaded.into()], "printf");

                            // comma if more
                            let one = self.context.i32_type().const_int(1, false);
                            let next = self.builder.build_int_add(i_val, one, "next");
                            let has_more = self.builder.build_int_compare(inkwell::IntPredicate::SLT, next, len, "has_more");
                            let comma_bb = self.context.append_basic_block(function, "pfd_comma");
                            let after_comma = self.context.append_basic_block(function, "pfd_after_comma");
                            self.builder.build_conditional_branch(has_more, comma_bb, after_comma);
                            self.builder.position_at_end(comma_bb);
                            let comma = self.builder.build_global_string_ptr(", ", ".comma").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[comma.into()], "printf");
                            self.builder.build_unconditional_branch(after_comma);
                            self.builder.position_at_end(after_comma);
                            self.builder.build_store(i_alloca, next);
                            self.builder.build_unconditional_branch(header);
                            self.builder.position_at_end(exit);
                            let cbrn = self.builder.build_global_string_ptr("}\n", ".cbrn").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[cbrn.into()], "printf");
                            return Ok(());
                        }

                        // Special-case printing for other functions returning strings (i8_ptr), like getenv
                        if let Some((_params, ret_ty)) = get_function_signature(&func_call.name) {
                            if ret_ty == "i8_ptr" {
                                let as_int = self.gen_function_call(func_call, function)?;
                                let ptr = self.builder.build_int_to_ptr(as_int, self.context.i8_type().ptr_type(inkwell::AddressSpace::default()), "ret_ptr");
                                let format_str = self.builder.build_global_string_ptr("%s\n", ".str").as_pointer_value();
                                self.builder.build_call(self.external_functions.printf, &[format_str.into(), ptr.into()], "printf_call");
                                return Ok(());
                            }
                        }
                        // Fallback default integer printing
                        let val = self.gen_expr(expr, function)?;
                        let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
                    },
                    // Pretty print: JSON string begins with '{' or '['
                    Expr::Literal(LiteralExpr::String(s)) if s.trim_start().starts_with('{') || s.trim_start().starts_with('[') => {
                        if let Ok(basic_val) = self.gen_expr_for_print(expr, function) {
                            let format_str = self.builder.build_global_string_ptr("%s\n", ".str").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                        }
                    }
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
                                Type::List(elem_ty_box) => {
                                    // Inline pretty print: [a, b, c]
                                    let (var_ptr, _vt) = self.named_values.get(name).unwrap();
                                    let base_i8 = self.builder.build_load(*var_ptr, name).into_pointer_value();
                                    let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "len_ptr_i8") };
                                    let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "len_ptr").into_pointer_value();
                                    let len = self.builder.build_load(len_ptr, "len").into_int_value();
                                    let data_off = self.context.i32_type().const_int(8, false);
                                    let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
                                    let llvm_elem_ty = self.get_llvm_type(&*elem_ty_box)?;
                                    let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();
                                    // print "["
                                    let lbr = self.builder.build_global_string_ptr("[", ".lbr").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[lbr.into()], "printf");
                                    // loop and print elems
                                    let i_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
                                    self.builder.build_store(i_alloca, self.context.i32_type().const_int(0, false));
                                    let header = self.context.append_basic_block(function, "pl_header");
                                    let body = self.context.append_basic_block(function, "pl_body");
                                    let exit = self.context.append_basic_block(function, "pl_exit");
                                    self.builder.build_unconditional_branch(header);
                                    self.builder.position_at_end(header);
                                    let i_val = self.builder.build_load(i_alloca, "i").into_int_value();
                                    let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i_lt_len");
                                    self.builder.build_conditional_branch(cond, body, exit);
                                    self.builder.position_at_end(body);
                                    let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[i_val], "elem_ptr") };
                                    let elem_val = self.builder.build_load(elem_ptr, "elem");
                                    // choose placeholder by type
                                    match &**elem_ty_box {
                                        Type::Float => {
                                            let fmt = self.builder.build_global_string_ptr("%f", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), elem_val.into()], "printf");
                                        }
                                        Type::Char => {
                                            let v8 = elem_val.into_int_value();
                                            let v32 = self.builder.build_int_z_extend(v8, self.context.i32_type(), "c32");
                                            let fmt = self.builder.build_global_string_ptr("%c", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), v32.into()], "printf");
                                        }
                                        Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                            let fmt = self.builder.build_global_string_ptr("%s", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), elem_val.into()], "printf");
                                        }
                                        _ => {
                                            let fmt = self.builder.build_global_string_ptr("%d", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), elem_val.into()], "printf");
                                        }
                                    }
                                    // print comma if i+1 < len
                                    let one = self.context.i32_type().const_int(1, false);
                                    let next = self.builder.build_int_add(i_val, one, "next");
                                    let has_more = self.builder.build_int_compare(inkwell::IntPredicate::SLT, next, len, "has_more");
                                    let comma_bb = self.context.append_basic_block(function, "comma_bb");
                                    let after_comma = self.context.append_basic_block(function, "after_comma");
                                    self.builder.build_conditional_branch(has_more, comma_bb, after_comma);
                                    self.builder.position_at_end(comma_bb);
                                    let comma = self.builder.build_global_string_ptr(", ", ".comma").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[comma.into()], "printf");
                                    self.builder.build_unconditional_branch(after_comma);
                                    self.builder.position_at_end(after_comma);
                                    self.builder.build_store(i_alloca, next);
                                    self.builder.build_unconditional_branch(header);
                                    self.builder.position_at_end(exit);
                                    // closing bracket and newline
                                    let rbrn = self.builder.build_global_string_ptr("]\n", ".rbrn").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[rbrn.into()], "printf");
                                }
                                Type::Dict(key_ty_box, val_ty_box) => {
                                    let (var_ptr, _vt) = self.named_values.get(name).unwrap();
                                    let base_i8 = self.builder.build_load(*var_ptr, name).into_pointer_value();
                                    // len and pairs base
                                    let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "d_len_ptr_i8") };
                                    let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "d_len_ptr").into_pointer_value();
                                    let len = self.builder.build_load(len_ptr, "d_len").into_int_value();
                                    let pairs_base_off = self.context.i32_type().const_int(8, false);
                                    let pairs_base_i8 = unsafe { self.builder.build_gep(base_i8, &[pairs_base_off], "pairs_base_i8") };
                                    let llvm_key_ty = self.get_llvm_type(&*key_ty_box)?;
                                    let llvm_val_ty = self.get_llvm_type(&*val_ty_box)?;
                                    let key_sz = llvm_key_ty.size_of().ok_or("Failed to get key size")?;
                                    // print "{"
                                    let obr = self.builder.build_global_string_ptr("{", ".obr").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[obr.into()], "printf");
                                    // loop
                                    let i_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
                                    self.builder.build_store(i_alloca, self.context.i32_type().const_int(0, false));
                                    let header = self.context.append_basic_block(function, "pd_header");
                                    let body = self.context.append_basic_block(function, "pd_body");
                                    let exit = self.context.append_basic_block(function, "pd_exit");
                                    self.builder.build_unconditional_branch(header);
                                    self.builder.position_at_end(header);
                                    let i_val = self.builder.build_load(i_alloca, "i").into_int_value();
                                    let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i_lt_len");
                                    self.builder.build_conditional_branch(cond, body, exit);
                                    self.builder.position_at_end(body);
                                    // compute kv base
                                    let i_i64 = self.builder.build_int_cast(i_val, self.context.i64_type(), "i_i64");
                                    let pair_sz = self.builder.build_int_add(key_sz, llvm_val_ty.size_of().ok_or("Failed to get val size")?, "pair_sz");
                                    let off = self.builder.build_int_mul(pair_sz, i_i64, "off");
                                    let off_i32 = self.builder.build_int_cast(off, self.context.i32_type(), "off_i32");
                                    let kv_base_i8 = unsafe { self.builder.build_gep(pairs_base_i8, &[off_i32], "kv_base_i8") };
                                    let key_ptr = self.builder.build_bitcast(kv_base_i8, llvm_key_ty.ptr_type(inkwell::AddressSpace::default()), "key_ptr").into_pointer_value();
                                    let key_loaded = self.builder.build_load(key_ptr, "k");
                                    let val_off_i32 = self.builder.build_int_cast(key_sz, self.context.i32_type(), "val_off_i32");
                                    let val_ptr_i8 = unsafe { self.builder.build_gep(kv_base_i8, &[val_off_i32], "val_ptr_i8") };
                                    let val_ptr = self.builder.build_bitcast(val_ptr_i8, llvm_val_ty.ptr_type(inkwell::AddressSpace::default()), "val_ptr").into_pointer_value();
                                    let val_loaded = self.builder.build_load(val_ptr, "v");
                                    // print key
                                    match &**key_ty_box {
                                        Type::Float => {
                                            let fmt = self.builder.build_global_string_ptr("%f: ", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), key_loaded.into()], "printf");
                                        }
                                        Type::Char => {
                                            let v8 = key_loaded.into_int_value();
                                            let v32 = self.builder.build_int_z_extend(v8, self.context.i32_type(), "c32");
                                            let fmt = self.builder.build_global_string_ptr("%c: ", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), v32.into()], "printf");
                                        }
                                        Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                            let fmt = self.builder.build_global_string_ptr("%s: ", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), key_loaded.into()], "printf");
                                        }
                                        _ => {
                                            let fmt = self.builder.build_global_string_ptr("%d: ", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), key_loaded.into()], "printf");
                                        }
                                    }
                                    // print value
                                    match &**val_ty_box {
                                        Type::Float => {
                                            let fmt = self.builder.build_global_string_ptr("%f", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), val_loaded.into()], "printf");
                                        }
                                        Type::Char => {
                                            let v8 = val_loaded.into_int_value();
                                            let v32 = self.builder.build_int_z_extend(v8, self.context.i32_type(), "c32");
                                            let fmt = self.builder.build_global_string_ptr("%c", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), v32.into()], "printf");
                                        }
                                        Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                            let fmt = self.builder.build_global_string_ptr("%s", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), val_loaded.into()], "printf");
                                        }
                                        _ => {
                                            let fmt = self.builder.build_global_string_ptr("%d", ".fmt").as_pointer_value();
                                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), val_loaded.into()], "printf");
                                        }
                                    }
                                    // comma logic
                                    let one = self.context.i32_type().const_int(1, false);
                                    let next = self.builder.build_int_add(i_val, one, "next");
                                    let has_more = self.builder.build_int_compare(inkwell::IntPredicate::SLT, next, len, "has_more");
                                    let comma_bb = self.context.append_basic_block(function, "comma2");
                                    let after_comma = self.context.append_basic_block(function, "after_comma2");
                                    self.builder.build_conditional_branch(has_more, comma_bb, after_comma);
                                    self.builder.position_at_end(comma_bb);
                                    let comma = self.builder.build_global_string_ptr(", ", ".comma").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[comma.into()], "printf");
                                    self.builder.build_unconditional_branch(after_comma);
                                    self.builder.position_at_end(after_comma);
                                    self.builder.build_store(i_alloca, next);
                                    self.builder.build_unconditional_branch(header);
                                    self.builder.position_at_end(exit);
                                    let cbrn = self.builder.build_global_string_ptr("}\n", ".cbrn").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[cbrn.into()], "printf");
                                }
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
                    Expr::MemberAccess(member_access) => {
                        // Determine member type from class definition
                        if let Some((class_name, member_type)) = self.get_object_and_member_type(&member_access.object, &member_access.member).ok().flatten() {
                            match member_type {
                                Type::String => {
                                    if let Ok(basic_val) = self.gen_member_access_for_print(member_access, function) {
                                        let format_str = self.builder.build_global_string_ptr("%s\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                Type::Float => {
                                    if let Ok(basic_val) = self.gen_member_access_for_print(member_access, function) {
                                        let format_str = self.builder.build_global_string_ptr("%.2f\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                Type::Char => {
                                    if let Ok(basic_val) = self.gen_member_access_for_print(member_access, function) {
                                        let format_str = self.builder.build_global_string_ptr("%c\n", ".str").as_pointer_value();
                                        self.builder.build_call(self.external_functions.printf, &[format_str.into(), basic_val.into()], "printf_call");
                                    }
                                },
                                _ => {
                                    // Default to integer printing
                                    let val = self.gen_member_access(member_access, function)?;
                                    let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
                                }
                            }
                            let _ = class_name; // avoid unused
                        } else {
                            // Fallback: print as integer
                            let val = self.gen_member_access(member_access, function)?;
                            let format_str = self.builder.build_global_string_ptr("%d\n", ".str").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[format_str.into(), val.into()], "printf_call");
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
                let alloca = self.build_entry_alloca(function, llvm_type, &var_decl.name);
                self.named_values.insert(var_decl.name.clone(), (alloca, var_type.clone()));

                'after_init: {
                if let Some(initializer) = &var_decl.initializer {
                    // If variable expects a pointer-like type and initializer is a stdlib function that returns i8_ptr,
                    // call it and store the pointer directly to avoid int truncation.
                    if matches!(var_type, Type::String | Type::List(_) | Type::Dict(_, _)) {
                        if let Expr::FunctionCall(fc) = initializer {
                            if let Some((param_types, ret_ty)) = get_function_signature(&fc.name) {
                                if ret_ty == "i8_ptr" {
                                    // Prepare args per signature
                                    let stdlib_func = self.external_functions.get_function_by_name(&fc.name)
                                        .ok_or_else(|| format!("Unknown function: {}", fc.name))?;
                                    let mut args = Vec::new();
                                    for (idx, arg_expr) in fc.args.iter().enumerate() {
                                        let expected = param_types.get(idx).copied().unwrap_or("i32");
                                        match expected {
                                            "i8_ptr" => {
                                                match arg_expr {
                                                    Expr::Literal(LiteralExpr::String(s)) => {
                                                        let ptr = self.builder.build_global_string_ptr(s, ".str").as_pointer_value();
                                                        args.push(ptr.into());
                                                    }
                                                    _ => {
                                                        let int_val = self.gen_expr(arg_expr, function)?;
                                                        let ptr_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                                                        let ptr = self.builder.build_int_to_ptr(int_val, ptr_ty, "int_to_ptr");
                                                        args.push(ptr.into());
                                                    }
                                                }
                                            }
                                            "f64" => {
                                                let int_val = self.gen_expr(arg_expr, function)?;
                                                let float_val = self.builder.build_signed_int_to_float(int_val, self.context.f64_type(), "int_to_f64");
                                                args.push(float_val.into());
                                            }
                                            "bool" => {
                                                let int_val = self.gen_expr(arg_expr, function)?;
                                                let i1 = self.builder.build_int_truncate_or_bit_cast(int_val, self.context.bool_type(), "to_bool");
                                                args.push(i1.into());
                                            }
                                            _ => {
                                                let int_val = self.gen_expr(arg_expr, function)?;
                                                args.push(int_val.into());
                                            }
                                        }
                                    }
                                    // Call and store pointer result
                                    let call = self.builder.build_call(stdlib_func, &args, "call_init_ptr");
                                    let ptr = call.try_as_basic_value().left().unwrap().into_pointer_value();
                                    self.builder.build_store(alloca, ptr);
                                    break 'after_init;
                                }
                            }
                        }
                    }
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
                } // end label
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
                    let ret_ty = self.current_function_return_type.clone().unwrap_or(Type::Int);
                    let return_val = self.gen_expr_typed(expr, function, &ret_ty)?;
                    match return_val {
                        BasicValueEnum::IntValue(v) => { self.builder.build_return(Some(&v)); },
                        BasicValueEnum::FloatValue(v) => { self.builder.build_return(Some(&v)); },
                        BasicValueEnum::PointerValue(v) => { self.builder.build_return(Some(&v)); },
                        _ => return Err("Unsupported return value type".to_string()),
                    }
                } else {
                    // Return default zero for integer functions
                    let ret_ty = self.current_function_return_type.clone();
                    if let Some(t) = ret_ty {
                        let llvm_ty = self.get_llvm_type(&t)?;
                        let zero_val = match llvm_ty {
                            inkwell::types::BasicTypeEnum::IntType(it) => it.const_int(0, false).into(),
                            inkwell::types::BasicTypeEnum::FloatType(ft) => ft.const_float(0.0).into(),
                            inkwell::types::BasicTypeEnum::PointerType(pt) => pt.const_null().into(),
                            _ => return Err("Unsupported default return type".to_string()),
                        };
                        match zero_val {
                            BasicValueEnum::IntValue(v) => { self.builder.build_return(Some(&v)); },
                            BasicValueEnum::FloatValue(v) => { self.builder.build_return(Some(&v)); },
                            BasicValueEnum::PointerValue(v) => { self.builder.build_return(Some(&v)); },
                            _ => return Err("Unsupported default return type".to_string()),
                        }
                    } else {
                    let zero = self.context.i32_type().const_int(0, false);
                    self.builder.build_return(Some(&zero));
                    }
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
            // No wildcard arm: all Stmt variants are handled above
        }
        Ok(())
    }

    fn gen_expr(&mut self, expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
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
                    Type::String => {
                        let ptr_val = loaded_val.into_pointer_value();
                        Ok(self.builder.build_ptr_to_int(ptr_val, self.context.i32_type(), "str_to_int"))
                    },
                    Type::Float => {
                        let float_val = loaded_val.into_float_value();
                        Ok(self.builder.build_float_to_signed_int(float_val, self.context.i32_type(), "float_to_int"))
                    },
                    Type::List(_) | Type::Dict(_, _) => {
                        let ptr_val = loaded_val.into_pointer_value();
                        Ok(self.builder.build_ptr_to_int(ptr_val, self.context.i32_type(), "coll_ptr_to_int"))
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
            Expr::IndexAccess(index_access) => {
                // Desugar to list_get or dict_get based on the object's type when possible
                match &*index_access.object {
                    Expr::Identifier(name) => {
                        if let Some((_ptr, var_ty)) = self.named_values.get(name) {
                            match var_ty {
                                Type::List(_) => {
                                    return self.gen_function_call(&FunctionCallExpr { name: "list_get".to_string(), args: vec![*index_access.object.clone(), *index_access.index.clone()] }, function);
                                }
                                Type::Dict(_, _) => {
                                    return self.gen_function_call(&FunctionCallExpr { name: "dict_get".to_string(), args: vec![*index_access.object.clone(), *index_access.index.clone()] }, function);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
                Err("Indexing currently supports identifiers of list or dict".to_string())
            }
            Expr::InterpolatedString(interp) => {
                // Build a printf call with a composite format string and collected args
                // Assemble format string
                let mut format_str = String::new();
                let mut printf_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
                for seg in &interp.segments {
                    match seg {
                        crate::ast::InterpolatedSegment::Text(t) => {
                            format_str.push_str(t);
                        }
                        crate::ast::InterpolatedSegment::Expr(e) => {
                            // Choose placeholder by inferred type
                            let ty = self.infer_type_from_expr_for_interpolation(e);
                            match ty {
                                Type::Float => {
                                    format_str.push_str("%f");
                                    let v = self.gen_expr_typed(e, function, &Type::Float)?;
                                    printf_args.push(v.into());
                                }
                                Type::Char => {
                                    format_str.push_str("%c");
                                    let v = self.gen_expr_typed(e, function, &Type::Char)?;
                                    // widen char to i32 for printf
                                    let c8 = v.into_int_value();
                                    let c32 = self.builder.build_int_z_extend(c8, self.context.i32_type(), "c_to_i32");
                                    printf_args.push(c32.into());
                                }
                                Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                    format_str.push_str("%s");
                                    let v = self.gen_expr_typed(e, function, &Type::String)?;
                                    printf_args.push(v.into());
                                }
                                _ => {
                                    format_str.push_str("%d");
                                    let v = self.gen_expr(e, function)?;
                                    printf_args.push(v.into());
                                }
                            }
                        }
                    }
                }
                // add newline like print does
                format_str.push('\n');
                let fmt_ptr = self.builder.build_global_string_ptr(&format_str, ".fmt").as_pointer_value();
                let mut args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![fmt_ptr.into()];
                args.extend(printf_args);
                self.builder.build_call(self.external_functions.printf, &args, "printf_interp");
                Ok(self.context.i32_type().const_int(0, false))
            }
            Expr::MemberAccess(member_access) => {
                self.gen_member_access(member_access, function)
            }
            Expr::ObjectInstantiation(obj_inst) => {
                self.gen_object_instantiation(obj_inst, function)
            }
        }
    }

    fn gen_expr_as_pointer(&mut self, expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        match expr {
            Expr::FunctionCall(func_call) => {
                // Check if this is a class instantiation
                if self.class_definitions.contains_key(&func_call.name) {
                    let constructor_name = format!("{}_new", func_call.name);
                    if let Some(&constructor_func) = self.user_functions.get(&constructor_name) {
                        // Generate arguments based on constructor signature for correct types
                        let mut args = Vec::new();
                        if let Some((param_tys, _ret_ty)) = self.user_function_signatures.get(&constructor_name).cloned() {
                            for (idx, arg_expr) in func_call.args.iter().enumerate() {
                                let target_ty = param_tys.get(idx).cloned().unwrap_or(Type::Int);
                                let arg_val = self.gen_expr_typed(arg_expr, function, &target_ty)?;
                                args.push(arg_val.into());
                            }
                        } else {
                        for arg_expr in &func_call.args {
                            let arg_val = self.gen_expr(arg_expr, function)?;
                            args.push(arg_val.into());
                            }
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

    fn gen_function_call(&mut self, func_call: &FunctionCallExpr, _function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        // Check if this is a member assignment first
        if func_call.name == "MEMBER_ASSIGN" {
            return self.gen_member_assignment(&func_call.args, _function);
        }
        // Index assignment sugar: INDEX_ASSIGN(coll, idx, val)
        if func_call.name == "INDEX_ASSIGN" {
            if func_call.args.len() != 3 { return Err("INDEX_ASSIGN requires 3 args".to_string()); }
            // Determine list or dict by collection type
            match &func_call.args[0] {
                Expr::Identifier(name) => {
                    if let Some((_ptr, var_ty)) = self.named_values.get(name) {
                        return match var_ty {
                            Type::List(_) => self.gen_builtin_list_set(&func_call.args[0], &func_call.args[1], &func_call.args[2], _function),
                            Type::Dict(_, _) => self.gen_builtin_dict_set(&func_call.args[0], &func_call.args[1], &func_call.args[2], _function),
                            _ => Err("INDEX_ASSIGN expects list or dict".to_string()),
                        };
                    } else {
                        return Err("Unknown variable".to_string());
                    }
                }
                _ => return Err("INDEX_ASSIGN currently supports identifiers only".to_string()),
            }
        }

        // Builtins dispatched via registry
        if let Some(op) = builtin_lookup(&func_call.name) {

            return match op {
                BuiltinOp::Argc => {
                    // return process arg count as i32
                    Ok(self.context.i32_type().const_int(std::env::args().count() as u64, false))
                }
                BuiltinOp::Argv => {
                    // argv(index) -> int: if used in int context, return pointer as int
                    if func_call.args.len() != 1 { return Err("argv requires 1 argument".to_string()); }
                    let idx = self.gen_expr(&func_call.args[0], _function)?;
                    if let Some(idx_const_int) = idx.get_zero_extended_constant() {
                        let i = idx_const_int as usize;
                        if i < self.argv_globals.len() {
                            let gv = self.argv_globals[i];
                            let zero = self.context.i32_type().const_int(0, false);
                            let cstr_ptr = unsafe { self.builder.build_gep(gv.as_pointer_value(), &[zero, zero], "argv_gep") };
                            return Ok(self.builder.build_ptr_to_int(cstr_ptr, self.context.i32_type(), "argv_pi"));
                        }
                    }
                    Ok(self.context.i32_type().const_int(0, false))
                }
                BuiltinOp::PrintStr => {
                    if func_call.args.len() != 1 { return Err("print_str requires 1 argument".to_string()); }
                    // Accept either a string literal, identifier of string, or argv(i) int-ptr
                    match &func_call.args[0] {
                        Expr::Literal(LiteralExpr::String(s)) => {
                            let global_str = self.builder.build_global_string_ptr(s, ".arg");
                            let fmt = self.builder.build_global_string_ptr("%s\n", ".fmt").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), global_str.as_pointer_value().into()], "printf");
                        }
                        Expr::Identifier(name) => {
                            if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                                if matches!(var_ty, Type::String) {
                                    let s_ptr = self.builder.build_load(*var_ptr, name).into_pointer_value();
                                    let fmt = self.builder.build_global_string_ptr("%s\n", ".fmt").as_pointer_value();
                                    self.builder.build_call(self.external_functions.printf, &[fmt.into(), s_ptr.into()], "printf");
                                } else {
                                    return Err("print_str expects string pointer".to_string());
                                }
                            }
                        }
                        Expr::FunctionCall(inner) if inner.name == "argv" => {
                            // Reuse argv handling to get pointer
                            let ptr_i32 = self.gen_function_call(inner, _function)?;
                            let ptr = self.builder.build_int_to_ptr(ptr_i32, self.context.i8_type().ptr_type(inkwell::AddressSpace::default()), "argv_ptr");
                            let fmt = self.builder.build_global_string_ptr("%s\n", ".fmt").as_pointer_value();
                            self.builder.build_call(self.external_functions.printf, &[fmt.into(), ptr.into()], "printf");
                        }
                        _ => return Err("print_str expects string or argv(i)".to_string()),
                    }
                    Ok(self.context.i32_type().const_int(0, false))
                }
                BuiltinOp::Len => {
                    if func_call.args.len() != 1 { Err("len requires 1 argument".to_string()) } else { self.gen_builtin_len(&func_call.args[0], _function) }
                }
                BuiltinOp::ListFill => {
                    Err("list_fill not implemented yet".to_string())
                }
                BuiltinOp::ListAddRange => {
                    Err("list_add_range not implemented yet".to_string())
                }
                BuiltinOp::ListGet => {
                    if func_call.args.len() != 2 { Err("list_get requires 2 arguments".to_string()) } else { self.gen_builtin_list_get(&func_call.args[0], &func_call.args[1], _function) }
                }
                BuiltinOp::ListSet => {
                    if func_call.args.len() != 3 { Err("list_set requires 3 arguments".to_string()) } else { self.gen_builtin_list_set(&func_call.args[0], &func_call.args[1], &func_call.args[2], _function) }
                }
                BuiltinOp::DictGet => {
                    if func_call.args.len() != 2 { Err("dict_get requires 2 arguments".to_string()) } else { self.gen_builtin_dict_get(&func_call.args[0], &func_call.args[1], _function) }
                }
                BuiltinOp::DictSet => {
                    if func_call.args.len() != 3 { Err("dict_set requires 3 arguments".to_string()) } else { self.gen_builtin_dict_set(&func_call.args[0], &func_call.args[1], &func_call.args[2], _function) }
                }
            };
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
                // Generate arguments based on constructor signature
                let mut args = Vec::new();
                if let Some((param_tys, _)) = self.user_function_signatures.get(&constructor_name).cloned() {
                    for (idx, arg_expr) in func_call.args.iter().enumerate() {
                        let target_ty = param_tys.get(idx).cloned().unwrap_or(Type::Int);
                        let arg_val = self.gen_expr_typed(arg_expr, _function, &target_ty)?;
                        args.push(arg_val.into());
                    }
                } else {
                    for arg_expr in &func_call.args {
                            let arg_val = self.gen_expr(arg_expr, _function)?;
                            args.push(arg_val.into());
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
            // Generate arguments for user function using signature if available
            let mut args = Vec::new();
            if let Some((param_tys, _ret)) = self.user_function_signatures.get(&func_call.name).cloned() {
                for (idx, arg_expr) in func_call.args.iter().enumerate() {
                    let target_ty = param_tys.get(idx).cloned().unwrap_or(Type::Int);
                    let arg_val = self.gen_expr_typed(arg_expr, _function, &target_ty)?;
                    args.push(arg_val.into());
                }
            } else {
                for arg_expr in &func_call.args {
                        let arg_val = self.gen_expr(arg_expr, _function)?;
                        args.push(arg_val.into());
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

        // If we know the signature, coerce arguments accordingly
        let mut args = Vec::new();
        if let Some((param_types, ret_ty)) = get_function_signature(&func_call.name) {
            for (idx, arg_expr) in func_call.args.iter().enumerate() {
                let expected = param_types.get(idx).copied().unwrap_or("i32");
                match expected {
                    "i8_ptr" => {
                        match arg_expr {
                            Expr::Literal(LiteralExpr::String(s)) => {
                                let ptr = self.builder.build_global_string_ptr(s, ".str").as_pointer_value();
                                args.push(ptr.into());
                            }
                            _ => {
                                let int_val = self.gen_expr(arg_expr, _function)?;
                                let ptr_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                                let ptr = self.builder.build_int_to_ptr(int_val, ptr_ty, "int_to_ptr");
                                args.push(ptr.into());
                            }
                        }
                    }
                    "f64" => {
                        let int_val = self.gen_expr(arg_expr, _function)?;
                        let float_val = self.builder.build_signed_int_to_float(int_val, self.context.f64_type(), "int_to_f64");
                        args.push(float_val.into());
                    }
                    "bool" => {
                        let int_val = self.gen_expr(arg_expr, _function)?;
                        let i1 = self.builder.build_int_truncate_or_bit_cast(int_val, self.context.bool_type(), "to_bool");
                        args.push(i1.into());
                    }
                    _ => {
                        // default to i32
                        match arg_expr {
                            Expr::Literal(LiteralExpr::String(s)) => {
                                let ptr = self.builder.build_global_string_ptr(s, ".str").as_pointer_value();
                                let int_val = self.builder.build_ptr_to_int(ptr, self.context.i32_type(), "ptr_to_int");
                                args.push(int_val.into());
                            }
                            _ => {
                                let int_val = self.gen_expr(arg_expr, _function)?;
                                args.push(int_val.into());
                            }
                        }
                    }
                }
            }
            // Call the function
            let call_result = self.builder.build_call(stdlib_func, &args, "call");
            // Coerce return
            return Ok(match ret_ty {
                "void" => self.context.i32_type().const_int(0, false),
                "i32" => call_result.try_as_basic_value().left().unwrap().into_int_value(),
                "i8_ptr" => {
                    let ptr = call_result.try_as_basic_value().left().unwrap().into_pointer_value();
                    self.builder.build_ptr_to_int(ptr, self.context.i32_type(), "ptr_to_int_ret")
                }
                "f64" => {
                    let f = call_result.try_as_basic_value().left().unwrap().into_float_value();
                    self.builder.build_float_to_signed_int(f, self.context.i32_type(), "f64_to_i32")
                }
                "bool" => {
                    let b = call_result.try_as_basic_value().left().unwrap().into_int_value();
                    self.builder.build_int_z_extend_or_bit_cast(b, self.context.i32_type(), "bool_to_i32")
                }
                _ => return Err("Unsupported function return type".to_string()),
            });
        }

        // Fallback: best-effort old behavior
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
                    let expr_val = self.gen_expr(arg_expr, _function)?;
                    args.push(expr_val.into());
                }
            }
        }
        let call_result = self.builder.build_call(stdlib_func, &args, "call");
        Ok(call_result.try_as_basic_value().left().map(|v| v.into_int_value()).unwrap_or_else(|| self.context.i32_type().const_int(0, false)))
    }

    // Builtin: len(list|dict)
    fn gen_builtin_len(&mut self, coll_expr: &Expr, _function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        match coll_expr {
            Expr::Identifier(name) => {
                if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                    match var_ty {
                        Type::List(_) | Type::Dict(_, _) => {
                            let base_i8 = self.builder.build_load(*var_ptr, name).into_pointer_value();
                            let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "len_ptr_i8") };
                            let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "len_ptr").into_pointer_value();
                            let len = self.builder.build_load(len_ptr, "len").into_int_value();
                            Ok(len)
                        }
                        _ => Err("len expects list or dict".to_string())
                    }
                } else {
                    Err("Unknown variable".to_string())
                }
            }
            _ => Err("len currently supports identifiers only".to_string())
        }
    }

    // Builtin: list_get(list, index)
    fn gen_builtin_list_get(&mut self, list_expr: &Expr, index_expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        let (base_i8, elem_ty) = match list_expr {
            Expr::Identifier(name) => {
                if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                    if let Type::List(elem) = var_ty { (self.builder.build_load(*var_ptr, name).into_pointer_value(), (*elem).clone()) } else { return Err("list_get expects a list".to_string()); }
                } else { return Err("Unknown variable".to_string()); }
            }
            _ => return Err("list_get currently supports identifiers only".to_string()),
        };
        let idx_i32 = self.gen_expr(index_expr, function)?;
        let data_off = self.context.i32_type().const_int(8, false);
        let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
        let llvm_elem_ty = self.get_llvm_type(&elem_ty)?;
        let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();
        let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[idx_i32], "elem_ptr") };
        let loaded = self.builder.build_load(elem_ptr, "elem");
        // Convert to int for expression compatibility
        let int_val = match *elem_ty {
            Type::Float => {
                let f = loaded.into_float_value();
                self.builder.build_float_to_signed_int(f, self.context.i32_type(), "f_to_i")
            },
            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                let p = loaded.into_pointer_value();
                self.builder.build_ptr_to_int(p, self.context.i32_type(), "p_to_i")
            },
            Type::Char => {
                let c8 = loaded.into_int_value();
                self.builder.build_int_z_extend(c8, self.context.i32_type(), "c_to_i")
            },
            _ => loaded.into_int_value(),
        };
        Ok(int_val)
    }

    // Builtin: list_set(list, index, value)
    fn gen_builtin_list_set(&mut self, list_expr: &Expr, index_expr: &Expr, value_expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        let (base_i8, elem_ty) = match list_expr {
            Expr::Identifier(name) => {
                if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                    if let Type::List(elem) = var_ty { (self.builder.build_load(*var_ptr, name).into_pointer_value(), (*elem).clone()) } else { return Err("list_set expects a list".to_string()); }
                } else { return Err("Unknown variable".to_string()); }
            }
            _ => return Err("list_set currently supports identifiers only".to_string()),
        };
        let idx_i32 = self.gen_expr(index_expr, function)?;
        let data_off = self.context.i32_type().const_int(8, false);
        let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
        let llvm_elem_ty = self.get_llvm_type(&elem_ty)?;
        let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();
        let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[idx_i32], "elem_ptr") };
        let typed_val = self.gen_expr_typed(value_expr, function, &elem_ty)?;
        self.builder.build_store(elem_ptr, typed_val);
        // Return assigned value as int representation
        let ret = match *elem_ty {
            Type::Float => {
                let f = typed_val.into_float_value();
                self.builder.build_float_to_signed_int(f, self.context.i32_type(), "f_to_i")
            },
            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                let p = typed_val.into_pointer_value();
                self.builder.build_ptr_to_int(p, self.context.i32_type(), "p_to_i")
            },
            Type::Char => {
                let c8 = typed_val.into_int_value();
                self.builder.build_int_z_extend(c8, self.context.i32_type(), "c_to_i")
            },
            _ => typed_val.into_int_value(),
        };
        Ok(ret)
    }

    // Builtin: dict_get(dict, key) — linear scan
    fn gen_builtin_dict_get(&mut self, dict_expr: &Expr, key_expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        let (base_i8, key_ty, val_ty) = match dict_expr {
            Expr::Identifier(name) => {
                if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                    if let Type::Dict(k, v) = var_ty { (self.builder.build_load(*var_ptr, name).into_pointer_value(), (*k).clone(), (*v).clone()) } else { return Err("dict_get expects a dict".to_string()); }
                } else { return Err("Unknown variable".to_string()); }
            }
            _ => return Err("dict_get currently supports identifiers only".to_string()),
        };

        // Load length
        let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "d_len_ptr_i8") };
        let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "d_len_ptr").into_pointer_value();
        let len = self.builder.build_load(len_ptr, "d_len").into_int_value();

        // Prepare sizes and bases
        let llvm_key_ty = self.get_llvm_type(&key_ty)?;
        let llvm_val_ty = self.get_llvm_type(&val_ty)?;
        let key_sz = llvm_key_ty.size_of().ok_or("Failed to get key size")?; // i64
        let val_sz = llvm_val_ty.size_of().ok_or("Failed to get val size")?; // i64
        let pair_sz = self.builder.build_int_add(key_sz, val_sz, "pair_sz");
        let pairs_base_off = self.context.i32_type().const_int(8, false);
        let pairs_base_i8 = unsafe { self.builder.build_gep(base_i8, &[pairs_base_off], "pairs_base_i8") };

        // result alloca (i32)
        let result_alloca = self.builder.build_alloca(self.context.i32_type(), "dict_get_result");
        self.builder.build_store(result_alloca, self.context.i32_type().const_int(0, false));

        // loop index alloca
        let idx_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
        self.builder.build_store(idx_alloca, self.context.i32_type().const_int(0, false));

        // blocks
        let parent = function;
        let header_bb = self.context.append_basic_block(parent, "dict_get_header");
        let body_bb = self.context.append_basic_block(parent, "dict_get_body");
        let exit_bb = self.context.append_basic_block(parent, "dict_get_exit");
        self.builder.build_unconditional_branch(header_bb);
        self.builder.position_at_end(header_bb);
        let i_val = self.builder.build_load(idx_alloca, "i").into_int_value();
        let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i<len");
        self.builder.build_conditional_branch(cond, body_bb, exit_bb);

        // body
        self.builder.position_at_end(body_bb);
        let i_i64 = self.builder.build_int_cast(i_val, self.context.i64_type(), "i_i64");
        let off_i64 = self.builder.build_int_mul(pair_sz, i_i64, "pair_off");
        let off_i32 = self.builder.build_int_cast(off_i64, self.context.i32_type(), "pair_off_i32");
        let kv_base_i8 = unsafe { self.builder.build_gep(pairs_base_i8, &[off_i32], "kv_base_i8") };
        let key_ptr = self.builder.build_bitcast(kv_base_i8, llvm_key_ty.ptr_type(inkwell::AddressSpace::default()), "key_ptr").into_pointer_value();
        let cur_key = self.builder.build_load(key_ptr, "cur_key");

        // compute equality
        let target_key_val = self.gen_expr_typed(key_expr, function, &key_ty)?;
        let eq_flag = match *key_ty {
            Type::Int | Type::Bool | Type::Char => {
                let a = cur_key.into_int_value();
                let b = target_key_val.into_int_value();
                self.builder.build_int_compare(inkwell::IntPredicate::EQ, a, b, "k_eq")
            },
            Type::Float => {
                let a = cur_key.into_float_value();
                let b = target_key_val.into_float_value();
                self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, a, b, "k_eqf")
            },
            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                let a = cur_key.into_pointer_value();
                let b = target_key_val.into_pointer_value();
                self.builder.build_int_compare(inkwell::IntPredicate::EQ, self.builder.build_ptr_to_int(a, self.context.i32_type(), "a_pi"), self.builder.build_ptr_to_int(b, self.context.i32_type(), "b_pi"), "k_eqp")
            },
        };

        // if equal, load value and store result
        let then_bb = self.context.append_basic_block(parent, "dict_get_then");
        let cont_bb = self.context.append_basic_block(parent, "dict_get_cont");
        self.builder.build_conditional_branch(eq_flag, then_bb, cont_bb);

        // then: load value, convert to int, store to result, jump to exit
        self.builder.position_at_end(then_bb);
        let val_off_i32 = self.builder.build_int_cast(key_sz, self.context.i32_type(), "val_off_i32");
        let val_ptr_i8 = unsafe { self.builder.build_gep(kv_base_i8, &[val_off_i32], "val_ptr_i8") };
        let val_ptr = self.builder.build_bitcast(val_ptr_i8, llvm_val_ty.ptr_type(inkwell::AddressSpace::default()), "val_ptr").into_pointer_value();
        let v_loaded = self.builder.build_load(val_ptr, "v");
        let v_int = match *val_ty {
            Type::Float => {
                let f = v_loaded.into_float_value();
                self.builder.build_float_to_signed_int(f, self.context.i32_type(), "f_to_i")
            },
            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                let p = v_loaded.into_pointer_value();
                self.builder.build_ptr_to_int(p, self.context.i32_type(), "p_to_i")
            },
            Type::Char => {
                let c8 = v_loaded.into_int_value();
                self.builder.build_int_z_extend(c8, self.context.i32_type(), "c_to_i")
            },
            _ => v_loaded.into_int_value(),
        };
        self.builder.build_store(result_alloca, v_int);
        self.builder.build_unconditional_branch(exit_bb);

        // cont: i++ and branch back to header
        self.builder.position_at_end(cont_bb);
        let next_i = self.builder.build_int_add(i_val, self.context.i32_type().const_int(1, false), "i_next");
        self.builder.build_store(idx_alloca, next_i);
        self.builder.build_unconditional_branch(header_bb);

        // exit
        self.builder.position_at_end(exit_bb);
        let res = self.builder.build_load(result_alloca, "res").into_int_value();
        Ok(res)
    }

    // Builtin: dict_set(dict, key, value) — updates existing key if found
    fn gen_builtin_dict_set(&mut self, dict_expr: &Expr, key_expr: &Expr, value_expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        let (base_i8, key_ty, val_ty) = match dict_expr {
            Expr::Identifier(name) => {
                if let Some((var_ptr, var_ty)) = self.named_values.get(name) {
                    if let Type::Dict(k, v) = var_ty { (self.builder.build_load(*var_ptr, name).into_pointer_value(), (*k).clone(), (*v).clone()) } else { return Err("dict_set expects a dict".to_string()); }
                } else { return Err("Unknown variable".to_string()); }
            }
            _ => return Err("dict_set currently supports identifiers only".to_string()),
        };

        // Load length and bases
        let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "d_len_ptr_i8") };
        let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "d_len_ptr").into_pointer_value();
        let len = self.builder.build_load(len_ptr, "d_len").into_int_value();

        let llvm_key_ty = self.get_llvm_type(&key_ty)?;
        let llvm_val_ty = self.get_llvm_type(&val_ty)?;
        let key_sz = llvm_key_ty.size_of().ok_or("Failed to get key size")?; // i64
        let val_sz = llvm_val_ty.size_of().ok_or("Failed to get val size")?; // i64
        let pair_sz = self.builder.build_int_add(key_sz, val_sz, "pair_sz");
        let pairs_base_off = self.context.i32_type().const_int(8, false);
        let pairs_base_i8 = unsafe { self.builder.build_gep(base_i8, &[pairs_base_off], "pairs_base_i8") };

        // loop index alloca
        let idx_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
        self.builder.build_store(idx_alloca, self.context.i32_type().const_int(0, false));

        // blocks
        let parent = function;
        let header_bb = self.context.append_basic_block(parent, "dict_set_header");
        let body_bb = self.context.append_basic_block(parent, "dict_set_body");
        let exit_bb = self.context.append_basic_block(parent, "dict_set_exit");
        self.builder.build_unconditional_branch(header_bb);
        self.builder.position_at_end(header_bb);
        let i_val = self.builder.build_load(idx_alloca, "i").into_int_value();
        let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i<len");
        self.builder.build_conditional_branch(cond, body_bb, exit_bb);

        // body
        self.builder.position_at_end(body_bb);
        let i_i64 = self.builder.build_int_cast(i_val, self.context.i64_type(), "i_i64");
        let off_i64 = self.builder.build_int_mul(pair_sz, i_i64, "pair_off");
        let off_i32 = self.builder.build_int_cast(off_i64, self.context.i32_type(), "pair_off_i32");
        let kv_base_i8 = unsafe { self.builder.build_gep(pairs_base_i8, &[off_i32], "kv_base_i8") };
        let key_ptr = self.builder.build_bitcast(kv_base_i8, llvm_key_ty.ptr_type(inkwell::AddressSpace::default()), "key_ptr").into_pointer_value();
        let cur_key = self.builder.build_load(key_ptr, "cur_key");

        let target_key_val = self.gen_expr_typed(key_expr, function, &key_ty)?;
        let eq_flag = match *key_ty {
            Type::Int | Type::Bool | Type::Char => {
                let a = cur_key.into_int_value();
                let b = target_key_val.into_int_value();
                self.builder.build_int_compare(inkwell::IntPredicate::EQ, a, b, "k_eq")
            },
            Type::Float => {
                let a = cur_key.into_float_value();
                let b = target_key_val.into_float_value();
                self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, a, b, "k_eqf")
            },
            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                let a = cur_key.into_pointer_value();
                let b = target_key_val.into_pointer_value();
                self.builder.build_int_compare(inkwell::IntPredicate::EQ, self.builder.build_ptr_to_int(a, self.context.i32_type(), "a_pi"), self.builder.build_ptr_to_int(b, self.context.i32_type(), "b_pi"), "k_eqp")
            },
        };

        let then_bb = self.context.append_basic_block(parent, "dict_set_then");
        let cont_bb = self.context.append_basic_block(parent, "dict_set_cont");
        self.builder.build_conditional_branch(eq_flag, then_bb, cont_bb);

        // then: store value and exit
        self.builder.position_at_end(then_bb);
        let val_off_i32 = self.builder.build_int_cast(key_sz, self.context.i32_type(), "val_off_i32");
        let val_ptr_i8 = unsafe { self.builder.build_gep(kv_base_i8, &[val_off_i32], "val_ptr_i8") };
        let val_ptr = self.builder.build_bitcast(val_ptr_i8, llvm_val_ty.ptr_type(inkwell::AddressSpace::default()), "val_ptr").into_pointer_value();
        let store_val = self.gen_expr_typed(value_expr, function, &val_ty)?;
        self.builder.build_store(val_ptr, store_val);
        self.builder.build_unconditional_branch(exit_bb);

        // cont: i++ back to header
        self.builder.position_at_end(cont_bb);
        let next_i = self.builder.build_int_add(i_val, self.context.i32_type().const_int(1, false), "i_next");
        self.builder.build_store(idx_alloca, next_i);
        self.builder.build_unconditional_branch(header_bb);

        // exit: return 0
        self.builder.position_at_end(exit_bb);
        Ok(self.context.i32_type().const_int(0, false))
    }

    fn declare_function(&mut self, func_decl: &FunctionDeclStmt) -> Result<(), String> {
        // Use declared parameter and return types
        let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();
        for (_name, ty) in &func_decl.params {
            param_types.push(self.get_llvm_type(ty)?.into());
        }
        let ret_ty = func_decl.return_type.clone().unwrap_or(Type::Int);
        let llvm_ret_ty = self.get_llvm_type(&ret_ty)?;
        let fn_type = match llvm_ret_ty {
            inkwell::types::BasicTypeEnum::IntType(t) => t.fn_type(&param_types, false),
            inkwell::types::BasicTypeEnum::FloatType(t) => t.fn_type(&param_types, false),
            inkwell::types::BasicTypeEnum::PointerType(t) => t.fn_type(&param_types, false),
            _ => return Err("Unsupported function return type".to_string()),
        };
        
        let function = self.module.add_function(&func_decl.name, fn_type, None);
        self.user_functions.insert(func_decl.name.clone(), function);
        self.user_function_signatures.insert(func_decl.name.clone(), (func_decl.params.iter().map(|(_, t)| t.clone()).collect(), func_decl.return_type.clone()));
        Ok(())
    }

    fn gen_function_body(&mut self, func_decl: &FunctionDeclStmt) -> Result<(), String> {
        let function = *self.user_functions.get(&func_decl.name).unwrap();
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // Save current named values (for nested scope)
        let saved_values = self.named_values.clone();
        let saved_ret = self.current_function_return_type.clone();
        self.current_function_return_type = func_decl.return_type.clone().or(Some(Type::Int));
        
        // Create allocas for parameters in entry block to avoid deep stack usage
        for (i, (param_name, param_type)) in func_decl.params.iter().enumerate() {
            let llvm_type = self.get_llvm_type(param_type)?;
            let alloca = self.build_entry_alloca(function, llvm_type, param_name);
            let param = function.get_nth_param(i as u32).unwrap();
            self.builder.build_store(alloca, param);
            self.named_values.insert(param_name.clone(), (alloca, param_type.clone()));
        }

        // Generate function body
        for stmt in &func_decl.body {
            self.gen_stmt(stmt, function)?;
        }

        // If no explicit return, return default zero for declared return type
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            let ret_ty = self.current_function_return_type.clone().unwrap_or(Type::Int);
            let llvm_ty = self.get_llvm_type(&ret_ty)?;
            match llvm_ty {
                inkwell::types::BasicTypeEnum::IntType(t) => {
                    let v = t.const_int(0, false);
                    self.builder.build_return(Some(&v));
                },
                inkwell::types::BasicTypeEnum::FloatType(t) => {
                    let v = t.const_float(0.0);
                    self.builder.build_return(Some(&v));
                },
                inkwell::types::BasicTypeEnum::PointerType(t) => {
                    let v = t.const_null();
                    self.builder.build_return(Some(&v));
                },
                _ => return Err("Unsupported default return type".to_string()),
            }
        }

        // Restore previous named values
        self.named_values = saved_values;
        self.current_function_return_type = saved_ret;
        
        Ok(())
    }

    fn gen_if_stmt(&mut self, if_stmt: &IfStmt, function: inkwell::values::FunctionValue<'ctx>) -> Result<(), String> {
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

    fn gen_while_stmt(&mut self, while_stmt: &WhileStmt, function: inkwell::values::FunctionValue<'ctx>) -> Result<(), String> {
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

    fn gen_for_stmt(&mut self, for_stmt: &ForStmt, function: inkwell::values::FunctionValue<'ctx>) -> Result<(), String> {
        // Support: for (i in range(start, end[, step])) {...}
        // and sugar: for (i in a..b) {...} (already parsed into range(a,b))
        if let Expr::FunctionCall(fc) = &for_stmt.iterable {
            if fc.name == "range" && (fc.args.len() == 2 || fc.args.len() == 3) {
                // Initialize i = start
                let start = self.gen_expr(&fc.args[0], function)?;
                let step = if fc.args.len() == 3 { self.gen_expr(&fc.args[2], function)? } else { self.context.i32_type().const_int(1, false) };
                let end = self.gen_expr(&fc.args[1], function)?;

                // Allocate loop variable
                let alloca = self.build_entry_alloca(function, self.context.i32_type().into(), &for_stmt.variable);
                self.builder.build_store(alloca, start);
                self.named_values.insert(for_stmt.variable.clone(), (alloca, Type::Int));

                // Blocks
                let header = self.context.append_basic_block(function, "for_header");
                let body = self.context.append_basic_block(function, "for_body");
                let exit = self.context.append_basic_block(function, "for_exit");
                self.builder.build_unconditional_branch(header);

                // Header: i < end
                self.builder.position_at_end(header);
                let i_val = self.builder.build_load(alloca, &for_stmt.variable).into_int_value();
                let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, end, "i_lt_end");
                self.builder.build_conditional_branch(cond, body, exit);

                // Body: loop body, then i += step
                self.builder.position_at_end(body);
                for s in &for_stmt.body {
                    self.gen_stmt(s, function)?;
                }
                let i_next = self.builder.build_int_add(i_val, step, "i_next");
                self.builder.build_store(alloca, i_next);
                self.builder.build_unconditional_branch(header);

                // Exit
                self.builder.position_at_end(exit);
                return Ok(());
            }
        }
        // Iteration over list: for (x in list) {...}
        if let Expr::Identifier(name) = &for_stmt.iterable {
            if let Some((list_ptr, list_ty)) = self.named_values.get(name) {
                if let Type::List(elem_ty_box) = list_ty {
                    // load base
                    let base_i8 = self.builder.build_load(*list_ptr, name).into_pointer_value();
                    // len at offset 0
                    let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "len_ptr_i8") };
                    let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "len_ptr").into_pointer_value();
                    let len = self.builder.build_load(len_ptr, "len").into_int_value();
                    // data pointer at offset 8
                    let data_off = self.context.i32_type().const_int(8, false);
                    let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
                    let llvm_elem_ty = self.get_llvm_type(&*elem_ty_box)?;
                    let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();

                    // allocate loop index and loop variable
                    let idx_alloca = self.build_entry_alloca(function, self.context.i32_type().into(), "i");
                    self.builder.build_store(idx_alloca, self.context.i32_type().const_int(0, false));

                    let iter_alloca = self.build_entry_alloca(function, llvm_elem_ty, &for_stmt.variable);
                    let iter_type: Type = elem_ty_box.as_ref().clone();
                    self.named_values.insert(for_stmt.variable.clone(), (iter_alloca, iter_type));

                    // blocks
                    let header = self.context.append_basic_block(function, "for_list_header");
                    let body = self.context.append_basic_block(function, "for_list_body");
                    let exit = self.context.append_basic_block(function, "for_list_exit");
                    self.builder.build_unconditional_branch(header);

                    // header: i < len
                    self.builder.position_at_end(header);
                    let i_val = self.builder.build_load(idx_alloca, "i").into_int_value();
                    let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "i_lt_len");
                    self.builder.build_conditional_branch(cond, body, exit);

                    // body: load element to iter var; gen body; i++
                    self.builder.position_at_end(body);
                    let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[i_val], "elem_ptr") };
                    let elem = self.builder.build_load(elem_ptr, "elem");
                    self.builder.build_store(iter_alloca, elem);
                    for s in &for_stmt.body { self.gen_stmt(s, function)?; }
                    let next_i = self.builder.build_int_add(i_val, self.context.i32_type().const_int(1, false), "i_next");
                    self.builder.build_store(idx_alloca, next_i);
                    self.builder.build_unconditional_branch(header);

                    self.builder.position_at_end(exit);
                    return Ok(());
                }
            }
        }

        Err("Unsupported for-iterable; expected range(start,end[,step]) or list".to_string())
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

    fn gen_expr_for_print(&mut self, expr: &Expr, function: inkwell::values::FunctionValue<'ctx>) -> Result<BasicValueEnum<'ctx>, String> {
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
                let (var_ptr, var_type) = self.named_values.get(name).ok_or("Unknown variable")?;
                let loaded_val = self.builder.build_load(*var_ptr, name);
                if matches!(var_type, Type::Char) {
                    let c8 = loaded_val.into_int_value();
                    let c32 = self.builder.build_int_z_extend(c8, self.context.i32_type(), "char_to_i32");
                    Ok(c32.into())
                } else {
                Ok(loaded_val)
                }
            },
            _ => {
                let int_val = self.gen_expr(expr, function)?;
                Ok(int_val.into())
            }
        }
    }

    fn gen_expr_typed(&mut self, expr: &Expr, function: inkwell::values::FunctionValue<'ctx>, target_type: &Type) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::Binary(bin) => {
                // Support [x] * N when assigned to list<T>
                if let Type::List(elem_ty_box) = target_type {
                    if matches!(bin.op, BinaryOp::Multiply) {
                        // Identify pattern: (array_literal) * count  OR count * (array_literal)
                        let (elements_opt, count_expr) = match (&*bin.left, &*bin.right) {
                            (Expr::Literal(LiteralExpr::Array(elements)), rhs) => (Some(elements), rhs),
                            (lhs, Expr::Literal(LiteralExpr::Array(elements))) => (Some(elements), lhs),
                            _ => (None, &*bin.right),
                        };

                        if let Some(elements) = elements_opt {
                            if elements.len() == 1 {
                                let elem_ty = (*elem_ty_box).clone();
                                let llvm_elem_ty = self.get_llvm_type(&elem_ty)?;
                                // Compute count
                                let count_i32 = self.gen_expr(count_expr, function)?;

                                // total bytes = 8 (len+cap) + elem_size * count
                                let elem_size_i64 = llvm_elem_ty.size_of().ok_or("Failed to get element size")?;
                                let count_i64 = self.builder.build_int_cast(count_i32, self.context.i64_type(), "count_i64");
                                let elems_bytes = self.builder.build_int_mul(elem_size_i64, count_i64, "repeat_bytes");
                                let header_bytes = self.context.i64_type().const_int(8, false);
                                let total_i64 = self.builder.build_int_add(header_bytes, elems_bytes, "total_bytes");
                                let total_i32 = self.builder.build_int_cast(total_i64, self.context.i32_type(), "total_i32");
                                let malloc_call = self.builder.build_call(self.external_functions.malloc, &[total_i32.into()], "malloc_list_repeat");
                                let base_i8 = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();

                                // Store len and cap
                                let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "len_ptr_i8") };
                                let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "len_ptr").into_pointer_value();
                                self.builder.build_store(len_ptr, count_i32);
                                let cap_off = self.context.i32_type().const_int(4, false);
                                let cap_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[cap_off], "cap_ptr_i8") };
                                let cap_ptr = self.builder.build_bitcast(cap_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "cap_ptr").into_pointer_value();
                                self.builder.build_store(cap_ptr, count_i32);

                                // Data pointer
                                let data_off = self.context.i32_type().const_int(8, false);
                                let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
                                let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();

                                // Precompute element value once
                                let repeated_val = self.gen_expr_typed(&elements[0], function, &elem_ty)?;

                                // Loop i from 0..count to fill
                                let idx_alloca = self.builder.build_alloca(self.context.i32_type(), "i");
                                self.builder.build_store(idx_alloca, self.context.i32_type().const_int(0, false));

                                let parent = function;
                                let header_bb = self.context.append_basic_block(parent, "rep_header");
                                let body_bb = self.context.append_basic_block(parent, "rep_body");
                                let exit_bb = self.context.append_basic_block(parent, "rep_exit");
                                self.builder.build_unconditional_branch(header_bb);

                                self.builder.position_at_end(header_bb);
                                let i_val = self.builder.build_load(idx_alloca, "i").into_int_value();
                                let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, count_i32, "i_lt_n");
                                self.builder.build_conditional_branch(cond, body_bb, exit_bb);

                                self.builder.position_at_end(body_bb);
                                let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[i_val], "elem_ptr") };
                                self.builder.build_store(elem_ptr, repeated_val);
                                let next_i = self.builder.build_int_add(i_val, self.context.i32_type().const_int(1, false), "i_next");
                                self.builder.build_store(idx_alloca, next_i);
                                self.builder.build_unconditional_branch(header_bb);

                                self.builder.position_at_end(exit_bb);
                                return Ok(base_i8.into());
                            }
                        }
                    }
                }
                // Fallback to default handling when not a list repetition in typed context
                let int_val = self.gen_expr(expr, function)?;
                match target_type {
                    Type::Float => Ok(self.builder.build_signed_int_to_float(int_val, self.context.f64_type(), "int_to_float").into()),
                    _ => Ok(int_val.into())
                }
            }
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
                    // Proper list layout: [i32 len][i32 cap][T elements...]
                    if let Type::List(elem_ty_box) = target_type {
                        let elem_ty = (*elem_ty_box).clone();
                        let llvm_elem_ty = self.get_llvm_type(&elem_ty)?;
                        let elem_size = llvm_elem_ty.size_of().ok_or("Failed to get element size")?; // i64
                        let len_u64 = elements.len() as u64;
                        let len_i32 = self.context.i32_type().const_int(len_u64, false);
                        let cap_i32 = len_i32;
                        let header_bytes = self.context.i64_type().const_int(8, false);
                        let elem_count_i64 = self.context.i64_type().const_int(len_u64, false);
                        let elems_bytes_i64 = self.builder.build_int_mul(elem_size, elem_count_i64, "list_elems_bytes");
                        let total_i64 = self.builder.build_int_add(header_bytes, elems_bytes_i64, "list_total_bytes");
                        let total_i32 = self.builder.build_int_cast(total_i64, self.context.i32_type(), "list_total_i32");
                        let malloc_call = self.builder.build_call(self.external_functions.malloc, &[total_i32.into()], "malloc_list");
                        let base_i8 = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();

                        // Store len at offset 0
                        let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "len_ptr_i8") };
                        let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "len_ptr").into_pointer_value();
                        self.builder.build_store(len_ptr, len_i32);

                        // Store cap at offset 4
                        let cap_off = self.context.i32_type().const_int(4, false);
                        let cap_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[cap_off], "cap_ptr_i8") };
                        let cap_ptr = self.builder.build_bitcast(cap_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "cap_ptr").into_pointer_value();
                        self.builder.build_store(cap_ptr, cap_i32);

                        // Data pointer at offset 8, cast to element*
                        let data_off = self.context.i32_type().const_int(8, false);
                        let data_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[data_off], "data_ptr_i8") };
                        let data_ptr = self.builder.build_bitcast(data_ptr_i8, llvm_elem_ty.ptr_type(inkwell::AddressSpace::default()), "data_ptr").into_pointer_value();

                        // Fill elements
                    for (i, element) in elements.iter().enumerate() {
                            let idx = self.context.i32_type().const_int(i as u64, false);
                            let elem_ptr = unsafe { self.builder.build_gep(data_ptr, &[idx], "elem_ptr") };
                            let elem_val = self.gen_expr_typed(element, function, &elem_ty)?;
                            self.builder.build_store(elem_ptr, elem_val);
                        }

                        Ok(base_i8.into())
                    } else {
                        // Not a list context: evaluate elements and ignore, return null pointer
                        let ptr_type = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                        Ok(ptr_type.const_null().into())
                    }
                },
                LiteralExpr::Dict(pairs) => {
                    // Proper dict layout (immutable default): [i32 len][padding 4][pairs...]
                    // Pair layout: [K][V] contiguous without extra alignment handling
                    if let Type::Dict(key_ty_box, val_ty_box) = target_type {
                        let key_ty = (*key_ty_box).clone();
                        let val_ty = (*val_ty_box).clone();
                        let llvm_key_ty = self.get_llvm_type(&key_ty)?;
                        let llvm_val_ty = self.get_llvm_type(&val_ty)?;
                        let key_sz = llvm_key_ty.size_of().ok_or("Failed to get key size")?; // i64
                        let val_sz = llvm_val_ty.size_of().ok_or("Failed to get val size")?; // i64
                        let pair_sz = self.builder.build_int_add(key_sz, val_sz, "pair_sz");
                        let count_i64 = self.context.i64_type().const_int(pairs.len() as u64, false);
                        let payload_sz = self.builder.build_int_mul(pair_sz, count_i64, "payload_sz");
                        let header_sz = self.context.i64_type().const_int(8, false); // len + padding
                        let total_i64 = self.builder.build_int_add(header_sz, payload_sz, "dict_total");
                        let total_i32 = self.builder.build_int_cast(total_i64, self.context.i32_type(), "dict_total_i32");
                        let malloc_call = self.builder.build_call(self.external_functions.malloc, &[total_i32.into()], "malloc_dict");
                        let base_i8 = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();

                        // Store len at offset 0
                        let len_i32 = self.context.i32_type().const_int(pairs.len() as u64, false);
                        let len_ptr_i8 = unsafe { self.builder.build_gep(base_i8, &[self.context.i32_type().const_int(0, false)], "d_len_ptr_i8") };
                        let len_ptr = self.builder.build_bitcast(len_ptr_i8, self.context.i32_type().ptr_type(inkwell::AddressSpace::default()), "d_len_ptr").into_pointer_value();
                        self.builder.build_store(len_ptr, len_i32);

                        // Base of pairs at offset 8
                        let pairs_off = self.context.i32_type().const_int(8, false);
                        let pairs_base_i8 = unsafe { self.builder.build_gep(base_i8, &[pairs_off], "pairs_base_i8") };

                        for (i, (k_expr, v_expr)) in pairs.iter().enumerate() {
                            let idx_i64 = self.context.i64_type().const_int(i as u64, false);
                            let pair_off_i64 = self.builder.build_int_mul(pair_sz, idx_i64, "pair_off_i64");
                            let pair_off_i32 = self.builder.build_int_cast(pair_off_i64, self.context.i32_type(), "pair_off_i32");
                            let kv_base_i8 = unsafe { self.builder.build_gep(pairs_base_i8, &[pair_off_i32], "kv_base_i8") };

                            // key at offset 0
                            let key_ptr = self.builder.build_bitcast(kv_base_i8, llvm_key_ty.ptr_type(inkwell::AddressSpace::default()), "key_ptr").into_pointer_value();
                            let key_val = self.gen_expr_typed(k_expr, function, &key_ty)?;
                            self.builder.build_store(key_ptr, key_val);

                            // value at offset key_sz
                            let val_off_i32 = self.builder.build_int_cast(key_sz, self.context.i32_type(), "val_off_i32");
                            let val_ptr_i8 = unsafe { self.builder.build_gep(kv_base_i8, &[val_off_i32], "val_ptr_i8") };
                            let val_ptr = self.builder.build_bitcast(val_ptr_i8, llvm_val_ty.ptr_type(inkwell::AddressSpace::default()), "val_ptr").into_pointer_value();
                            let val_val = self.gen_expr_typed(v_expr, function, &val_ty)?;
                            self.builder.build_store(val_ptr, val_val);
                        }

                        Ok(base_i8.into())
                    } else {
                        // Not a dict context
                    let ptr_type = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                    Ok(ptr_type.const_null().into())
                    }
                },
            },
            Expr::Identifier(name) => {
                let (var_ptr, var_ty) = self.named_values.get(name).ok_or("Unknown variable")?;
                let loaded = self.builder.build_load(*var_ptr, name);
                Ok(match target_type {
                    Type::Float => {
                        match var_ty {
                            Type::Float => loaded,
                            _ => self.builder.build_signed_int_to_float(loaded.into_int_value(), self.context.f64_type(), "int_to_float").into(),
                        }
                    },
                    Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => loaded,
                    Type::Char => loaded,
                    _ => {
                        match var_ty {
                            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                let p = loaded.into_pointer_value();
                                self.builder.build_ptr_to_int(p, self.context.i32_type(), "ptr_to_int").into()
                            },
                            Type::Float => self.builder.build_float_to_signed_int(loaded.into_float_value(), self.context.i32_type(), "float_to_int").into(),
                            _ => loaded,
                        }
                    }
                })
            }
            Expr::MemberAccess(member_access) => {
                // Load the member value and coerce to target type if necessary
                let loaded = self.gen_member_access_for_print(member_access, function)?;
                Ok(match target_type {
                    Type::Float => match loaded {
                        BasicValueEnum::FloatValue(_) => loaded,
                        BasicValueEnum::IntValue(iv) => self.builder.build_signed_int_to_float(iv, self.context.f64_type(), "int_to_float").into(),
                        BasicValueEnum::PointerValue(_) => return Err("Cannot cast pointer to float".to_string()),
                        _ => return Err("Unsupported member value for float".to_string()),
                    },
                    Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => loaded,
                    Type::Char => loaded,
                    _ => match loaded {
                        BasicValueEnum::IntValue(_) => loaded,
                        BasicValueEnum::FloatValue(fv) => self.builder.build_float_to_signed_int(fv, self.context.i32_type(), "float_to_int").into(),
                        BasicValueEnum::PointerValue(pv) => self.builder.build_ptr_to_int(pv, self.context.i32_type(), "ptr_to_int").into(),
                        _ => return Err("Unsupported member value".to_string()),
                    }
                })
            }
            Expr::FunctionCall(func_call) => {
                // If we are assigning to a custom type, attempt to get pointer from constructor call
                match target_type {
                    Type::Custom(class_name) => {
                        if let Ok(ptr) = self.gen_expr_as_pointer(&Expr::FunctionCall(func_call.clone()), function) {
                            Ok(ptr.into())
                        } else {
                            // Fallback: interpret as pointer to class or i8*
                            let int_val = self.gen_function_call(func_call, function)?;
                            if let Some(struct_ty) = self.class_types.get(class_name) {
                                let ptr_ty = struct_ty.ptr_type(inkwell::AddressSpace::default());
                                Ok(self.builder.build_int_to_ptr(int_val, ptr_ty, "int_to_class_ptr").into())
                            } else {
                                let i8ptr = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                                Ok(self.builder.build_int_to_ptr(int_val, i8ptr, "int_to_ptr").into())
                            }
                        }
                    }
                    Type::String | Type::List(_) | Type::Dict(_, _) => {
                        // Interpret function int return as an i8* pointer value
                        let int_val = self.gen_function_call(func_call, function)?;
                        let ptr_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                        Ok(self.builder.build_int_to_ptr(int_val, ptr_ty, "int_to_ptr_typed").into())
                    }
                    Type::Float => {
                        let int_val = self.gen_function_call(func_call, function)?;
                        Ok(self.builder.build_signed_int_to_float(int_val, self.context.f64_type(), "int_to_float").into())
                    }
                    _ => {
                        let int_val = self.gen_function_call(func_call, function)?;
                        Ok(int_val.into())
                    }
                }
            }
            _ => {
                // Fallback: use untyped int and coerce only for floats
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
        self.user_function_signatures.insert(format!("{}_new", class_name), (method.params.iter().map(|(_, t)| t.clone()).collect(), Some(Type::Custom(class_name.to_string()))));
        
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
        self.user_function_signatures.insert(format!("{}_{}", class_name, method.name), (std::iter::once(Type::Custom(class_name.to_string())).chain(method.params.iter().map(|(_, t)| t.clone())).collect(), method.return_type.clone()));
        
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
        let saved_ret = self.current_function_return_type.clone();
        self.current_function_return_type = Some(Type::Custom(class_name.to_string()));
        
        // Allocate memory for the class instance using malloc
        let class_type = *self.class_types.get(class_name).unwrap();
        let size = class_type.size_of().unwrap();
        // malloc expects i32 size; cast if needed
        let size_i32 = if size.get_type().get_bit_width() != 32 {
            self.builder.build_int_cast(size, self.context.i32_type(), "size_to_i32")
        } else { size };
        let malloc_call = self.builder.build_call(
            self.external_functions.malloc, 
            &[size_i32.into()], 
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
        self.current_function_return_type = saved_ret;
        Ok(())
    }

    fn gen_method_body(&mut self, class_name: &str, method: &FunctionDeclStmt) -> Result<(), String> {
        let function = *self.user_functions.get(&format!("{}_{}", class_name, method.name)).unwrap();
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let saved_values = self.named_values.clone();
        let saved_ret = self.current_function_return_type.clone();
        self.current_function_return_type = method.return_type.clone().or(Some(Type::Int));
        
        // First parameter is 'this'
        let this_param = function.get_nth_param(0).unwrap().into_pointer_value();
        self.named_values.insert("this".to_string(), (this_param, Type::Custom(class_name.to_string())));
        
        // Initialize other parameters
        for (i, (param_name, param_type)) in method.params.iter().enumerate() {
            let llvm_type = self.get_llvm_type(param_type)?;
            let param_alloca = self.build_entry_alloca(function, llvm_type, param_name);
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
        self.current_function_return_type = saved_ret;
        Ok(())
    }

    fn gen_member_access(&mut self, member_access: &MemberAccessExpr, _function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
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
                                    // Determine type to convert appropriately
                                    let member_ty = member.type_annotation.clone().unwrap_or(Type::Int);
                                    return Ok(match member_ty {
                                        Type::Float => {
                                            let f = loaded_val.into_float_value();
                                            self.builder.build_float_to_signed_int(f, self.context.i32_type(), "float_to_int")
                                        },
                                        Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                            let p = loaded_val.into_pointer_value();
                                            self.builder.build_ptr_to_int(p, self.context.i32_type(), "ptr_to_int")
                                        },
                                        Type::Char => {
                                            let c8 = loaded_val.into_int_value();
                                            self.builder.build_int_z_extend(c8, self.context.i32_type(), "char_to_i32")
                                        },
                                        _ => loaded_val.into_int_value(),
                                    });
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

    fn gen_object_instantiation(&mut self, obj_inst: &ObjectInstantiationExpr, function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
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

    fn gen_method_call(&mut self, method_name: &str, args: &[Expr], function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
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
                            let sig = self.user_function_signatures.get(&method_func_name).cloned();
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
                            if let Some((param_tys, _ret_ty)) = sig {
                                // param_tys includes 'this' at position 0
                                for (idx, arg_expr) in method_args.iter().enumerate() {
                                    let target_ty = param_tys.get(idx + 1).cloned().unwrap_or(Type::Int);
                                    let arg_val = self.gen_expr_typed(arg_expr, function, &target_ty)?;
                                    call_args.push(arg_val.into());
                                }
                            } else {
                                for arg_expr in method_args {
                                        let arg_val = self.gen_expr(arg_expr, function)?;
                                        call_args.push(arg_val.into());
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

    fn gen_member_assignment(&mut self, args: &[Expr], function: inkwell::values::FunctionValue<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, String> {
        if args.len() != 2 {
            return Err("Member assignment requires exactly 2 arguments".to_string());
        }
        
        let member_access = &args[0];
        let value_expr = &args[1];
        
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
                                        let member_ty = member.type_annotation.clone().unwrap_or(Type::Int);
                                        let typed_val = self.gen_expr_typed(value_expr, function, &member_ty)?;
                                        self.builder.build_store(member_ptr, typed_val);
                                        // Return an int representation of the assigned value for expression semantics
                                        let ret = match member_ty {
                                            Type::Float => {
                                                let f = typed_val.into_float_value();
                                                self.builder.build_float_to_signed_int(f, self.context.i32_type(), "float_to_int")
                                            },
                                            Type::String | Type::List(_) | Type::Dict(_, _) | Type::Custom(_) => {
                                                let p = typed_val.into_pointer_value();
                                                self.builder.build_ptr_to_int(p, self.context.i32_type(), "ptr_to_int")
                                            },
                                            Type::Char => {
                                                let c8 = typed_val.into_int_value();
                                                self.builder.build_int_z_extend(c8, self.context.i32_type(), "char_to_i32")
                                            },
                                            _ => typed_val.into_int_value(),
                                        };
                                        return Ok(ret);
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

    // Helper: get object class and member type for a member access
    fn get_object_and_member_type(&self, object_expr: &Expr, member_name: &str) -> Result<Option<(String, Type)>, String> {
        if let Expr::Identifier(obj_name) = object_expr {
            if let Some((_obj_ptr, obj_type)) = self.named_values.get(obj_name) {
                if let Type::Custom(class_name) = obj_type {
                    if let Some(class_def) = self.class_definitions.get(class_name) {
                        for member in &class_def.members {
                            if member.name == member_name {
                                let ty = member.type_annotation.clone().unwrap_or(Type::Int);
                                return Ok(Some((class_name.clone(), ty)));
                            }
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    // Typed load for printing member access
    fn gen_member_access_for_print(&mut self, member_access: &MemberAccessExpr, _function: inkwell::values::FunctionValue<'ctx>) -> Result<BasicValueEnum<'ctx>, String> {
        match &*member_access.object {
            Expr::Identifier(obj_name) => {
                if let Some((obj_var_ptr, obj_type)) = self.named_values.get(obj_name) {
                    if let Type::Custom(class_name) = obj_type {
                        if let Some(class_def) = self.class_definitions.get(class_name) {
                            let obj_ptr = if obj_name == "this" {
                                *obj_var_ptr
                            } else {
                                self.builder.build_load(*obj_var_ptr, obj_name).into_pointer_value()
                            };
                            for (i, member) in class_def.members.iter().enumerate() {
                                if member.name == member_access.member {
                                    let member_ptr = self.builder.build_struct_gep(obj_ptr, i as u32, &format!("{}_member", member.name)).unwrap();
                                    let loaded_val = self.builder.build_load(member_ptr, &member.name);
                                    return Ok(loaded_val);
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
}
