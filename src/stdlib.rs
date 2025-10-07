//! Standard library integration.
//!
//! How to add a new std function (one place to change):
//! 1) Add the function to `define_stdlib_functions!` with its C-level signature.
//!    This exposes it to LLVM and provides helper lookups.
//! 2) If the function is implemented natively in Rust (e.g., a shim), ensure the
//!    symbol is registered in `codegen.rs` (native -> JIT mapping). If it's a C
//!    or libc symbol, the declaration is sufficient and will be linked at runtime.
//! 3) Calls appear in the AST as `Expr::FunctionCall { name, args }` with `name`
//!    as a plain identifier. No lexer or parser changes are needed.
//!
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;
use inkwell::AddressSpace;

// Builtin registry (language-level functions implemented in codegen)
#[macro_export]
macro_rules! define_builtins {
    ($($name:literal => $variant:ident),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum BuiltinOp {
            $( $variant, )*
        }

        pub fn builtin_lookup(name: &str) -> Option<BuiltinOp> {
            match name {
                $( $name => Some(BuiltinOp::$variant), )*
                _ => None,
            }
        }
    }
}

// Instantiate builtin registry with current builtins
define_builtins! {
    "argc" => Argc,
    "argv" => Argv,
    "print_str" => PrintStr,
    "list_fill" => ListFill,
    "list_add_range" => ListAddRange,
    "len" => Len,
    "list_get" => ListGet,
    "list_set" => ListSet,
    "dict_get" => DictGet,
    "dict_set" => DictSet,
}

// Exported macro for generating stdlib tokens in lexer
#[macro_export]
macro_rules! stdlib_tokens {
    () => {
        #[token("printf")]
        Printf,
        #[token("puts")]
        Puts,
        #[token("malloc")]
        Malloc,
        #[token("free")]
        Free,
        #[token("abs")]
        Abs,
        #[token("sqrt")]
        Sqrt,
        #[token("pow")]
        Pow,
    };
}

// No lexer coupling: std functions are regular identifiers in the parser.

/// Comprehensive macro to define standard library functions with automatic lexer, parser, and codegen support
macro_rules! define_stdlib_functions {
    ($(
        $fn_name:literal => $rust_name:ident($($param_name:ident: $param_type:ident),*) -> $return_type:ident
    ),* $(,)?) => {
        // Generate the lexer tokens for function names (used for introspection)
        #[allow(dead_code)]
        pub fn get_stdlib_function_tokens() -> Vec<(&'static str, &'static str)> {
            vec![
                $(($fn_name, stringify!($rust_name)),)*
            ]
        }



        // Generate the external functions struct for codegen
        pub struct ExternalFunctions<'ctx> {
            $(pub $rust_name: FunctionValue<'ctx>,)*
        }

        impl<'ctx> ExternalFunctions<'ctx> {
            pub fn declare_all(module: &Module<'ctx>, context: &'ctx Context) -> Self {
                $(
                    let $rust_name = Self::declare_function(
                        module,
                        context,
                        $fn_name,
                        &[$(stringify!($param_type)),*],
                        stringify!($return_type),
                    );
                )*

                ExternalFunctions {
                    $($rust_name,)*
                }
            }

            fn declare_function(
                module: &Module<'ctx>,
                context: &'ctx Context,
                name: &str,
                param_types: &[&str],
                return_type: &str,
            ) -> FunctionValue<'ctx> {
                let mut llvm_param_types = Vec::new();
                
                for param_type in param_types {
                    let llvm_type = match *param_type {
                        "i32" => context.i32_type().into(),
                        "i64" => context.i64_type().into(),
                        "i8_ptr" => context.i8_type().ptr_type(AddressSpace::default()).into(),
                        "f64" => context.f64_type().into(),
                        "bool" => context.bool_type().into(),
                        _ => panic!("Unsupported parameter type: {}", param_type),
                    };
                    llvm_param_types.push(llvm_type);
                }

                let is_variadic = name == "printf"; // Special case for printf
                
                let fn_type = match return_type {
                    "i32" => context.i32_type().fn_type(&llvm_param_types, is_variadic),
                    "i64" => context.i64_type().fn_type(&llvm_param_types, is_variadic),
                    "void" => context.void_type().fn_type(&llvm_param_types, is_variadic),
                    "f64" => context.f64_type().fn_type(&llvm_param_types, is_variadic),
                    "bool" => context.bool_type().fn_type(&llvm_param_types, is_variadic),
                    "i8_ptr" => context.i8_type().ptr_type(AddressSpace::default()).fn_type(&llvm_param_types, is_variadic),
                    _ => panic!("Unsupported return type: {}", return_type),
                };

                module.add_function(name, fn_type, None)
            }

            // Helper method to get function by name for dynamic calling
            pub fn get_function_by_name(&self, name: &str) -> Option<FunctionValue<'ctx>> {
                match name {
                    $(
                        $fn_name => Some(self.$rust_name),
                    )*
                    _ => None,
                }
            }
        }

        // Generate function signature information for parser validation
        #[allow(dead_code)]
        pub fn get_function_signature(name: &str) -> Option<(Vec<&'static str>, &'static str)> {
            match name {
                $(
                    $fn_name => Some((vec![$(stringify!($param_type)),*], stringify!($return_type))),
                )*
                _ => None,
            }
        }

        // Check if a name is a standard library function
        #[allow(dead_code)]
        pub fn is_stdlib_function(name: &str) -> bool {
            matches!(name, $($fn_name)|*)
        }
    };
}

// Define our standard library functions using the comprehensive macro
define_stdlib_functions! {
    "printf" => printf(format: i8_ptr) -> i32,
    "puts" => puts(s: i8_ptr) -> i32,
    "malloc" => malloc(size: i32) -> i8_ptr,
    "free" => free(ptr: i8_ptr) -> void,
    "abs" => abs(x: i32) -> i32,
    "sqrt" => sqrt(x: f64) -> f64,
    "pow" => pow(base: f64, exp: f64) -> f64,
    // file I/O
    "fopen" => fopen(path: i8_ptr, mode: i8_ptr) -> i8_ptr,
    "fclose" => fclose(file: i8_ptr) -> i32,
    "fread" => fread(ptr: i8_ptr, size: i32, nmemb: i32, stream: i8_ptr) -> i32,
    "fwrite" => fwrite(ptr: i8_ptr, size: i32, nmemb: i32, stream: i8_ptr) -> i32,
    // time
    "time" => time(tloc: i8_ptr) -> i64,
    // env
    "getenv" => getenv(name: i8_ptr) -> i8_ptr,
    // random
    "rand" => rand() -> i32,
    "srand" => srand(seed: i32) -> void,
    // networking shims (C ABI)
    "http_request" => http_request(method: i8_ptr, url: i8_ptr, headers_json: i8_ptr, body: i8_ptr) -> i8_ptr,
    "ws_connect" => ws_connect(url: i8_ptr) -> i32,
    "ws_send" => ws_send(handle: i32, msg: i8_ptr) -> i32,
    "ws_recv" => ws_recv(handle: i32) -> i8_ptr,
    "ws_close" => ws_close(handle: i32) -> i32,
    // json utilities
    "json_pretty" => json_pretty(s: i8_ptr) -> i8_ptr,
    "json_to_dict_ss" => json_to_dict_ss(s: i8_ptr) -> i8_ptr,
    // libc helpers
    "strcmp" => strcmp(a: i8_ptr, b: i8_ptr) -> i32,
    // threading
    "spawn" => spawn(func: i8_ptr) -> i32,
    "join" => join(handle: i32) -> i32,
    // channels
    "chan_new" => chan_new() -> i32,
    "chan_send" => chan_send(handle: i32, msg: i8_ptr) -> i32,
    "chan_recv" => chan_recv(handle: i32) -> i8_ptr,
    // thread pool
    "spawn_pool" => spawn_pool(size: i32) -> i32,
    "pool_exec" => pool_exec(handle: i32, func: i8_ptr) -> i32,
    "pool_join" => pool_join(handle: i32) -> i32,
    // filesystem
    "fs_read" => fs_read(path: i8_ptr) -> i8_ptr,
    "fs_write" => fs_write(path: i8_ptr, contents: i8_ptr) -> i32,
    "fs_exists" => fs_exists(path: i8_ptr) -> i32,
    // time formatting
    "time_format" => time_format(fmt: i8_ptr, epoch_secs: i64) -> i8_ptr,
    // regex
    "regex_is_match" => regex_is_match(pattern: i8_ptr, text: i8_ptr) -> i32,
    // crypto
    "sha256_hex" => sha256_hex(s: i8_ptr) -> i8_ptr,
    "hmac_sha256_hex" => hmac_sha256_hex(key: i8_ptr, data: i8_ptr) -> i8_ptr,
    // uuid
    "uuid_v4" => uuid_v4() -> i8_ptr,
    // url
    "url_encode" => url_encode(s: i8_ptr) -> i8_ptr,
    "url_decode" => url_decode(s: i8_ptr) -> i8_ptr,
    // timing
    "sleep" => sleep(ms: i32) -> void,
}

// Map namespaced aliases like nerv::std::http::request -> http_request, etc.
pub fn resolve_std_alias(name: &str) -> Option<&'static str> {
    match name {
        // http
        "nerv::std::http::request" => Some("http_request"),
        // random
        "nerv::std::random::rand" => Some("rand"),
        "nerv::std::random::srand" => Some("srand"),
        // websocket
        "nerv::std::ws::connect" => Some("ws_connect"),
        "nerv::std::ws::send" => Some("ws_send"),
        "nerv::std::ws::recv" => Some("ws_recv"),
        "nerv::std::ws::close" => Some("ws_close"),
        // json
        "nerv::std::json::pretty" => Some("json_pretty"),
        "nerv::std::json::to_dict" => Some("json_to_dict_ss"),
        // threading
        "nerv::std::thread::spawn" => Some("spawn"),
        "nerv::std::thread::join" => Some("join"),
        // channels
        "nerv::std::sync::chan::new" => Some("chan_new"),
        "nerv::std::sync::chan::send" => Some("chan_send"),
        "nerv::std::sync::chan::recv" => Some("chan_recv"),
        // pool
        "nerv::std::threadpool::spawn" => Some("spawn_pool"),
        "nerv::std::threadpool::exec" => Some("pool_exec"),
        "nerv::std::threadpool::join" => Some("pool_join"),
        // time
        "nerv::std::time::time" => Some("time"),
        "nerv::std::time::sleep" => Some("sleep"),
        // env
        "nerv::std::env::get" => Some("getenv"),
        // libc-ish basics
        "nerv::std::io::puts" => Some("puts"),
        // fs
        "nerv::std::fs::read" => Some("fs_read"),
        "nerv::std::fs::write" => Some("fs_write"),
        "nerv::std::fs::exists" => Some("fs_exists"),
        // time fmt
        "nerv::std::time::format" => Some("time_format"),
        // regex
        "nerv::std::regex::is_match" => Some("regex_is_match"),
        // crypto
        "nerv::std::crypto::sha256_hex" => Some("sha256_hex"),
        "nerv::std::crypto::hmac_sha256_hex" => Some("hmac_sha256_hex"),
        // uuid
        "nerv::std::uuid::v4" => Some("uuid_v4"),
        // url
        "nerv::std::url::encode" => Some("url_encode"),
        "nerv::std::url::decode" => Some("url_decode"),
        _ => None,
    }
}
