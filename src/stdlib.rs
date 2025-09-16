use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;
use inkwell::AddressSpace;

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

// Exported macro for generating stdlib token helper functions
#[macro_export]
macro_rules! stdlib_token_helpers {
    () => {
        impl Token {
            /// Check if this token represents a standard library function
            pub fn is_stdlib_function(&self) -> bool {
                matches!(self, 
                    Token::Printf | Token::Puts | Token::Malloc | Token::Free |
                    Token::Abs | Token::Sqrt | Token::Pow
                )
            }

            /// Get the function name as a string for stdlib functions
            pub fn as_function_name(&self) -> Option<&'static str> {
                match self {
                    Token::Printf => Some("printf"),
                    Token::Puts => Some("puts"),
                    Token::Malloc => Some("malloc"),
                    Token::Free => Some("free"),
                    Token::Abs => Some("abs"),
                    Token::Sqrt => Some("sqrt"),
                    Token::Pow => Some("pow"),
                    _ => None,
                }
            }
        }
    };
}

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
}
