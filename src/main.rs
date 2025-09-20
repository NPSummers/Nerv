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

// Simple heap for returning C strings to JIT
fn to_c_string_owned(s: String) -> *mut i8 {
    let c = std::ffi::CString::new(s).unwrap();
    c.into_raw()
}

unsafe fn write_ptr(dst: *mut u8, offset: usize, p: *mut i8) {
    let slot = dst.add(offset) as *mut *mut i8;
    std::ptr::write_unaligned(slot, p);
}

unsafe fn write_i32(dst: *mut u8, offset: usize, v: i32) {
    let slot = dst.add(offset) as *mut i32;
    std::ptr::write_unaligned(slot, v);
}

fn c_string_ptr(s: &str) -> *mut i8 {
    std::ffi::CString::new(s).unwrap().into_raw()
}

fn alloc_dict_ss(pairs: &[(String, String)]) -> *mut i8 {
    let len = pairs.len() as i32;
    let ptr_sz = std::mem::size_of::<*mut i8>();
    let total = 8 + (pairs.len() * ptr_sz * 2);
    unsafe {
        let base = libc::malloc(total) as *mut u8;
        if base.is_null() { return std::ptr::null_mut(); }
        write_i32(base, 0, len);
        // padding at 4 left as-is
        let pairs_base = 8usize;
        for (i, (k, v)) in pairs.iter().enumerate() {
            let k_ptr = c_string_ptr(k);
            let v_ptr = c_string_ptr(v);
            let off = pairs_base + i * (ptr_sz * 2);
            write_ptr(base, off, k_ptr);
            write_ptr(base, off + ptr_sz, v_ptr);
        }
        base as *mut i8
    }
}

fn json_to_pairs_ss(raw: &str) -> Vec<(String, String)> {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(map)) => map
            .into_iter()
            .map(|(k, v)| (k, match v {
                serde_json::Value::String(s) => s,
                _ => serde_json::to_string(&v).unwrap_or_else(|_| "".to_string()),
            }))
            .collect(),
        Ok(v) => vec![("value".to_string(), serde_json::to_string(&v).unwrap_or_else(|_| raw.to_string()))],
        Err(_) => vec![("text".to_string(), raw.to_string())],
    }
}

#[no_mangle]
pub extern "C" fn nerv_http_get(url: *const i8) -> *mut i8 {
    if url.is_null() { return std::ptr::null_mut(); }
    let cstr = unsafe { std::ffi::CStr::from_ptr(url) };
    let url_s = cstr.to_string_lossy();
    match ureq::get(&url_s).call() {
        Ok(resp) => {
            if let Ok(text) = resp.into_string() {
                let pairs = json_to_pairs_ss(&text);
                alloc_dict_ss(&pairs)
            } else { std::ptr::null_mut() }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn nerv_http_post(url: *const i8, body: *const i8) -> *mut i8 {
    if url.is_null() { return std::ptr::null_mut(); }
    let url_s = unsafe { std::ffi::CStr::from_ptr(url) }.to_string_lossy().to_string();
    let body_s = if body.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(body) }.to_string_lossy().to_string() };
    match ureq::post(&url_s).send_string(&body_s) {
        Ok(resp) => {
            if let Ok(text) = resp.into_string() { alloc_dict_ss(&json_to_pairs_ss(&text)) } else { std::ptr::null_mut() }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// Very simple websocket handle store
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;
static WS_REG: Lazy<Mutex<HashMap<i32, tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_WS_ID: Lazy<Mutex<i32>> = Lazy::new(|| Mutex::new(1));

#[no_mangle]
pub extern "C" fn nerv_ws_connect(url: *const i8) -> i32 {
    if url.is_null() { return 0; }
    let url_s = unsafe { std::ffi::CStr::from_ptr(url) }.to_string_lossy().to_string();
    match tungstenite::connect(url_s) {
        Ok((socket, _response)) => {
            let mut id_guard = NEXT_WS_ID.lock().unwrap();
            let id = *id_guard;
            *id_guard += 1;
            WS_REG.lock().unwrap().insert(id, socket);
            id
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn nerv_ws_send(handle: i32, msg: *const i8) -> i32 {
    let mut reg = WS_REG.lock().unwrap();
    let Some(sock) = reg.get_mut(&handle) else { return -1; };
    let msg_s = if msg.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy().to_string() };
    match sock.write_message(tungstenite::protocol::Message::Text(msg_s)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn nerv_ws_recv(handle: i32) -> *mut i8 {
    let mut reg = WS_REG.lock().unwrap();
    let Some(sock) = reg.get_mut(&handle) else { return std::ptr::null_mut(); };
    match sock.read_message() {
        Ok(msg) => match msg {
            tungstenite::protocol::Message::Text(s) => to_c_string_owned(s),
            tungstenite::protocol::Message::Binary(b) => to_c_string_owned(base64::encode(b)),
            _ => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn nerv_json_pretty(s: *const i8) -> *mut i8 {
    if s.is_null() { return std::ptr::null_mut(); }
    let raw = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().to_string();
    let val: Result<serde_json::Value, _> = serde_json::from_str(&raw);
    match val {
        Ok(v) => to_c_string_owned(serde_json::to_string_pretty(&v).unwrap_or(raw)),
        Err(_) => to_c_string_owned(raw),
    }
}

#[no_mangle]
pub extern "C" fn nerv_json_to_dict_ss(s: *const i8) -> *mut i8 {
    if s.is_null() { return std::ptr::null_mut(); }
    let raw = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().to_string();
    let pairs = json_to_pairs_ss(&raw);
    alloc_dict_ss(&pairs)
}

#[no_mangle]
pub extern "C" fn nerv_ws_close(handle: i32) -> i32 {
    let mut reg = WS_REG.lock().unwrap();
    if let Some(mut sock) = reg.remove(&handle) {
        let _ = sock.close(None);
        0
    } else { -1 }
}

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
