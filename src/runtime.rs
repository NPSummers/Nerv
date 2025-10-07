//! Native runtime shims and helpers exposed to the JIT.
//!
//! Contains C-ABI functions for HTTP, WebSocket, and JSON utilities, and
//! helpers to pass data across the FFI boundary.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

// ---- FFI helpers ----

pub fn to_c_string_owned(s: String) -> *mut i8 {
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

/// Allocate a simple dict of string->string for return to JIT.
pub fn alloc_dict_ss(pairs: &[(String, String)]) -> *mut i8 {
    let len = pairs.len() as i32;
    let ptr_sz = std::mem::size_of::<*mut i8>();
    let total = 8 + (pairs.len() * ptr_sz * 2);
    unsafe {
        let base = libc::malloc(total) as *mut u8;
        if base.is_null() { return std::ptr::null_mut(); }
        write_i32(base, 0, len);
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

// ---- HTTP ----

#[no_mangle]
pub extern "C" fn nerv_http_request(method: *const i8, url: *const i8, headers_json: *const i8, body: *const i8) -> *mut i8 {
    if method.is_null() || url.is_null() { return std::ptr::null_mut(); }
    let method_s = unsafe { std::ffi::CStr::from_ptr(method) }.to_string_lossy().to_uppercase();
    let url_s = unsafe { std::ffi::CStr::from_ptr(url) }.to_string_lossy().to_string();
    let headers_s = if headers_json.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(headers_json) }.to_string_lossy().to_string() };
    let body_s = if body.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(body) }.to_string_lossy().to_string() };

    // Parse headers JSON as a dict of string->string (best-effort)
    let mut req = match method_s.as_str() {
        "GET" => ureq::get(&url_s),
        "POST" => ureq::post(&url_s),
        "PUT" => ureq::put(&url_s),
        "PATCH" => ureq::patch(&url_s),
        "DELETE" => ureq::delete(&url_s),
        _ => ureq::request(&method_s, &url_s),
    };

    if !headers_s.is_empty() {
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(&headers_s) {
            for (k, v) in map.into_iter() {
                let val = match v { serde_json::Value::String(s) => s, other => other.to_string() };
                req = req.set(&k, &val);
            }
        }
    }

    let resp = if matches!(method_s.as_str(), "GET") { req.call() } else { req.send_string(&body_s) };

    match resp {
        Ok(r) => {
            let text = match r.into_string() {
                Ok(s) => s,
                Err(_) => return std::ptr::null_mut(),
            };
            alloc_dict_ss(&json_to_pairs_ss(&text))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// ---- WebSocket ----

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
            tungstenite::protocol::Message::Binary(b) => {
                use base64::prelude::{Engine as _, BASE64_STANDARD};
                to_c_string_owned(BASE64_STANDARD.encode(b))
            }
            _ => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn nerv_ws_close(handle: i32) -> i32 {
    let mut reg = WS_REG.lock().unwrap();
    if let Some(mut sock) = reg.remove(&handle) {
        let _ = sock.close(None);
        0
    } else { -1 }
}

// ---- JSON ----

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


