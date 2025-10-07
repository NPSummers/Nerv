//! Native runtime shims and helpers exposed to the JIT.
//!
//! Contains C-ABI functions for HTTP, WebSocket, and JSON utilities, and
//! helpers to pass data across the FFI boundary.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::Duration;
use std::fs;
// use std::io::Read as _; // not currently needed

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


// ---- Threads ----

static THREAD_REG: Lazy<Mutex<HashMap<i32, thread::JoinHandle<i32>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_THREAD_ID: Lazy<Mutex<i32>> = Lazy::new(|| Mutex::new(1));

#[no_mangle]
pub extern "C" fn nerv_spawn(func: *const i8) -> i32 {
    if func.is_null() { return 0; }
    // Safety: We expect a pointer to a JIT-emitted function with C ABI: fn() -> i32
    let f: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func) };
    let handle = thread::spawn(move || {
        // Call the JIT function on this OS thread
        f()
    });
    let mut id_guard = NEXT_THREAD_ID.lock().unwrap();
    let id = *id_guard;
    *id_guard += 1;
    THREAD_REG.lock().unwrap().insert(id, handle);
    id
}

// ---- Sleep ----

#[no_mangle]
pub extern "C" fn nerv_sleep(ms: i32) {
    if ms <= 0 { return; }
    let dur = Duration::from_millis(ms as u64);
    thread::sleep(dur);
}

// ---- Filesystem / Process / Utilities ----

#[no_mangle]
pub extern "C" fn nerv_fs_read(path: *const i8) -> *mut i8 {
    if path.is_null() { return std::ptr::null_mut(); }
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_string_lossy().to_string();
    match fs::read_to_string(&p) {
        Ok(s) => match std::ffi::CString::new(s) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ffi::CString::new("").unwrap().into_raw(),
        },
        Err(_) => std::ffi::CString::new("").unwrap().into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn nerv_fs_write(path: *const i8, contents: *const i8) -> i32 {
    if path.is_null() || contents.is_null() { return -1; }
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_string_lossy().to_string();
    let c = unsafe { std::ffi::CStr::from_ptr(contents) }.to_string_lossy().to_string();
    match fs::write(&p, c) { Ok(_) => 0, Err(_) => -1 }
}

#[no_mangle]
pub extern "C" fn nerv_fs_exists(path: *const i8) -> i32 {
    if path.is_null() { return 0; }
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_string_lossy().to_string();
    if std::path::Path::new(&p).exists() { 1 } else { 0 }
}

// ---- Time/Date formatting ----

#[no_mangle]
pub extern "C" fn nerv_time_format(fmt: *const i8, epoch_secs: i64) -> *mut i8 {
    if fmt.is_null() { return std::ptr::null_mut(); }
    let fmt_s = unsafe { std::ffi::CStr::from_ptr(fmt) }.to_string_lossy().to_string();
    let dt = chrono::NaiveDateTime::from_timestamp_opt(epoch_secs, 0);
    match dt {
        Some(t) => to_c_string_owned(t.format(&fmt_s).to_string()),
        None => std::ptr::null_mut(),
    }
}

// ---- Regex ----

#[no_mangle]
pub extern "C" fn nerv_regex_is_match(pattern: *const i8, text: *const i8) -> i32 {
    if pattern.is_null() || text.is_null() { return 0; }
    let pat = unsafe { std::ffi::CStr::from_ptr(pattern) }.to_string_lossy().to_string();
    let txt = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy().to_string();
    match regex::Regex::new(&pat) {
        Ok(re) => if re.is_match(&txt) { 1 } else { 0 },
        Err(_) => 0,
    }
}

// ---- Crypto (hash/HMAC) ----

#[no_mangle]
pub extern "C" fn nerv_sha256_hex(s: *const i8) -> *mut i8 {
    if s.is_null() { return std::ptr::null_mut(); }
    let raw = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().as_bytes().to_vec();
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&raw);
    let out = hasher.finalize();
    to_c_string_owned(hex::encode(out))
}

#[no_mangle]
pub extern "C" fn nerv_hmac_sha256_hex(key: *const i8, data: *const i8) -> *mut i8 {
    if key.is_null() || data.is_null() { return std::ptr::null_mut(); }
    let key_b = unsafe { std::ffi::CStr::from_ptr(key) }.to_bytes().to_vec();
    let data_b = unsafe { std::ffi::CStr::from_ptr(data) }.to_bytes().to_vec();
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac = match Hmac::<Sha256>::new_from_slice(&key_b) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };
    mac.update(&data_b);
    let out = mac.finalize().into_bytes();
    to_c_string_owned(hex::encode(out))
}

// ---- UUID ----

#[no_mangle]
pub extern "C" fn nerv_uuid_v4() -> *mut i8 {
    to_c_string_owned(uuid::Uuid::new_v4().to_string())
}

// ---- URL helpers ----

#[no_mangle]
pub extern "C" fn nerv_url_encode(s: *const i8) -> *mut i8 {
    if s.is_null() { return std::ptr::null_mut(); }
    let raw = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().to_string();
    to_c_string_owned(urlencoding::encode(&raw).into_owned())
}

#[no_mangle]
pub extern "C" fn nerv_url_decode(s: *const i8) -> *mut i8 {
    if s.is_null() { return std::ptr::null_mut(); }
    let raw = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().to_string();
    match urlencoding::decode(&raw) {
        Ok(cow) => to_c_string_owned(cow.into_owned()),
        Err(_) => std::ptr::null_mut(),
    }
}
#[no_mangle]
pub extern "C" fn nerv_join(handle: i32) -> i32 {
    let mut reg = THREAD_REG.lock().unwrap();
    match reg.remove(&handle) {
        Some(h) => match h.join() {
            Ok(code) => code,
            Err(_) => -1,
        },
        None => -1,
    }
}

// ---- Channels (MPSC, string messages) ----

struct ChanPair {
    tx: mpsc::Sender<String>,
    rx: mpsc::Receiver<String>,
}

static CHAN_REG: Lazy<Mutex<HashMap<i32, ChanPair>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_CHAN_ID: Lazy<Mutex<i32>> = Lazy::new(|| Mutex::new(1));

#[no_mangle]
pub extern "C" fn nerv_chan_new() -> i32 {
    let (tx, rx) = mpsc::channel::<String>();
    let mut id_guard = NEXT_CHAN_ID.lock().unwrap();
    let id = *id_guard;
    *id_guard += 1;
    CHAN_REG.lock().unwrap().insert(id, ChanPair { tx, rx });
    id
}

#[no_mangle]
pub extern "C" fn nerv_chan_send(handle: i32, msg: *const i8) -> i32 {
    let msg_s = if msg.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy().to_string() };
    let reg = CHAN_REG.lock().unwrap();
    let Some(pair) = reg.get(&handle) else { return -1; };
    match pair.tx.send(msg_s) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn nerv_chan_recv(handle: i32) -> *mut i8 {
    let reg = CHAN_REG.lock().unwrap();
    let Some(pair) = reg.get(&handle) else { return std::ptr::null_mut(); };
    match pair.rx.recv() {
        Ok(s) => to_c_string_owned(s),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---- Thread pool ----

enum PoolMsg {
    Job(extern "C" fn() -> i32),
    Shutdown,
}

struct ThreadPool {
    tx: mpsc::Sender<PoolMsg>,
    workers: Vec<thread::JoinHandle<()>>,
}

static POOL_REG: Lazy<Mutex<HashMap<i32, ThreadPool>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_POOL_ID: Lazy<Mutex<i32>> = Lazy::new(|| Mutex::new(1));

#[no_mangle]
pub extern "C" fn nerv_spawn_pool(size: i32) -> i32 {
    let n = if size <= 0 { 1 } else { size as usize };
    let (tx, rx) = mpsc::channel::<PoolMsg>();
    let rx = std::sync::Arc::new(Mutex::new(rx));
    let mut workers = Vec::with_capacity(n);
    for _ in 0..n {
        let rx_cloned = rx.clone();
        let h = thread::spawn(move || loop {
            let msg_opt = { rx_cloned.lock().unwrap().recv().ok() };
            match msg_opt {
                Some(PoolMsg::Job(f)) => { let _ = f(); }
                Some(PoolMsg::Shutdown) | None => break,
            }
        });
        workers.push(h);
    }
    let mut id_guard = NEXT_POOL_ID.lock().unwrap();
    let id = *id_guard; *id_guard += 1;
    POOL_REG.lock().unwrap().insert(id, ThreadPool { tx, workers });
    id
}

#[no_mangle]
pub extern "C" fn nerv_pool_exec(handle: i32, func: *const i8) -> i32 {
    if func.is_null() { return -1; }
    let f: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func) };
    let reg = POOL_REG.lock().unwrap();
    let Some(pool) = reg.get(&handle) else { return -1; };
    match pool.tx.send(PoolMsg::Job(f)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn nerv_pool_join(handle: i32) -> i32 {
    let mut reg = POOL_REG.lock().unwrap();
    let Some(mut pool) = reg.remove(&handle) else { return -1; };
    // signal shutdown to each worker
    for _ in 0..pool.workers.len() { let _ = pool.tx.send(PoolMsg::Shutdown); }
    let mut ok = true;
    for h in pool.workers.drain(..) { if h.join().is_err() { ok = false; } }
    if ok { 0 } else { -1 }
}

