//! Small C ABI around Hugging Face's native tokenizer. No Python or network access.
use std::{ffi::{c_char, CStr}, ptr};
use tokenizers::{Tokenizer, PaddingParams, PaddingStrategy, TruncationParams};

#[no_mangle]
pub unsafe extern "C" fn vae_tokenizer_load(path: *const c_char) -> *mut Tokenizer {
    if path.is_null() { return ptr::null_mut(); }
    let Ok(path) = CStr::from_ptr(path).to_str() else { return ptr::null_mut(); };
    let Ok(mut t) = Tokenizer::from_file(path) else { return ptr::null_mut(); };
    if t.with_truncation(Some(TruncationParams { max_length: 256, ..Default::default() })).is_err() {
        return ptr::null_mut();
    }
    t.with_padding(Some(PaddingParams { strategy: PaddingStrategy::Fixed(256), pad_id: 0,
        pad_token: "<pad>".into(), ..Default::default() }));
    Box::into_raw(Box::new(t))
}
#[no_mangle]
pub unsafe extern "C" fn vae_tokenizer_free(t: *mut Tokenizer) {
    if !t.is_null() { drop(Box::from_raw(t)); }
}
#[no_mangle]
pub unsafe extern "C" fn vae_tokenizer_encode(t: *const Tokenizer, text: *const c_char,
    ids: *mut f32, mask: *mut f32) -> bool {
    if t.is_null() || text.is_null() || ids.is_null() || mask.is_null() { return false; }
    let Ok(text) = CStr::from_ptr(text).to_str() else { return false; };
    let Ok(e) = (*t).encode(text, true) else { return false; };
    if e.len() != 256 { return false; }
    for i in 0..256 { *ids.add(i) = e.get_ids()[i] as f32; *mask.add(i) = e.get_attention_mask()[i] as f32; }
    true
}

/// Configurable path (Lab prompt model); the 256 / pad 0 / float functions above stay as they are.
/// Truncates longest-first to `max_length`. `pad_to` 0 means no padding ("longest" for one prompt),
/// otherwise pads to exactly `pad_to` with `pad_id`.
#[no_mangle]
pub unsafe extern "C" fn vae_tokenizer_load_config(path: *const c_char, max_length: usize, pad_to: usize,
    pad_id: u32) -> *mut Tokenizer {
    if path.is_null() || max_length == 0 || pad_to > max_length { return ptr::null_mut(); }
    let Ok(path) = CStr::from_ptr(path).to_str() else { return ptr::null_mut(); };
    let Ok(mut t) = Tokenizer::from_file(path) else { return ptr::null_mut(); };
    if t.with_truncation(Some(TruncationParams { max_length, ..Default::default() })).is_err() {
        return ptr::null_mut();
    }
    if pad_to == 0 {
        t.with_padding(None);
    } else {
        let Some(pad_token) = t.id_to_token(pad_id) else { return ptr::null_mut(); };
        t.with_padding(Some(PaddingParams { strategy: PaddingStrategy::Fixed(pad_to), pad_id, pad_token,
            ..Default::default() }));
    }
    Box::into_raw(Box::new(t))
}
/// Writes up to `capacity` int64 ids and mask values; returns the token count, or -1 on error or overflow.
#[no_mangle]
pub unsafe extern "C" fn vae_tokenizer_encode_i64(t: *const Tokenizer, text: *const c_char,
    ids: *mut i64, mask: *mut i64, capacity: usize) -> isize {
    if t.is_null() || text.is_null() || ids.is_null() || mask.is_null() { return -1; }
    let Ok(text) = CStr::from_ptr(text).to_str() else { return -1; };
    let Ok(e) = (*t).encode(text, true) else { return -1; };
    if e.len() > capacity { return -1; }
    for i in 0..e.len() { *ids.add(i) = e.get_ids()[i] as i64; *mask.add(i) = e.get_attention_mask()[i] as i64; }
    e.len() as isize
}
