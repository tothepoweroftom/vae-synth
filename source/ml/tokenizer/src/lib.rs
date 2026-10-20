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
