//! `cv::AsyncArray` and `cv::AsyncPromise` — a matrix that is not ready yet
//! (`modules/core/include/opencv2/core/async.hpp`, `modules/core/src/async.cpp`).
//!
//! One side holds an [`AsyncPromise`] and eventually calls
//! [`AsyncPromise::set_value`] or [`AsyncPromise::set_exception`]; the other
//! holds an [`AsyncArray`] and blocks in [`AsyncArray::get`] until it does.
//! Upstream uses this for DNN inference queued on an accelerator.
//!
//! Nothing in this port produces one — every kernel here is synchronous — but
//! the type is a complete, working future, so code translated from C++ that
//! threads an `AsyncArray` through its own pipeline keeps working.
//!
//! A dropped promise is not a hang: it resolves the array with `StsError`, so
//! a waiting [`AsyncArray::get`] returns rather than blocking forever. C++
//! leaves that case deadlocked.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::error::{Code, Error, Result};
use crate::mat::Mat;

/// The shared slot both halves point at.
#[derive(Debug)]
struct Shared {
    /// `None` until the promise resolves.
    slot: Mutex<Option<Result<Mat>>>,
    ready: Condvar,
}

impl Shared {
    fn new() -> Arc<Shared> {
        Arc::new(Shared { slot: Mutex::new(None), ready: Condvar::new() })
    }

    /// Store a result and wake every waiter. A second call is ignored, so a
    /// promise cannot overwrite a value someone may already have read.
    fn resolve(&self, value: Result<Mat>) {
        let mut g = self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.is_none() {
            *g = Some(value);
            self.ready.notify_all();
        }
    }
}

/// `cv::AsyncArray` — the reading half.
#[derive(Debug, Clone)]
pub struct AsyncArray {
    shared: Option<Arc<Shared>>,
}

impl Default for AsyncArray {
    /// `AsyncArray()` — an invalid array, as upstream's default constructor gives.
    fn default() -> Self {
        AsyncArray { shared: None }
    }
}

impl AsyncArray {
    /// `cv::AsyncArray::AsyncArray()` — an invalid, unattached array.
    #[must_use]
    pub fn new() -> Self {
        AsyncArray::default()
    }

    /// `cv::AsyncArray::valid()` — is this attached to a promise?
    #[must_use]
    pub fn valid(&self) -> bool {
        self.shared.is_some()
    }

    /// `cv::AsyncArray::release()` — detach, making this invalid.
    pub fn release(&mut self) {
        self.shared = None;
    }

    /// `cv::AsyncArray::get(dst)` — block until the value arrives.
    ///
    /// Returns the promise's error if it set one, or `StsError` if the promise
    /// was dropped without resolving.
    pub fn get(&self) -> Result<Mat> {
        let Some(shared) = &self.shared else {
            return Err(Error::new(Code::StsError, "AsyncArray is not valid"));
        };
        let mut g = shared.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while g.is_none() {
            g = shared.ready.wait(g).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        g.clone().expect("just waited for it to be set")
    }

    /// `cv::AsyncArray::get(dst, timeoutNs)` — block up to `timeout`.
    ///
    /// `Ok(None)` means the wait expired with no value, which is upstream's
    /// `false` return; that is distinct from the promise having failed.
    pub fn get_timeout(&self, timeout: Duration) -> Result<Option<Mat>> {
        let Some(shared) = &self.shared else {
            return Err(Error::new(Code::StsError, "AsyncArray is not valid"));
        };
        let g = shared.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (g, wait) = shared
            .ready
            .wait_timeout_while(g, timeout, |s| s.is_none())
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if wait.timed_out() {
            return Ok(None);
        }
        g.clone().expect("not timed out, so a value is present").map(Some)
    }

    /// `cv::AsyncArray::wait_for(timeoutNs)` — is the value ready within `timeout`?
    pub fn wait_for(&self, timeout: Duration) -> bool {
        let Some(shared) = &self.shared else { return false };
        let g = shared.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (_g, wait) = shared
            .ready
            .wait_timeout_while(g, timeout, |s| s.is_none())
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        !wait.timed_out()
    }
}

/// `cv::AsyncPromise` — the writing half.
#[derive(Debug)]
pub struct AsyncPromise {
    shared: Arc<Shared>,
    resolved: bool,
}

impl Default for AsyncPromise {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncPromise {
    /// `cv::AsyncPromise::AsyncPromise()`.
    #[must_use]
    pub fn new() -> Self {
        AsyncPromise { shared: Shared::new(), resolved: false }
    }

    /// `cv::AsyncPromise::getArrayResult()` — the reading half.
    ///
    /// May be called more than once; every copy sees the same value, which is
    /// what upstream's reference-counted handle does.
    #[must_use]
    pub fn get_array_result(&self) -> AsyncArray {
        AsyncArray { shared: Some(Arc::clone(&self.shared)) }
    }

    /// `cv::AsyncPromise::setValue(value)`.
    pub fn set_value(&mut self, value: Mat) {
        self.resolved = true;
        self.shared.resolve(Ok(value));
    }

    /// `cv::AsyncPromise::setException(e)`.
    pub fn set_exception(&mut self, e: Error) {
        self.resolved = true;
        self.shared.resolve(Err(e));
    }
}

impl Drop for AsyncPromise {
    /// Resolve with an error rather than leaving readers blocked forever.
    fn drop(&mut self) {
        if !self.resolved {
            self.shared.resolve(Err(Error::new(
                Code::StsError,
                "AsyncPromise was dropped without a value",
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CV_8UC1;

    #[test]
    fn a_default_array_is_invalid() {
        let mut a = AsyncArray::new();
        assert!(!a.valid());
        assert!(a.get().is_err());
        assert!(!a.wait_for(Duration::from_millis(1)));
        a.release();
        assert!(!a.valid());
    }

    #[test]
    fn a_value_set_before_the_read_is_returned_immediately() {
        let mut p = AsyncPromise::new();
        let a = p.get_array_result();
        assert!(a.valid());
        p.set_value(Mat::from_vec(1, 3, 1, &[1u8, 2, 3]).unwrap());
        assert!(a.wait_for(Duration::from_millis(0)));
        assert_eq!(a.get().unwrap().to_vec::<u8>().unwrap(), vec![1, 2, 3]);
        // Reading twice gives the same value.
        assert_eq!(a.get().unwrap().to_vec::<u8>().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn get_blocks_until_another_thread_resolves() {
        let mut p = AsyncPromise::new();
        let a = p.get_array_result();
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            p.set_value(Mat::from_vec(1, 1, 1, &[42u8]).unwrap());
        });
        // Blocks here until the thread above sets the value.
        assert_eq!(a.get().unwrap().to_vec::<u8>().unwrap(), vec![42]);
        h.join().unwrap();
    }

    #[test]
    fn a_timeout_expires_without_a_value() {
        let mut p = AsyncPromise::new();
        let a = p.get_array_result();
        // `Mat` is deliberately not `PartialEq`, so check the shape of the
        // result rather than comparing it.
        assert!(a.get_timeout(Duration::from_millis(5)).unwrap().is_none());
        assert!(!a.wait_for(Duration::from_millis(5)));
        p.set_value(Mat::zeros(1, 1, CV_8UC1).unwrap());
        assert!(a.get_timeout(Duration::from_millis(5)).unwrap().is_some());
    }

    #[test]
    fn an_exception_reaches_the_reader() {
        let mut p = AsyncPromise::new();
        let a = p.get_array_result();
        p.set_exception(Error::new(Code::StsBadArg, "inference failed"));
        let e = a.get().unwrap_err();
        assert_eq!(e.code, Code::StsBadArg);
        assert_eq!(e.msg, "inference failed");
    }

    #[test]
    fn dropping_the_promise_unblocks_the_reader() {
        let a = {
            let p = AsyncPromise::new();
            p.get_array_result()
        };
        // Would deadlock upstream; here the drop resolves it with an error.
        let e = a.get().unwrap_err();
        assert_eq!(e.code, Code::StsError);
        assert!(e.msg.contains("dropped"), "{}", e.msg);
    }

    #[test]
    fn the_first_resolution_wins() {
        let mut p = AsyncPromise::new();
        let a = p.get_array_result();
        p.set_value(Mat::from_vec(1, 1, 1, &[7u8]).unwrap());
        p.set_exception(Error::new(Code::StsBadArg, "too late"));
        assert_eq!(a.get().unwrap().to_vec::<u8>().unwrap(), vec![7]);
    }
}
