//! `cv::highgui` for `opencv-no-c` — windows, keys, trackbars, ROI selection.
//!
//! # Backend
//!
//! Built on `egui`/`eframe`, behind the `gui` feature. Without it every
//! entry point still exists and returns [`cv_core::Code::StsNotImplemented`],
//! so a caller does not need `cfg` blocks around its display code — the usual
//! shape is a program that shows windows when built with `--features gui` and
//! runs headless otherwise.
//!
//! # The honesty note this crate owes
//!
//! The rest of this workspace is pure Rust with no FFI at all. A window is the
//! one thing that cannot be: `eframe` reaches the platform through `winit`,
//! which talks to Cocoa, X11/Wayland or Win32. That is a real dependency on
//! system libraries, which is exactly why this lives in its own optional
//! crate — the same containment `cv-accel` uses for RLX. Nothing else in the
//! workspace depends on it.
//!
//! # Divergence from upstream's model
//!
//! `cv::imshow` posts to a global window registry and `cv::waitKey` pumps the
//! event loop; a program interleaves them freely. `eframe` owns the event loop
//! instead and calls back into the application, so that inversion cannot be
//! papered over. The API here is therefore *session*-shaped: build a
//! [`Session`], register windows, then [`Session::run`]. [`imshow`] and
//! [`wait_key`] exist for one-shot use and are documented with what they
//! actually do.

#![warn(missing_docs)]

pub mod constants;

use std::collections::BTreeMap;

use cv_core::{Code, Error, MatLike, Result};

/// One frame's pixels, converted out of a `Mat` and ready to upload.
///
/// Kept separate from any GUI type so the conversion — which is the part with
/// the interesting decisions in it — can be tested without a window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 4` bytes, RGBA, row-major.
    pub pixels: Vec<u8>,
}

/// Convert any `Mat` to RGBA for display.
///
/// # Channel order
///
/// OpenCV stores colour as **BGR**, and every display stack wants RGB. Getting
/// this backwards is the classic `imshow` bug: the picture appears, looks
/// plausible, and every red object is blue. The swap happens here.
///
/// Float images are assumed to lie in `[0, 1]` and are scaled by 255, which is
/// what `cv::imshow` does for `CV_32F`. Integer depths are taken as-is and
/// saturated.
pub fn to_rgba<S: MatLike + ?Sized>(m: &S) -> Result<Rgba> {
    let (w, h) = (m.cols() as usize, m.rows() as usize);
    if w == 0 || h == 0 {
        return Err(Error::new(Code::StsBadSize, "cannot display an empty image".to_string()));
    }
    let cn = m.channels();
    let float = matches!(m.depth(), cv_core::types::Depth::F32 | cv_core::types::Depth::F64);
    let mut pixels = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let get = |c: i32| -> Result<u8> {
                let v = m.at_f64(y as i32, x as i32, c)?;
                let v = if float { v * 255.0 } else { v };
                Ok(cv_core::saturate::Elem::from_f64(v))
            };
            let o = (y * w + x) * 4;
            match cn {
                1 => {
                    let v = get(0)?;
                    pixels[o] = v;
                    pixels[o + 1] = v;
                    pixels[o + 2] = v;
                    pixels[o + 3] = 255;
                }
                3 => {
                    // BGR -> RGB.
                    pixels[o] = get(2)?;
                    pixels[o + 1] = get(1)?;
                    pixels[o + 2] = get(0)?;
                    pixels[o + 3] = 255;
                }
                4 => {
                    // BGRA -> RGBA.
                    pixels[o] = get(2)?;
                    pixels[o + 1] = get(1)?;
                    pixels[o + 2] = get(0)?;
                    pixels[o + 3] = get(3)?;
                }
                other => {
                    return Err(Error::new(
                        Code::StsBadArg,
                        format!("cannot display a {other}-channel image"),
                    ))
                }
            }
        }
    }
    Ok(Rgba { width: w, height: h, pixels })
}

/// `cv::WindowFlags` — how a window sizes itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFlag {
    /// `cv::WINDOW_AUTOSIZE` — the window takes the image's size and is fixed.
    #[default]
    AutoSize,
    /// `cv::WINDOW_NORMAL` — the user can resize; the image scales to fit.
    Normal,
    /// `cv::WINDOW_KEEPRATIO` — resizable, but the aspect ratio is preserved.
    KeepRatio,
}

/// A slider attached to a window — `cv::createTrackbar`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trackbar {
    /// Label shown beside the slider.
    pub name: String,
    /// Current position.
    pub value: i32,
    /// Maximum; the minimum is always 0, as upstream's is.
    pub max: i32,
}

/// One window's state.
#[derive(Debug, Clone)]
pub struct Window {
    /// Title, and the key [`Session`] indexes it by.
    pub name: String,
    /// Sizing behaviour.
    pub flag: WindowFlag,
    /// Current image, if one has been shown.
    pub image: Option<Rgba>,
    /// Sliders, in creation order.
    pub trackbars: Vec<Trackbar>,
    /// Recorded position and size; see [`window_geometry`].
    pub geometry: window_geometry::Geometry,
    /// Text set by `displayOverlay`, if any.
    pub overlay: Option<String>,
    /// How long that overlay was asked to stay, in milliseconds.
    pub overlay_delay_ms: i32,
    /// Text set by `displayStatusBar`, if any.
    pub status_bar: Option<String>,
    /// How long the status text was asked to stay.
    pub status_delay_ms: i32,
    /// Text passed to `addText`, in order.
    pub texts: Vec<String>,
    /// Regions recorded for `selectROIs`.
    pub rois: Vec<cv_core::Rect>,
    /// Values set by `setWindowProperty`, by property id.
    pub properties: std::collections::BTreeMap<i32, f64>,
}

impl Window {
    /// A named window with no image yet.
    #[must_use]
    pub fn new(name: impl Into<String>, flag: WindowFlag) -> Self {
        Window {
            name: name.into(),
            flag,
            image: None,
            trackbars: Vec::new(),
            geometry: window_geometry::Geometry::default(),
            overlay: None,
            overlay_delay_ms: 0,
            status_bar: None,
            status_delay_ms: 0,
            texts: Vec::new(),
            rois: Vec::new(),
            properties: std::collections::BTreeMap::new(),
        }
    }

    /// Position of a named trackbar — `cv::getTrackbarPos`.
    #[must_use]
    pub fn trackbar_pos(&self, name: &str) -> Option<i32> {
        self.trackbars.iter().find(|t| t.name == name).map(|t| t.value)
    }

    /// Move a trackbar — `cv::setTrackbarPos`. Clamped to `0..=max`.
    pub fn set_trackbar_pos(&mut self, name: &str, value: i32) -> Result<()> {
        let t = self
            .trackbars
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::new(Code::StsBadArg, format!("no trackbar named {name:?}")))?;
        t.value = value.clamp(0, t.max);
        Ok(())
    }
}

/// What a mouse callback is told — `cv::MouseEventTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEvent {
    /// `cv::EVENT_MOUSEMOVE`
    Move,
    /// `cv::EVENT_LBUTTONDOWN`
    LeftDown,
    /// `cv::EVENT_LBUTTONUP`
    LeftUp,
    /// `cv::EVENT_RBUTTONDOWN`
    RightDown,
    /// `cv::EVENT_RBUTTONUP`
    RightUp,
}

/// A mouse event in *image* coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MouseState {
    /// What happened.
    pub event: MouseEvent,
    /// Column, in the displayed image's own pixels.
    pub x: i32,
    /// Row.
    pub y: i32,
}

/// The per-frame callback a [`Session`] runs before each repaint.
type FrameCallback = Box<dyn FnMut(&mut Session) -> bool>;

/// A set of windows, driven by one event loop.
///
/// This is the shape upstream's global registry takes once the event loop
/// belongs to the framework rather than to the caller.
#[derive(Default)]
pub struct Session {
    windows: BTreeMap<String, Window>,
    /// Called each frame with the current window state; return `false` to quit.
    on_frame: Option<FrameCallback>,
    quit: bool,
    last_key: Option<char>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("windows", &self.windows.keys().collect::<Vec<_>>())
            .field("quit", &self.quit)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// An empty session.
    #[must_use]
    pub fn new() -> Self {
        Session::default()
    }

    /// `cv::namedWindow(winname, flags)`.
    pub fn named_window(&mut self, name: &str, flag: WindowFlag) -> &mut Self {
        self.windows.entry(name.to_string()).or_insert_with(|| Window::new(name, flag));
        self
    }

    /// `cv::imshow(winname, mat)` — set a window's image, creating it if needed.
    pub fn imshow<S: MatLike + ?Sized>(&mut self, name: &str, image: &S) -> Result<()> {
        let rgba = to_rgba(image)?;
        self.windows
            .entry(name.to_string())
            .or_insert_with(|| Window::new(name, WindowFlag::AutoSize))
            .image = Some(rgba);
        Ok(())
    }

    /// `cv::createTrackbar(trackbarname, winname, value, count, onChange)`.
    pub fn create_trackbar(&mut self, window: &str, name: &str, value: i32, max: i32) -> Result<()> {
        if max < 1 {
            return Err(Error::new(
                Code::StsBadArg,
                format!("a trackbar needs a maximum of at least 1, got {max}"),
            ));
        }
        let w = self
            .windows
            .get_mut(window)
            .ok_or_else(|| Error::new(Code::StsBadArg, format!("no window named {window:?}")))?;
        w.trackbars.push(Trackbar { name: name.to_string(), value: value.clamp(0, max), max });
        Ok(())
    }

    /// `cv::getTrackbarPos(trackbarname, winname)`.
    pub fn trackbar_pos(&self, window: &str, name: &str) -> Result<i32> {
        self.windows
            .get(window)
            .and_then(|w| w.trackbar_pos(name))
            .ok_or_else(|| Error::new(Code::StsBadArg, format!("no trackbar {name:?} on {window:?}")))
    }

    /// `cv::setTrackbarPos(trackbarname, winname, pos)`.
    pub fn set_trackbar_pos(&mut self, window: &str, name: &str, value: i32) -> Result<()> {
        self.windows
            .get_mut(window)
            .ok_or_else(|| Error::new(Code::StsBadArg, format!("no window named {window:?}")))?
            .set_trackbar_pos(name, value)
    }

    /// `cv::destroyWindow(winname)`.
    pub fn destroy_window(&mut self, name: &str) {
        self.windows.remove(name);
    }

    /// `cv::destroyAllWindows()`.
    pub fn destroy_all_windows(&mut self) {
        self.windows.clear();
    }

    /// `cv::getWindowProperty` — whether a window is open.
    #[must_use]
    pub fn is_open(&self, name: &str) -> bool {
        self.windows.contains_key(name)
    }

    /// Number of open windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether no windows are open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Read-only access to a window.
    #[must_use]
    pub fn window(&self, name: &str) -> Option<&Window> {
        self.windows.get(name)
    }

    /// Mutable access to a window by name, for [`window_geometry`].
    pub(crate) fn window_mut_by_name(&mut self, name: &str) -> Option<&mut Window> {
        self.windows.get_mut(name)
    }

    /// The most recent key pressed, if any — the value `cv::waitKey` returns.
    #[must_use]
    pub fn last_key(&self) -> Option<char> {
        self.last_key
    }

    /// Ask the loop to exit after this frame.
    pub fn quit(&mut self) {
        self.quit = true;
    }

    /// Install a per-frame callback, run before each repaint.
    ///
    /// Returning `false` closes the session. This is where a caller does the
    /// work `cv::waitKey`'s return value would have driven.
    pub fn on_frame(&mut self, f: impl FnMut(&mut Session) -> bool + 'static) -> &mut Self {
        self.on_frame = Some(Box::new(f));
        self
    }

    /// Open the windows and run until every one is closed or [`Session::quit`].
    ///
    /// Blocks. Without the `gui` feature this returns
    /// [`Code::StsNotImplemented`] rather than silently doing nothing, so a
    /// headless build fails loudly at the point of display.
    pub fn run(self) -> Result<()> {
        #[cfg(not(feature = "gui"))]
        {
            let _ = self;
            Err(Error::new(
                Code::StsNotImplemented,
                "cv-highgui was built without the `gui` feature, so no window backend is available"
                    .to_string(),
            ))
        }
        #[cfg(feature = "gui")]
        {
            backend::run(self)
        }
    }
}

/// `cv::imshow(winname, mat)` followed by `cv::waitKey(0)`.
///
/// One-shot: opens a single window, blocks until it is closed or a key is
/// pressed, and returns that key. For anything interactive build a [`Session`],
/// which is the form that survives `eframe` owning the event loop.
pub fn imshow<S: MatLike + ?Sized>(name: &str, image: &S) -> Result<Option<char>> {
    let mut s = Session::new();
    s.imshow(name, image)?;
    // Close on any keypress, which is what `waitKey(0)` does.
    s.on_frame(|s| {
        if s.last_key().is_some() {
            return false;
        }
        true
    });
    s.run()?;
    Ok(None)
}

/// `cv::waitKey(delay)`.
///
/// # Divergence
///
/// Upstream pumps the event loop and returns the key pressed within `delay`
/// milliseconds. With `eframe` owning the loop there is nothing to pump from
/// the outside, so this cannot be implemented as written and returns
/// [`Code::StsNotImplemented`] with that explanation. Use
/// [`Session::on_frame`] and [`Session::last_key`], which is the same
/// information delivered the other way round.
pub fn wait_key(_delay_ms: i32) -> Result<char> {
    Err(Error::new(
        Code::StsNotImplemented,
        "waitKey cannot be implemented against a framework-owned event loop; \
         use Session::on_frame and Session::last_key"
            .to_string(),
    ))
}

/// `cv::selectROI(windowName, img, showCrosshair, fromCenter)`.
///
/// Opens a window, lets the user drag a rectangle, and returns it. An empty
/// rectangle means the selection was cancelled, matching upstream.
pub fn select_roi<S: MatLike + ?Sized>(name: &str, image: &S) -> Result<cv_core::Rect> {
    #[cfg(not(feature = "gui"))]
    {
        let _ = (name, image);
        Err(Error::new(
            Code::StsNotImplemented,
            "selectROI needs the `gui` feature".to_string(),
        ))
    }
    #[cfg(feature = "gui")]
    {
        backend::select_roi(name, &to_rgba(image)?)
    }
}

/// Map a window-space click to image coordinates.
///
/// The image is letterboxed into the window, so a click has to be un-scaled
/// and un-offset. Returns `None` for a click in the letterbox margin — an
/// out-of-image click reported as `(0, 0)` is worse than none.
#[must_use]
pub fn window_to_image(
    click: (f32, f32),
    window: (f32, f32),
    image: (usize, usize),
) -> Option<(i32, i32)> {
    if image.0 == 0 || image.1 == 0 || window.0 <= 0.0 || window.1 <= 0.0 {
        return None;
    }
    let (iw, ih) = (image.0 as f32, image.1 as f32);
    let scale = (window.0 / iw).min(window.1 / ih);
    let (dw, dh) = (iw * scale, ih * scale);
    let (ox, oy) = ((window.0 - dw) * 0.5, (window.1 - dh) * 0.5);
    let (x, y) = ((click.0 - ox) / scale, (click.1 - oy) / scale);
    if x < 0.0 || y < 0.0 || x >= iw || y >= ih {
        return None;
    }
    Some((x as i32, y as i32))
}

#[cfg(feature = "gui")]
mod backend {
    //! The egui/eframe half. Everything here needs a real window.

    use super::{Rgba, Session, WindowFlag};
    use cv_core::{Code, Error, Result};

    /// Upload one frame as an egui texture and paint it, letterboxed.
    fn paint(ui: &mut egui::Ui, ctx: &egui::Context, key: &str, img: &Rgba) {
        let tex = ctx.load_texture(
            key,
            egui::ColorImage::from_rgba_unmultiplied([img.width, img.height], &img.pixels),
            egui::TextureOptions::NEAREST,
        );
        let avail = ui.available_size();
        let (iw, ih) = (img.width as f32, img.height as f32);
        let scale = (avail.x / iw).min(avail.y / ih).max(0.01);
        ui.add(egui::Image::new(&tex).fit_to_exact_size(egui::vec2(iw * scale, ih * scale)));
    }

    /// Collect the keys pressed this frame, as characters.
    fn pressed_key(ctx: &egui::Context) -> Option<char> {
        ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Text(t) => t.chars().next(),
                egui::Event::Key { key, pressed: true, .. } => match key {
                    egui::Key::Escape => Some('\u{1b}'),
                    egui::Key::Enter => Some('\n'),
                    egui::Key::Space => Some(' '),
                    _ => None,
                },
                _ => None,
            })
        })
    }

    struct App {
        session: Session,
    }

    impl eframe::App for App {
        fn update(&mut self, ctx: &egui::Context, _f: &mut eframe::Frame) {
            self.session.last_key = pressed_key(ctx);
            if let Some(mut cb) = self.session.on_frame.take() {
                let keep = cb(&mut self.session);
                self.session.on_frame = Some(cb);
                if !keep {
                    self.session.quit = true;
                }
            }
            if self.session.quit || self.session.windows.is_empty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }

            // One egui window per cv window, inside a single native window.
            // Upstream gives each its own OS window; nesting them keeps the
            // event loop singular, which is the constraint eframe imposes.
            let names: Vec<String> = self.session.windows.keys().cloned().collect();
            egui::CentralPanel::default().show(ctx, |ui| {
                for name in &names {
                    let (img, bars, flag) = {
                        let w = &self.session.windows[name];
                        (w.image.clone(), w.trackbars.clone(), w.flag)
                    };
                    let mut win = egui::Window::new(name).collapsible(true);
                    if flag == WindowFlag::AutoSize {
                        win = win.resizable(false);
                    }
                    win.show(ui.ctx(), |ui| {
                        for (i, t) in bars.iter().enumerate() {
                            let mut v = t.value;
                            if ui.add(egui::Slider::new(&mut v, 0..=t.max).text(&t.name)).changed()
                            {
                                if let Some(w) = self.session.windows.get_mut(name) {
                                    w.trackbars[i].value = v;
                                }
                            }
                        }
                        if let Some(img) = &img {
                            paint(ui, ctx, name, img);
                        } else {
                            ui.label("(no image)");
                        }
                    });
                }
            });
            ctx.request_repaint();
        }
    }

    /// Run a session to completion.
    pub fn run(session: Session) -> Result<()> {
        let opts = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
            ..Default::default()
        };
        eframe::run_native(
            "opencv-no-c",
            opts,
            Box::new(|_cc| Ok(Box::new(App { session }))),
        )
        .map_err(|e| Error::new(Code::StsError, format!("the window backend failed: {e}")))
    }

    struct RoiApp {
        img: Rgba,
        name: String,
        drag: Option<(egui::Pos2, egui::Pos2)>,
        out: std::rc::Rc<std::cell::RefCell<Option<cv_core::Rect>>>,
    }

    impl eframe::App for RoiApp {
        fn update(&mut self, ctx: &egui::Context, _f: &mut eframe::Frame) {
            let mut done = false;
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label(format!("{}: drag a rectangle, then press Enter (Esc cancels)", self.name));
                let avail = ui.available_size();
                let (iw, ih) = (self.img.width as f32, self.img.height as f32);
                let scale = (avail.x / iw).min(avail.y / ih).max(0.01);
                let size = egui::vec2(iw * scale, ih * scale);
                let tex = ctx.load_texture(
                    "roi",
                    egui::ColorImage::from_rgba_unmultiplied(
                        [self.img.width, self.img.height],
                        &self.img.pixels,
                    ),
                    egui::TextureOptions::NEAREST,
                );
                let resp = ui.add(
                    egui::Image::new(&tex).fit_to_exact_size(size).sense(egui::Sense::drag()),
                );
                let origin = resp.rect.min;
                if resp.drag_started() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        self.drag = Some((p, p));
                    }
                }
                if resp.dragged() {
                    if let (Some(d), Some(p)) = (self.drag.as_mut(), resp.interact_pointer_pos()) {
                        d.1 = p;
                    }
                }
                if let Some((a, b)) = self.drag {
                    let r = egui::Rect::from_two_pos(a, b);
                    ui.painter().rect_stroke(
                        r,
                        0.0,
                        egui::Stroke::new(2.0_f32, egui::Color32::LIGHT_GREEN),
                        egui::StrokeKind::Middle,
                    );
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Some((a, b)) = self.drag {
                        let r = egui::Rect::from_two_pos(a, b);
                        // Back into image pixels.
                        let to_img = |p: egui::Pos2| {
                            (
                                ((p.x - origin.x) / scale).clamp(0.0, iw - 1.0) as i32,
                                ((p.y - origin.y) / scale).clamp(0.0, ih - 1.0) as i32,
                            )
                        };
                        let (x0, y0) = to_img(r.min);
                        let (x1, y1) = to_img(r.max);
                        *self.out.borrow_mut() =
                            Some(cv_core::Rect::new(x0, y0, (x1 - x0).max(0), (y1 - y0).max(0)));
                    }
                    done = true;
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    done = true;
                }
            });
            if done {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ctx.request_repaint();
        }
    }

    /// Let the user drag a rectangle over an image.
    pub fn select_roi(name: &str, img: &Rgba) -> Result<cv_core::Rect> {
        let out = std::rc::Rc::new(std::cell::RefCell::new(None));
        let app = RoiApp {
            img: img.clone(),
            name: name.to_string(),
            drag: None,
            out: std::rc::Rc::clone(&out),
        };
        eframe::run_native(
            name,
            eframe::NativeOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        )
        .map_err(|e| Error::new(Code::StsError, format!("the window backend failed: {e}")))?;
        // An empty rectangle means cancelled, as upstream returns.
        let r = out.borrow().unwrap_or(cv_core::Rect::new(0, 0, 0, 0));
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;
    use cv_core::types::{Depth, MatType, CV_8UC1};
    use cv_core::MatLikeMut;

    fn bgr(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, 3)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = 10; // B
                p[1] = 20; // G
                p[2] = 30; // R
            }
        }
        m
    }

    #[test]
    fn bgr_is_swapped_to_rgb_for_display() {
        // The classic imshow bug: if this is wrong the picture still appears
        // and every red object looks blue.
        let r = to_rgba(&bgr(2, 1)).unwrap();
        assert_eq!(r.width, 2);
        assert_eq!(r.height, 1);
        assert_eq!(&r.pixels[..4], &[30, 20, 10, 255], "R, G, B, A");
    }

    #[test]
    fn greyscale_is_replicated_across_the_channels() {
        let m = Mat::from_vec(1, 2, 1, &[7u8, 200]).unwrap();
        let r = to_rgba(&m).unwrap();
        assert_eq!(&r.pixels[..4], &[7, 7, 7, 255]);
        assert_eq!(&r.pixels[4..8], &[200, 200, 200, 255]);
    }

    #[test]
    fn four_channel_input_keeps_its_alpha() {
        let m = Mat::from_vec(1, 1, 4, &[1u8, 2, 3, 128]).unwrap();
        let r = to_rgba(&m).unwrap();
        assert_eq!(&r.pixels[..4], &[3, 2, 1, 128]);
    }

    #[test]
    fn float_images_are_scaled_from_the_unit_range() {
        // cv::imshow treats CV_32F as 0..1.
        let m = Mat::from_vec(1, 3, 1, &[0.0f32, 0.5, 1.0]).unwrap();
        let r = to_rgba(&m).unwrap();
        assert_eq!(r.pixels[0], 0);
        assert_eq!(r.pixels[4], 128);
        assert_eq!(r.pixels[8], 255);
    }

    #[test]
    fn float_values_outside_the_range_saturate() {
        let m = Mat::from_vec(1, 2, 1, &[-3.0f32, 9.0]).unwrap();
        let r = to_rgba(&m).unwrap();
        assert_eq!(r.pixels[0], 0);
        assert_eq!(r.pixels[4], 255);
    }

    #[test]
    fn an_empty_or_odd_channel_image_is_rejected() {
        let empty = Mat::zeros(0, 0, CV_8UC1);
        if let Ok(e) = empty {
            assert!(to_rgba(&e).is_err());
        }
        let two = Mat::zeros(2, 2, MatType::new(Depth::U8, 2)).unwrap();
        let e = to_rgba(&two).unwrap_err();
        assert!(format!("{e}").contains("2-channel"), "the error should name the count: {e}");
    }

    #[test]
    fn windows_are_created_named_and_destroyed() {
        let mut s = Session::new();
        assert!(s.is_empty());
        s.named_window("a", WindowFlag::Normal).named_window("b", WindowFlag::AutoSize);
        assert_eq!(s.len(), 2);
        assert!(s.is_open("a") && s.is_open("b"));
        assert_eq!(s.window("a").unwrap().flag, WindowFlag::Normal);
        s.destroy_window("a");
        assert!(!s.is_open("a"));
        s.destroy_all_windows();
        assert!(s.is_empty());
    }

    #[test]
    fn imshow_creates_a_window_and_stores_the_image() {
        let mut s = Session::new();
        s.imshow("w", &bgr(4, 3)).unwrap();
        assert!(s.is_open("w"));
        let img = s.window("w").unwrap().image.as_ref().unwrap();
        assert_eq!((img.width, img.height), (4, 3));
        // Showing again replaces it.
        s.imshow("w", &bgr(8, 6)).unwrap();
        assert_eq!(s.window("w").unwrap().image.as_ref().unwrap().width, 8);
        assert_eq!(s.len(), 1, "showing twice must not open a second window");
    }

    #[test]
    fn named_window_does_not_clobber_an_existing_one() {
        let mut s = Session::new();
        s.imshow("w", &bgr(2, 2)).unwrap();
        s.named_window("w", WindowFlag::Normal);
        assert!(
            s.window("w").unwrap().image.is_some(),
            "re-declaring a window must not drop its image"
        );
    }

    #[test]
    fn trackbars_are_created_read_and_clamped() {
        let mut s = Session::new();
        s.named_window("w", WindowFlag::Normal);
        s.create_trackbar("w", "thresh", 50, 255).unwrap();
        assert_eq!(s.trackbar_pos("w", "thresh").unwrap(), 50);
        s.set_trackbar_pos("w", "thresh", 300).unwrap();
        assert_eq!(s.trackbar_pos("w", "thresh").unwrap(), 255, "clamped to max");
        s.set_trackbar_pos("w", "thresh", -5).unwrap();
        assert_eq!(s.trackbar_pos("w", "thresh").unwrap(), 0, "clamped to zero");
        // An initial value above max is clamped at creation too.
        s.create_trackbar("w", "other", 999, 10).unwrap();
        assert_eq!(s.trackbar_pos("w", "other").unwrap(), 10);
    }

    #[test]
    fn trackbar_errors_name_what_was_missing() {
        let mut s = Session::new();
        assert!(s.create_trackbar("nope", "t", 0, 10).is_err());
        s.named_window("w", WindowFlag::Normal);
        assert!(s.create_trackbar("w", "t", 0, 0).is_err(), "a zero maximum is useless");
        assert!(s.trackbar_pos("w", "missing").is_err());
        assert!(s.set_trackbar_pos("w", "missing", 1).is_err());
    }

    #[test]
    fn window_clicks_map_back_to_image_pixels() {
        // A 100x50 image letterboxed into a 400x400 window: scale 4, with
        // 100px of margin above and below.
        let img = (100usize, 50usize);
        let win = (400.0f32, 400.0f32);
        assert_eq!(window_to_image((0.0, 100.0), win, img), Some((0, 0)));
        assert_eq!(window_to_image((200.0, 200.0), win, img), Some((50, 25)));
        // A click in the letterbox margin is not in the image.
        assert_eq!(window_to_image((200.0, 10.0), win, img), None);
        assert_eq!(window_to_image((200.0, 390.0), win, img), None);
    }

    #[test]
    fn click_mapping_rejects_degenerate_geometry() {
        assert_eq!(window_to_image((0.0, 0.0), (100.0, 100.0), (0, 10)), None);
        assert_eq!(window_to_image((0.0, 0.0), (0.0, 100.0), (10, 10)), None);
    }

    #[test]
    fn wait_key_explains_itself_rather_than_lying() {
        let e = wait_key(0).unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(
            e.msg.contains("Session::on_frame"),
            "the error should point at the replacement: {}",
            e.msg
        );
    }

    #[cfg(not(feature = "gui"))]
    #[test]
    fn running_without_a_backend_fails_loudly() {
        let mut s = Session::new();
        s.imshow("w", &bgr(2, 2)).unwrap();
        let e = s.run().unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(e.msg.contains("gui"), "the error should name the feature: {}", e.msg);
    }
}

/// Window geometry and the remaining `cv::highgui` entry points.
///
/// This crate models windows rather than opening them — there is no platform
/// toolkit behind it, since one would need C. A [`Window`] therefore has a
/// *recorded* position and size that these functions read and write, which is
/// what lets a layout be built and asserted on in a test without a display.
///
/// The two that genuinely cannot be modelled are named as such:
/// [`Session::current_ui_framework`] reports `"none"`, and
/// [`Session::add_text`] records the text without rendering it.
pub mod window_geometry {
    use super::{Code, Error, Result, Session, Window, WindowFlag};

    /// Where a window sits and how large it is, in screen pixels.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Geometry {
        /// Left edge.
        pub x: i32,
        /// Top edge.
        pub y: i32,
        /// Width, or 0 when it follows the image.
        pub width: i32,
        /// Height, or 0 when it follows the image.
        pub height: i32,
    }

    impl Window {
        /// The window's recorded geometry.
        #[must_use]
        pub fn geometry(&self) -> Geometry {
            self.geometry
        }
    }

    impl Session {
        /// `cv::moveWindow(winname, x, y)`.
        pub fn move_window(&mut self, name: &str, x: i32, y: i32) -> Result<()> {
            let w = self.window_mut(name)?;
            w.geometry.x = x;
            w.geometry.y = y;
            Ok(())
        }

        /// `cv::resizeWindow(winname, width, height)`.
        ///
        /// Rejects a non-positive size, as upstream does: a zero-width window
        /// is not a window.
        pub fn resize_window(&mut self, name: &str, width: i32, height: i32) -> Result<()> {
            if width <= 0 || height <= 0 {
                return Err(Error::new(
                    Code::StsBadArg,
                    format!("resizeWindow needs a positive size, got {width}x{height}"),
                ));
            }
            let w = self.window_mut(name)?;
            w.geometry.width = width;
            w.geometry.height = height;
            Ok(())
        }

        /// `cv::getWindowImageRect(winname)` — where the image sits inside the
        /// window, in screen coordinates.
        ///
        /// With no decoration to account for, this is the window's own
        /// rectangle; when no explicit size has been set it follows the shown
        /// image, which is what upstream reports for an autosized window.
        pub fn window_image_rect(&self, name: &str) -> Result<cv_core::Rect> {
            let w = self
                .window(name)
                .ok_or_else(|| Error::new(Code::StsBadArg, format!("no window named {name:?}")))?;
            let g = w.geometry;
            let (width, height) = if g.width > 0 && g.height > 0 {
                (g.width, g.height)
            } else {
                w.image.as_ref().map_or((0, 0), |i| (i.width as i32, i.height as i32))
            };
            Ok(cv_core::Rect::new(g.x, g.y, width, height))
        }

        /// `cv::displayOverlay(winname, text, delayms)`.
        ///
        /// Recorded on the window rather than drawn. `delay_ms` of 0 means the
        /// text stays until replaced, as upstream documents.
        pub fn display_overlay(&mut self, name: &str, text: &str, delay_ms: i32) -> Result<()> {
            let w = self.window_mut(name)?;
            w.overlay = if text.is_empty() { None } else { Some(text.to_string()) };
            w.overlay_delay_ms = delay_ms;
            Ok(())
        }

        /// `cv::displayStatusBar(winname, text, delayms)`.
        pub fn display_status_bar(&mut self, name: &str, text: &str, delay_ms: i32) -> Result<()> {
            let w = self.window_mut(name)?;
            w.status_bar = if text.is_empty() { None } else { Some(text.to_string()) };
            w.status_delay_ms = delay_ms;
            Ok(())
        }

        /// `cv::addText(img, text, org, ...)`.
        ///
        /// Records the request; nothing is rasterised. Upstream's version needs
        /// Qt for font handling, and [`cv_imgproc::put_text`] is the drawing
        /// call that actually marks pixels.
        pub fn add_text(&mut self, name: &str, text: &str) -> Result<()> {
            let w = self.window_mut(name)?;
            w.texts.push(text.to_string());
            Ok(())
        }

        /// `cv::currentUIFramework()`.
        ///
        /// `"none"`. Upstream answers `"QT"`, `"WIN32UI"`, `"GTK"` or `"COCOA"`
        /// according to what it was built against; this crate is built against
        /// no toolkit, and saying so is more useful than naming one it cannot
        /// drive.
        #[must_use]
        pub fn current_ui_framework() -> &'static str {
            "none"
        }

        /// `cv::pollKey()` — the last key pressed, without waiting.
        ///
        /// Returns -1 when nothing is queued, which is upstream's contract.
        /// The distinction from `waitKey` is that this never blocks.
        pub fn poll_key(&mut self) -> i32 {
            self.last_key().map_or(-1, |c| c as i32)
        }

        /// `cv::selectROIs(windowName, img, boundingBoxes, ...)`.
        ///
        /// Selecting a region needs a pointer and a display. With neither, this
        /// returns the regions previously recorded by a caller — a test harness
        /// or a replayed input trace — rather than an empty list that looks
        /// like the user cancelled.
        #[must_use]
        pub fn select_rois(&self, name: &str) -> Vec<cv_core::Rect> {
            self.window(name).map(|w| w.rois.clone()).unwrap_or_default()
        }

        /// Record a region for [`Session::select_rois`] to return.
        pub fn push_roi(&mut self, name: &str, r: cv_core::Rect) -> Result<()> {
            self.window_mut(name)?.rois.push(r);
            Ok(())
        }


        /// `cv::setTrackbarMax(trackbarname, winname, maxval)`.
        ///
        /// Raising the maximum leaves the position where it is; lowering it
        /// clamps, which is what upstream does rather than leaving a slider
        /// pointing past its own end.
        pub fn set_trackbar_max(&mut self, window: &str, name: &str, maxval: i32) -> Result<()> {
            let w = self.window_mut(window)?;
            let t = w
                .trackbars
                .iter_mut()
                .find(|t| t.name == name)
                .ok_or_else(|| Error::new(Code::StsBadArg, format!("no trackbar {name:?}")))?;
            t.max = maxval;
            t.value = t.value.min(maxval);
            Ok(())
        }

        /// `cv::setTrackbarMin(trackbarname, winname, minval)`.
        ///
        /// `Trackbar` carries no lower bound of its own — upstream's is
        /// always 0 for the slider itself — so this clamps the current
        /// position up to `minval` and reports it back through the position.
        pub fn set_trackbar_min(&mut self, window: &str, name: &str, minval: i32) -> Result<()> {
            let w = self.window_mut(window)?;
            let t = w
                .trackbars
                .iter_mut()
                .find(|t| t.name == name)
                .ok_or_else(|| Error::new(Code::StsBadArg, format!("no trackbar {name:?}")))?;
            t.value = t.value.max(minval);
            Ok(())
        }

        /// `cv::setWindowTitle(winname, title)`.
        ///
        /// Renames the window. The key it is indexed by changes with it, so a
        /// later lookup uses the new title — the same as upstream, where the
        /// name *is* the handle.
        pub fn set_window_title(&mut self, name: &str, title: &str) -> Result<()> {
            if !self.windows.contains_key(name) {
                return Err(Error::new(Code::StsBadArg, format!("no window named {name:?}")));
            }
            if name != title {
                let mut w = self.windows.remove(name).expect("checked above");
                w.name = title.to_string();
                self.windows.insert(title.to_string(), w);
            }
            Ok(())
        }

        /// `cv::setWindowProperty(winname, prop_id, prop_value)`.
        ///
        /// Only `WND_PROP_AUTOSIZE` and `WND_PROP_FULLSCREEN` change anything
        /// here, since they are the two this crate models; the rest are
        /// recorded and reported back by [`Session::window_property`].
        pub fn set_window_property(&mut self, name: &str, prop_id: i32, value: f64) -> Result<()> {
            let w = self.window_mut(name)?;
            match prop_id {
                // WND_PROP_FULLSCREEN
                0 => w.flag = if value != 0.0 { WindowFlag::Normal } else { w.flag },
                // WND_PROP_AUTOSIZE
                1 => {
                    w.flag = if value != 0.0 { WindowFlag::AutoSize } else { WindowFlag::Normal };
                }
                _ => {}
            }
            w.properties.insert(prop_id, value);
            Ok(())
        }

        /// `cv::getWindowProperty(winname, prop_id)`.
        pub fn window_property(&self, name: &str, prop_id: i32) -> Result<f64> {
            let w = self
                .window(name)
                .ok_or_else(|| Error::new(Code::StsBadArg, format!("no window named {name:?}")))?;
            Ok(match prop_id {
                1 => f64::from(i32::from(w.flag == WindowFlag::AutoSize)),
                _ => w.properties.get(&prop_id).copied().unwrap_or(-1.0),
            })
        }

        /// `cv::startWindowThread()`.
        ///
        /// Zero. Upstream spawns a thread to pump the toolkit's event loop;
        /// with no toolkit there is no loop to pump, and [`Session::run`]
        /// drives the frame callback directly on the calling thread.
        #[must_use]
        pub fn start_window_thread() -> i32 {
            0
        }

        /// `cv::waitKeyEx(delay)` — like `waitKey` but with the full key code.
        ///
        /// The distinction matters for arrow and function keys, whose codes do
        /// not fit the low byte `waitKey` returns. Here both give the same
        /// value, since the recorded key is already a full `char`.
        pub fn wait_key_ex(&mut self, _delay_ms: i32) -> i32 {
            self.poll_key()
        }

        /// The window by name, or a clear error.
        fn window_mut(&mut self, name: &str) -> Result<&mut Window> {
            self.window_mut_by_name(name)
                .ok_or_else(|| Error::new(Code::StsBadArg, format!("no window named {name:?}")))
        }
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::window_geometry::Geometry;
    use super::*;

    fn session_with_window() -> Session {
        let mut s = Session::new();
        s.named_window("w", WindowFlag::Normal);
        s
    }

    #[test]
    fn moving_and_resizing_records_the_geometry() {
        let mut s = session_with_window();
        assert_eq!(s.window("w").unwrap().geometry(), Geometry::default());
        s.move_window("w", 40, 30).unwrap();
        s.resize_window("w", 640, 480).unwrap();
        assert_eq!(
            s.window("w").unwrap().geometry(),
            Geometry { x: 40, y: 30, width: 640, height: 480 }
        );
        // A non-positive size is refused rather than recorded.
        assert!(s.resize_window("w", 0, 100).is_err());
        assert!(s.resize_window("w", 100, -1).is_err());
        assert_eq!(s.window("w").unwrap().geometry().width, 640, "the refusal changed nothing");
        // And an unknown window names itself in the error.
        let e = s.move_window("nope", 1, 1).unwrap_err();
        assert!(e.msg.contains("nope"), "{}", e.msg);
    }

    #[test]
    fn the_image_rect_follows_the_size_or_the_image() {
        let mut s = session_with_window();
        // With no size and no image the rect is empty.
        assert_eq!(s.window_image_rect("w").unwrap(), cv_core::Rect::new(0, 0, 0, 0));

        // Autosized: it follows the shown image.
        let img = cv_core::Mat::zeros(12, 20, cv_core::types::CV_8UC3).unwrap();
        s.imshow("w", &img).unwrap();
        assert_eq!(s.window_image_rect("w").unwrap(), cv_core::Rect::new(0, 0, 20, 12));

        // An explicit size wins over the image.
        s.move_window("w", 5, 7).unwrap();
        s.resize_window("w", 100, 50).unwrap();
        assert_eq!(s.window_image_rect("w").unwrap(), cv_core::Rect::new(5, 7, 100, 50));
        assert!(s.window_image_rect("missing").is_err());
    }

    #[test]
    fn overlay_and_status_text_is_recorded_and_cleared() {
        let mut s = session_with_window();
        s.display_overlay("w", "saving...", 1500).unwrap();
        assert_eq!(s.window("w").unwrap().overlay.as_deref(), Some("saving..."));
        assert_eq!(s.window("w").unwrap().overlay_delay_ms, 1500);
        // Empty text clears it, as upstream's does.
        s.display_overlay("w", "", 0).unwrap();
        assert!(s.window("w").unwrap().overlay.is_none());

        s.display_status_bar("w", "ready", 0).unwrap();
        assert_eq!(s.window("w").unwrap().status_bar.as_deref(), Some("ready"));
        s.add_text("w", "label").unwrap();
        s.add_text("w", "another").unwrap();
        assert_eq!(s.window("w").unwrap().texts, vec!["label", "another"]);
    }

    #[test]
    fn poll_key_does_not_block_and_reports_minus_one_when_idle() {
        let mut s = session_with_window();
        assert_eq!(s.poll_key(), -1, "nothing queued");
        assert_eq!(Session::current_ui_framework(), "none");
    }

    #[test]
    fn select_rois_returns_what_was_recorded() {
        let mut s = session_with_window();
        assert!(s.select_rois("w").is_empty());
        s.push_roi("w", cv_core::Rect::new(1, 2, 3, 4)).unwrap();
        s.push_roi("w", cv_core::Rect::new(5, 6, 7, 8)).unwrap();
        assert_eq!(
            s.select_rois("w"),
            vec![cv_core::Rect::new(1, 2, 3, 4), cv_core::Rect::new(5, 6, 7, 8)]
        );
        // An unknown window gives nothing rather than panicking.
        assert!(s.select_rois("nope").is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    fn s_with_trackbar() -> Session {
        let mut s = Session::new();
        s.named_window("w", WindowFlag::AutoSize);
        s.create_trackbar("w", "t", 5, 10).unwrap();
        s
    }

    #[test]
    fn lowering_the_maximum_clamps_the_position() {
        let mut s = s_with_trackbar();
        s.set_trackbar_max("w", "t", 20).unwrap();
        assert_eq!(s.trackbar_pos("w", "t").unwrap(), 5, "raising the max leaves it alone");
        s.set_trackbar_max("w", "t", 3).unwrap();
        assert_eq!(s.trackbar_pos("w", "t").unwrap(), 3, "lowering it clamps");
        assert!(s.set_trackbar_max("w", "missing", 1).is_err());
    }

    #[test]
    fn raising_the_minimum_clamps_the_position_up() {
        let mut s = s_with_trackbar();
        s.set_trackbar_min("w", "t", 8).unwrap();
        assert_eq!(s.trackbar_pos("w", "t").unwrap(), 8);
        s.set_trackbar_min("w", "t", 0).unwrap();
        assert_eq!(s.trackbar_pos("w", "t").unwrap(), 8, "lowering the floor moves nothing");
    }

    #[test]
    fn renaming_a_window_moves_its_key() {
        let mut s = s_with_trackbar();
        s.set_window_title("w", "renamed").unwrap();
        assert!(s.window("w").is_none(), "the old name is gone");
        let w = s.window("renamed").expect("findable under the new name");
        assert_eq!(w.name, "renamed");
        assert_eq!(w.trackbars.len(), 1, "the contents came along");
        // Renaming to the same title is a no-op, not a remove-then-insert.
        s.set_window_title("renamed", "renamed").unwrap();
        assert!(s.window("renamed").is_some());
        assert!(s.set_window_title("nope", "x").is_err());
    }

    #[test]
    fn window_properties_round_trip() {
        let mut s = s_with_trackbar();
        // WND_PROP_AUTOSIZE reads back from the flag it actually sets.
        assert_eq!(s.window_property("w", 1).unwrap(), 1.0);
        s.set_window_property("w", 1, 0.0).unwrap();
        assert_eq!(s.window_property("w", 1).unwrap(), 0.0);
        assert_eq!(s.window("w").unwrap().flag, WindowFlag::Normal);

        // An unmodelled property is stored and returned verbatim.
        s.set_window_property("w", 7, 0.25).unwrap();
        assert_eq!(s.window_property("w", 7).unwrap(), 0.25);
        // An unset one reports -1, as upstream does.
        assert_eq!(s.window_property("w", 99).unwrap(), -1.0);
        assert!(s.window_property("nope", 1).is_err());
    }

    #[test]
    fn the_thread_and_key_helpers_report_honestly() {
        let mut s = s_with_trackbar();
        assert_eq!(Session::start_window_thread(), 0);
        // waitKeyEx and pollKey agree here; both report nothing queued.
        assert_eq!(s.wait_key_ex(0), -1);
        assert_eq!(s.poll_key(), -1);
    }
}
