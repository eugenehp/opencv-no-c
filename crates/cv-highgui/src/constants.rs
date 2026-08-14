//! `cv::highgui` window, event and control constants
//! (`modules/highgui/include/opencv2/highgui.hpp`).
//!
//! Window flags, mouse and keyboard event codes, trackbar and button types,
//! and the Qt-only font settings. Generated from the upstream header.
//!
//! # No windows are opened here
//!
//! A GUI needs a platform toolkit — Cocoa, Win32, GTK — reached through C.
//! This crate does not link one, so it draws nothing; see the crate header for
//! what it does provide instead.
//!
//! These constants are still the right thing to declare. Event codes in
//! particular travel: a callback signature, a recorded input trace, or a test
//! fixture written against `EVENT_LBUTTONDOWN` means the same integer whether
//! or not this process can open a window.

/// `WINDOW_OPENGL` — window with opengl support.
pub const WINDOW_OPENGL: i32 = 4096;
/// `WINDOW_FULLSCREEN` — change the window to fullscreen.
pub const WINDOW_FULLSCREEN: i32 = 1;
/// `WINDOW_FREERATIO` — the image expends as much as it can (no ratio
///   constraint).
pub const WINDOW_FREERATIO: i32 = 256;
/// `WINDOW_GUI_EXPANDED` — status bar and tool bar.
pub const WINDOW_GUI_EXPANDED: i32 = 0;
/// `WINDOW_GUI_NORMAL` — old fashious way.
pub const WINDOW_GUI_NORMAL: i32 = 16;
/// `WND_PROP_FULLSCREEN` — fullscreen property (can be WINDOW_NORMAL or
///   WINDOW_FULLSCREEN).
pub const WND_PROP_FULLSCREEN: i32 = 0;
/// `WND_PROP_AUTOSIZE` — autosize property (can be WINDOW_NORMAL or
///   WINDOW_AUTOSIZE).
pub const WND_PROP_AUTOSIZE: i32 = 1;
/// `WND_PROP_ASPECT_RATIO` — window's aspect ration (can be set to
///   WINDOW_FREERATIO or WINDOW_KEEPRATIO).
pub const WND_PROP_ASPECT_RATIO: i32 = 2;
/// `WND_PROP_OPENGL` — opengl support.
pub const WND_PROP_OPENGL: i32 = 3;
/// `WND_PROP_VISIBLE` — checks whether the window exists and is visible.
pub const WND_PROP_VISIBLE: i32 = 4;
/// `WND_PROP_TOPMOST` — property to toggle normal window being topmost or
///   not.
pub const WND_PROP_TOPMOST: i32 = 5;
/// `WND_PROP_VSYNC` — enable or disable VSYNC (in OpenGL mode).
pub const WND_PROP_VSYNC: i32 = 6;
/// `EVENT_MBUTTONDOWN` — indicates that the middle mouse button is pressed.
pub const EVENT_MBUTTONDOWN: i32 = 3;
/// `EVENT_MBUTTONUP` — indicates that middle mouse button is released.
pub const EVENT_MBUTTONUP: i32 = 6;
/// `EVENT_LBUTTONDBLCLK` — indicates that left mouse button is double
///   clicked.
pub const EVENT_LBUTTONDBLCLK: i32 = 7;
/// `EVENT_RBUTTONDBLCLK` — indicates that right mouse button is double
///   clicked.
pub const EVENT_RBUTTONDBLCLK: i32 = 8;
/// `EVENT_MBUTTONDBLCLK` — indicates that middle mouse button is double
///   clicked.
pub const EVENT_MBUTTONDBLCLK: i32 = 9;
/// `EVENT_MOUSEWHEEL` — positive and negative values mean forward and
///   backward scrolling, respectively.
pub const EVENT_MOUSEWHEEL: i32 = 10;
/// `EVENT_MOUSEHWHEEL` — positive and negative values mean right and left
///   scrolling, respectively.
pub const EVENT_MOUSEHWHEEL: i32 = 11;
/// `EVENT_FLAG_LBUTTON` — indicates that the left mouse button is down.
pub const EVENT_FLAG_LBUTTON: i32 = 1;
/// `EVENT_FLAG_RBUTTON` — indicates that the right mouse button is down.
pub const EVENT_FLAG_RBUTTON: i32 = 2;
/// `EVENT_FLAG_MBUTTON` — indicates that the middle mouse button is down.
pub const EVENT_FLAG_MBUTTON: i32 = 4;
/// `EVENT_FLAG_CTRLKEY` — indicates that CTRL Key is pressed.
pub const EVENT_FLAG_CTRLKEY: i32 = 8;
/// `EVENT_FLAG_SHIFTKEY` — indicates that SHIFT Key is pressed.
pub const EVENT_FLAG_SHIFTKEY: i32 = 16;
/// `EVENT_FLAG_ALTKEY` — indicates that ALT Key is pressed.
pub const EVENT_FLAG_ALTKEY: i32 = 32;
/// `QT_FONT_LIGHT` — Weight of 25.
pub const QT_FONT_LIGHT: i32 = 25;
/// `QT_FONT_NORMAL` — Weight of 50.
pub const QT_FONT_NORMAL: i32 = 50;
/// `QT_FONT_DEMIBOLD` — Weight of 63.
pub const QT_FONT_DEMIBOLD: i32 = 63;
/// `QT_FONT_BOLD` — Weight of 75.
pub const QT_FONT_BOLD: i32 = 75;
/// `QT_FONT_BLACK` — Weight of 87.
pub const QT_FONT_BLACK: i32 = 87;
/// `QT_STYLE_NORMAL` — Normal font.
pub const QT_STYLE_NORMAL: i32 = 0;
/// `QT_STYLE_ITALIC` — Italic font.
pub const QT_STYLE_ITALIC: i32 = 1;
/// `QT_STYLE_OBLIQUE` — Oblique font.
pub const QT_STYLE_OBLIQUE: i32 = 2;
/// `QT_PUSH_BUTTON` — Push button.
pub const QT_PUSH_BUTTON: i32 = 0;
/// `QT_CHECKBOX` — Checkbox button.
pub const QT_CHECKBOX: i32 = 1;
/// `QT_RADIOBOX` — Radiobox button.
pub const QT_RADIOBOX: i32 = 2;
/// `QT_NEW_BUTTONBAR` — Button should create a new buttonbar.
pub const QT_NEW_BUTTONBAR: i32 = 1024;
