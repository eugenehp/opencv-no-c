//! `cv::ogl` — the OpenGL interop constants
//! (`modules/core/include/opencv2/core/opengl.hpp`).
//!
//! Only the enumerations are ported. `ogl::Buffer` and `ogl::Texture2D` wrap
//! live GL object names, which needs a GL context — something a library cannot
//! create for itself and this port does not link. The constants, though, are
//! just the GL token values, and code that names a primitive type or a buffer
//! target should not have to hard-code `0x8892`.
//!
//! Every value here equals the corresponding `GL_*` token, so passing one
//! through to a GL binding crate works directly.

/// `cv::ogl::Buffer::Target` — what a buffer object will be bound as.
pub mod target {
    /// `Buffer::ARRAY_BUFFER` — vertex data. `GL_ARRAY_BUFFER`.
    pub const ARRAY_BUFFER: i32 = 0x8892;
    /// `Buffer::ELEMENT_ARRAY_BUFFER` — indices. `GL_ELEMENT_ARRAY_BUFFER`.
    pub const ELEMENT_ARRAY_BUFFER: i32 = 0x8893;
    /// `Buffer::PIXEL_PACK_BUFFER` — reading back from a texture.
    pub const PIXEL_PACK_BUFFER: i32 = 0x88EB;
    /// `Buffer::PIXEL_UNPACK_BUFFER` — writing into a texture.
    pub const PIXEL_UNPACK_BUFFER: i32 = 0x88EC;
}

/// `cv::ogl::Texture2D::Format` — a texture's internal format.
pub mod format {
    /// `Texture2D::NONE`
    pub const NONE: i32 = 0;
    /// `Texture2D::DEPTH_COMPONENT` — `GL_DEPTH_COMPONENT`.
    pub const DEPTH_COMPONENT: i32 = 0x1902;
    /// `Texture2D::RGB` — `GL_RGB`.
    pub const RGB: i32 = 0x1907;
    /// `Texture2D::RGBA` — `GL_RGBA`.
    pub const RGBA: i32 = 0x1908;
}

/// `cv::ogl::RenderModes` — the primitive `ogl::render` draws.
pub mod render_mode {
    /// `ogl::POINTS`
    pub const POINTS: i32 = 0x0000;
    /// `ogl::LINES`
    pub const LINES: i32 = 0x0001;
    /// `ogl::LINE_LOOP`
    pub const LINE_LOOP: i32 = 0x0002;
    /// `ogl::LINE_STRIP`
    pub const LINE_STRIP: i32 = 0x0003;
    /// `ogl::TRIANGLES`
    pub const TRIANGLES: i32 = 0x0004;
    /// `ogl::TRIANGLE_STRIP`
    pub const TRIANGLE_STRIP: i32 = 0x0005;
    /// `ogl::TRIANGLE_FAN`
    pub const TRIANGLE_FAN: i32 = 0x0006;
    /// `ogl::QUADS`
    pub const QUADS: i32 = 0x0007;
    /// `ogl::QUAD_STRIP`
    pub const QUAD_STRIP: i32 = 0x0008;
    /// `ogl::POLYGON`
    pub const POLYGON: i32 = 0x0009;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_values_are_the_gl_tokens() {
        assert_eq!(target::ARRAY_BUFFER, 0x8892);
        assert_eq!(target::PIXEL_UNPACK_BUFFER, 0x88EC);
        assert_eq!(format::RGB, 0x1907);
        assert_eq!(format::RGBA, format::RGB + 1);
        assert_eq!(format::DEPTH_COMPONENT, 0x1902);
        assert_eq!(format::NONE, 0);
    }

    #[test]
    fn render_modes_are_numbered_in_gl_order() {
        use render_mode::*;
        // GL numbers the primitives consecutively from zero.
        let all = [
            POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN,
            QUADS, QUAD_STRIP, POLYGON,
        ];
        for (i, v) in all.iter().enumerate() {
            assert_eq!(*v, i as i32, "primitive {i}");
        }
    }
}
