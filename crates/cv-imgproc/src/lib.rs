//! Pure-Rust port of the OpenCV `imgproc` module.
//!
//! Targets upstream **OpenCV 4.15.0-dev** (`77dfa297d08fdecdc509fc01ad92a2e9ec776a57`).
//!
//! Where upstream uses fixed-point arithmetic — grayscale conversion, HSV,
//! YCrCb, and 8-bit bilinear resize — this port reproduces the same integer
//! expressions rather than computing in floating point, so results match
//! level-for-level rather than approximately.

#![doc(html_root_url = "https://docs.rs/cv-imgproc/0.1.0")]
#![warn(missing_docs)]

pub mod constants;
pub mod color;
pub mod color2;
pub mod colormap_data;
pub mod fitting2;
pub mod misc2;
pub mod subdiv;
pub mod yuv;
pub mod contours;
pub mod draw2;
pub mod drawing;
pub mod edge;
pub mod extra;
pub mod features;
pub mod filter;
pub mod fitting;
pub mod demosaic;
pub mod hal;
pub mod objects;
pub mod hough;
pub mod hist2;
pub mod hist;
pub mod misc;
pub mod morph;
pub mod pyramids;
pub mod resize;
pub mod shape;
pub mod text;
pub mod threshold;
pub mod warp;

pub use contours::{
    distance_transform, draw_contours, find_contours, flood_fill, flood_mask, hu_moments,
    is_contour_convex, match_shapes, watershed, ContourApproximation, DistanceMetric, Hierarchy,
    RetrievalMode, ShapeMatchMode,
};
pub use features::{
    corner_eigen_vals_and_vecs, corner_harris, corner_min_eigen_val, corner_sub_pix,
    get_gabor_kernel, good_features_to_track, pre_corner_detect, spatial_gradient,
};
pub use fitting::{
    box_points, closest_ellipse_points, convexity_defects, fit_ellipse, fit_line,
    intersect_convex_convex, intersect_convex_f64, min_area_rect, min_enclosing_circle, rotated_rectangle_intersection,
    ConvexityDefect, FitDistance, RotatedRect,
};
pub use draw2::{
    arrowed_line, clip_line, draw_marker, ellipse, ellipse_2_poly, fill_convex_poly, MarkerType,
};
pub use hist2::{
    accumulate, accumulate_product, accumulate_square, accumulate_weighted, apply_color_map,
    bilateral_filter, blend_linear, calc_back_project, clahe, compare_hist, get_rect_sub_pix,
    linear_polar, log_polar, sqr_box_filter, stack_blur, warp_polar, ColorMap, HistCompare,
    PolarMode,
};
pub use text::{get_text_size, put_text, FontFace};
pub use extra::{
    convert_maps, create_hanning_window, demosaicing, div_spectrums, get_font_scale_from_height,
    grab_cut, min_enclosing_triangle, phase_correlate, pyr_mean_shift_filtering,
    threshold_with_mask, BayerPattern, GrabCutClass,
};
pub use color2::{cvt_color2, ColorConversion2};
pub use color::{cvt_color, ColorConversion};
pub use filter::{
    blur, box_filter, filter_2d, gaussian_blur, get_deriv_kernels, get_gaussian_kernel, get_gaussian_kernel_fixed,
    median_blur, sep_filter_2d, sobel,
};
pub use drawing::{circle, fill_poly, line, polylines, rectangle};
pub use edge::{canny, laplacian};
pub use hist::{calc_hist, equalize_hist};
pub use misc::{
    adaptive_threshold, integral, match_template, AdaptiveMethod, TemplateMatchMode,
};
pub use morph::{
    dilate, erode, get_structuring_element, morphology_ex, MorphShape, MorphType,
};
pub use pyramids::{build_pyramid, pyr_down, pyr_up};
pub use resize::{resize, Interpolation};
pub use shape::{
    approx_poly_dp, arc_length, bounding_rect, connected_components,
    connected_components_with_stats, contour_area, contour_moments, convex_hull, moments,
    point_polygon_test, ComponentStats, LabelledComponents, Moments,
};
pub use warp::{
    get_affine_transform, get_perspective_transform, get_rotation_matrix_2d, remap, warp_affine,
    warp_perspective, Affine2x3, Perspective3x3,
};
pub use threshold::{threshold, Threshold, ThresholdMethod, ThresholdType};
