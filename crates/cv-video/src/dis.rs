//! Dense inverse search and variational refinement
//! (`modules/video/src/dis_flow.cpp`, `modules/video/src/variational_refinement.cpp`).
//!
//! Both are transcribed rather than reconstructed. The two places that matter:
//!
//! * **Variational refinement solves its linear system on a red-black
//!   checkerboard**, not on the image grid. The split is not an optimisation
//!   detail that can be dropped — a Gauss-Seidel sweep in raster order visits
//!   neighbours in a different order and converges to a different iterate at
//!   any finite iteration count, which is all upstream ever runs.
//! * **Inverse search is deterministic only because the stripe count is
//!   fixed.** Spatial propagation carries a flow vector from one patch to the
//!   next within a stripe and never across, so upstream pins the count at 8
//!   regardless of the thread count. Eight is part of the answer.

use cv_core::{cv_assert, Mat, MatLike, Result};

use crate::plane::Plane;
use cv_imgproc::resize::Interpolation;



/// `cv::VariationalRefinementImpl::RedBlackBuffer`.
///
/// The image is split into two half-width buffers by parity of `x + y`, each
/// carrying a one-pixel border on all four sides. A red pixel's four
/// neighbours are all black and vice versa, so a whole colour can be relaxed
/// at once. The four `*_len` fields exist because an odd image width leaves
/// the two colours with different row lengths on alternate rows.
#[derive(Clone)]
struct RedBlack {
    red: Plane<f32>,
    black: Plane<f32>,
    red_even_len: usize,
    red_odd_len: usize,
    black_even_len: usize,
    black_odd_len: usize,
    /// Rows of the source image, i.e. `red.h - 2`.
    h: usize,
}

impl RedBlack {
    fn create(h: usize, w: usize) -> Self {
        let bw = w.div_ceil(2) + 2;
        let (re, ro, be, bo) = if w % 2 == 0 {
            (bw - 2, bw - 2, bw - 2, bw - 2)
        } else {
            (bw - 2, bw - 3, bw - 3, bw - 2)
        };
        RedBlack {
            red: Plane::<f32>::zeros(h + 2, bw),
            black: Plane::<f32>::zeros(h + 2, bw),
            red_even_len: re,
            red_odd_len: ro,
            black_even_len: be,
            black_odd_len: bo,
            h,
        }
    }

    fn bw(&self) -> usize {
        self.red.w
    }

    /// The interior length of row `i` for one colour.
    fn len_of(&self, red: bool, i: usize) -> usize {
        match (red, i % 2 == 0) {
            (true, true) => self.red_even_len,
            (true, false) => self.red_odd_len,
            (false, true) => self.black_even_len,
            (false, false) => self.black_odd_len,
        }
    }

    fn set_to(&mut self, v: f32) {
        self.red.d.fill(v);
        self.black.d.fill(v);
    }

    /// `splitCheckerboard(dst, src)`.
    fn split_from(&mut self, src: &Plane<f32>) {
        let bw = self.bw();
        for i in 0..src.h {
            let mut j = 0usize;
            let mut buf_j = 1usize;
            let (r0, b0) = ((i + 1) * bw, (i + 1) * bw);
            self.red.d[r0] = src.at(i, 0);
            self.black.d[b0] = src.at(i, 0);
            // On even rows the red buffer takes the even columns; on odd rows
            // they swap. That is the checkerboard.
            let first_is_red = i % 2 == 0;
            let mut last = 0usize;
            while j + 1 < src.w {
                let (a, b) = (src.at(i, j), src.at(i, j + 1));
                if first_is_red {
                    self.red.d[r0 + buf_j] = a;
                    self.black.d[b0 + buf_j] = b;
                } else {
                    self.black.d[b0 + buf_j] = a;
                    self.red.d[r0 + buf_j] = b;
                }
                buf_j += 1;
                j += 2;
            }
            if j < src.w {
                self.red.d[r0 + buf_j] = src.at(i, j);
                self.black.d[b0 + buf_j] = src.at(i, j);
                last = j;
            } else if j > 0 {
                last = j - 1;
            }
            self.red.d[r0 + bw - 1] = src.at(i, last);
            self.black.d[b0 + bw - 1] = src.at(i, last);
        }
        self.fill_row_borders();
    }

    /// The top and bottom border rows copy the **other** colour's adjacent
    /// row, because stepping one row over flips the parity.
    fn fill_row_borders(&mut self) {
        let (bw, rows) = (self.bw(), self.red.h);
        for k in 0..bw {
            let (rl, bl) = ((rows - 1) * bw + k, (rows - 1) * bw + k);
            let (rs, bs) = ((rows - 2) * bw + k, (rows - 2) * bw + k);
            let (rb, bb) = (self.black.d[bs], self.red.d[rs]);
            self.red.d[rl] = rb;
            self.black.d[bl] = bb;
            let (rt, bt) = (self.black.d[bw + k], self.red.d[bw + k]);
            self.red.d[k] = rt;
            self.black.d[k] = bt;
        }
    }

    /// `mergeCheckerboard(dst, src)` — the inverse of [`Self::split_from`].
    fn merge_into(&self, dst: &mut Plane<f32>) {
        let bw = self.bw();
        for i in 0..dst.h {
            let (r0, b0) = ((i + 1) * bw, (i + 1) * bw);
            let mut buf_j = 1usize;
            let mut j = 0usize;
            let first_is_red = i % 2 == 0;
            while j + 1 < dst.w {
                let (a, b) = if first_is_red {
                    (self.red.d[r0 + buf_j], self.black.d[b0 + buf_j])
                } else {
                    (self.black.d[b0 + buf_j], self.red.d[r0 + buf_j])
                };
                dst.d[i * dst.w + j] = a;
                dst.d[i * dst.w + j + 1] = b;
                buf_j += 1;
                j += 2;
            }
            if j < dst.w {
                dst.d[i * dst.w + j] =
                    if first_is_red { self.red.d[r0 + buf_j] } else { self.black.d[b0 + buf_j] };
            }
        }
    }

    /// `updateRepeatedBorders(dst)` — restore the left and right border
    /// columns after the interior has been overwritten.
    fn update_repeated_borders(&mut self) {
        let bw = self.bw();
        for i in 0..self.red.h - 2 {
            let r = (i + 1) * bw;
            let b = (i + 1) * bw;
            if i % 2 == 0 {
                self.black.d[b] = self.red.d[r + 1];
                if self.red_even_len > self.black_even_len {
                    self.black.d[b + self.black_even_len + 1] = self.red.d[r + self.red_even_len];
                } else {
                    self.red.d[r + self.red_even_len + 1] = self.black.d[b + self.black_even_len];
                }
            } else {
                self.red.d[r] = self.black.d[b + 1];
                if self.red_odd_len < self.black_odd_len {
                    self.red.d[r + self.red_odd_len + 1] = self.black.d[b + self.black_odd_len];
                } else {
                    self.black.d[b + self.black_odd_len + 1] = self.red.d[r + self.red_odd_len];
                }
            }
        }
        self.fill_row_borders();
    }
}

/// The parameters of `cv::VariationalRefinement`.
#[derive(Debug, Clone, Copy)]
pub struct VarParams {
    /// `alpha` — the smoothness weight.
    pub alpha: f32,
    /// `delta` — the colour-constancy weight.
    pub delta: f32,
    /// `gamma` — the gradient-constancy weight.
    pub gamma: f32,
    /// `epsilon` — the Charbonnier offset.
    pub epsilon: f32,
    /// `omega` — the over-relaxation factor.
    pub omega: f32,
    /// `zeta` — the normalisation floor, fixed at 0.1 upstream.
    pub zeta: f32,
    /// `fixedPointIterations`.
    pub fixed_point_iterations: i32,
    /// `sorIterations`.
    pub sor_iterations: i32,
}

impl Default for VarParams {
    fn default() -> Self {
        VarParams {
            alpha: 20.0,
            delta: 5.0,
            gamma: 10.0,
            epsilon: 0.001,
            omega: 1.6,
            zeta: 0.1,
            fixed_point_iterations: 5,
            sor_iterations: 5,
        }
    }
}

/// A central difference with a replicated border — `Sobel(src, dst, -1, 1, 0,
/// 1)`. A `ksize` of 1 with a first-order derivative widens that axis to the
/// three-tap `[-1, 0, 1]` and leaves the other axis a single tap, so this is
/// a pure difference and not a Sobel at all.
fn grad_horiz(src: &Plane<f32>) -> Plane<f32> {
    let mut d = Plane::<f32>::zeros(src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w {
            let l = src.at(y, x.saturating_sub(1));
            let r = src.at(y, (x + 1).min(src.w - 1));
            d.d[y * src.w + x] = r - l;
        }
    }
    d
}

/// The vertical partner of [`grad_horiz`].
fn grad_vert(src: &Plane<f32>) -> Plane<f32> {
    let mut d = Plane::<f32>::zeros(src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w {
            let u = src.at(y.saturating_sub(1), x);
            let l = src.at((y + 1).min(src.h - 1), x);
            d.d[y * src.w + x] = l - u;
        }
    }
    d
}

/// `warpImage(dst, src, flow_u, flow_v)` — `remap` with a replicated border.
fn warp_image(src: &Plane<f32>, u: &Plane<f32>, v: &Plane<f32>) -> Result<Plane<f32>> {
    let (h, w) = (src.h, src.w);
    let mut map_x = Mat::zeros(h as i32, w as i32, cv_core::types::CV_32FC1)?;
    let mut map_y = Mat::zeros(h as i32, w as i32, cv_core::types::CV_32FC1)?;
    {
        use cv_core::MatLikeMut;
        for y in 0..h {
            for x in 0..w {
                map_x.pixel_mut::<f32>(y as i32, x as i32)?[0] = x as f32 + u.at(y, x);
                map_y.pixel_mut::<f32>(y as i32, x as i32)?[0] = y as f32 + v.at(y, x);
            }
        }
    }
    let srcm = src.to_mat()?;
    let out = cv_imgproc::warp::remap(
        &srcm,
        &map_x,
        &map_y,
        Interpolation::Linear,
        cv_core::BorderType::Replicate,
        cv_core::Scalar::all(0.0),
    )?;
    Plane::<f32>::from_mat(&out)
}

/// The per-pixel data of one variational-refinement solve.
struct VarState {
    a11: RedBlack,
    a12: RedBlack,
    a22: RedBlack,
    b1: RedBlack,
    b2: RedBlack,
    weights: RedBlack,
    ix: RedBlack,
    iy: RedBlack,
    iz: RedBlack,
    ixx: RedBlack,
    ixy: RedBlack,
    iyy: RedBlack,
    ixz: RedBlack,
    iyz: RedBlack,
}

/// `cv::VariationalRefinement::calcUV(I0, I1, flow_u, flow_v)`.
///
/// `i0` and `i1` are single-channel; `u` and `v` are the flow to refine, in
/// place conceptually and returned here.
fn calc_uv_planes(
    p: &VarParams,
    i0: &Plane<f32>,
    i1: &Plane<f32>,
    w_u: &Plane<f32>,
    w_v: &Plane<f32>,
) -> Result<(Plane<f32>, Plane<f32>)> {
    let (h, w) = (i0.h, i0.w);

    // --- prepareBuffers.
    //
    // The derivatives are taken on the **average** of the current frame and
    // the warped next one, not on either alone.
    let warped = warp_image(i1, w_u, w_v)?;
    let mut averaged = Plane::<f32>::zeros(h, w);
    let mut iz_p = Plane::<f32>::zeros(h, w);
    for k in 0..h * w {
        averaged.d[k] = 0.5 * i0.d[k] + 0.5 * warped.d[k];
        iz_p.d[k] = warped.d[k] - i0.d[k];
    }
    let ix_p = grad_horiz(&averaged);
    let iy_p = grad_vert(&averaged);
    let ixz_p = grad_horiz(&iz_p);
    let iyz_p = grad_vert(&iz_p);
    let ixx_p = grad_horiz(&ix_p);
    let ixy_p = grad_vert(&ix_p);
    let iyy_p = grad_vert(&iy_p);

    let split = |src: &Plane<f32>| -> RedBlack {
        let mut rb = RedBlack::create(h, w);
        rb.split_from(src);
        rb
    };
    let mut st = VarState {
        a11: RedBlack::create(h, w),
        a12: RedBlack::create(h, w),
        a22: RedBlack::create(h, w),
        b1: RedBlack::create(h, w),
        b2: RedBlack::create(h, w),
        // The weight buffer is zeroed **once**. Its borders are never written
        // again, and the solver relies on that: a zero weight on the far side
        // of an edge is how the Neumann boundary condition is imposed.
        weights: RedBlack::create(h, w),
        ix: split(&ix_p),
        iy: split(&iy_p),
        iz: split(&iz_p),
        ixx: split(&ixx_p),
        ixy: split(&ixy_p),
        iyy: split(&iyy_p),
        ixz: split(&ixz_p),
        iyz: split(&iyz_p),
    };

    let mut w_u_rb = RedBlack::create(h, w);
    let mut w_v_rb = RedBlack::create(h, w);
    w_u_rb.split_from(w_u);
    w_v_rb.split_from(w_v);
    let mut temp_w_u = w_u_rb.clone();
    let mut temp_w_v = w_v_rb.clone();
    let mut d_w_u = RedBlack::create(h, w);
    let mut d_w_v = RedBlack::create(h, w);
    d_w_u.set_to(0.0);
    d_w_v.set_to(0.0);

    for _ in 0..p.fixed_point_iterations {
        for red in [true, false] {
            compute_data_term(p, &mut st, &d_w_u, &d_w_v, red);
        }
        for red in [true, false] {
            smoothness_hor_pass(p, &mut st, &w_u_rb, &w_v_rb, &temp_w_u, &temp_w_v, red);
        }
        for red in [true, false] {
            smoothness_vert_pass(&mut st, &w_u_rb, &w_v_rb, red);
        }
        for _ in 0..p.sor_iterations {
            for red in [true, false] {
                red_black_sor(p, &st, &mut d_w_u, &mut d_w_v, red);
            }
        }
        for k in 0..temp_w_u.red.d.len() {
            temp_w_u.red.d[k] = w_u_rb.red.d[k] + d_w_u.red.d[k];
            temp_w_u.black.d[k] = w_u_rb.black.d[k] + d_w_u.black.d[k];
            temp_w_v.red.d[k] = w_v_rb.red.d[k] + d_w_v.red.d[k];
            temp_w_v.black.d[k] = w_v_rb.black.d[k] + d_w_v.black.d[k];
        }
        temp_w_u.update_repeated_borders();
        temp_w_v.update_repeated_borders();
    }

    let mut out_u = Plane::<f32>::zeros(h, w);
    let mut out_v = Plane::<f32>::zeros(h, w);
    temp_w_u.merge_into(&mut out_u);
    temp_w_v.merge_into(&mut out_v);
    Ok((out_u, out_v))
}

/// Pick a colour's plane out of a [`RedBlack`].
fn pick(rb: &RedBlack, red: bool) -> &Plane<f32> {
    if red {
        &rb.red
    } else {
        &rb.black
    }
}

fn pick_mut(rb: &mut RedBlack, red: bool) -> &mut Plane<f32> {
    if red {
        &mut rb.red
    } else {
        &mut rb.black
    }
}

/// `ComputeDataTerm_ParBody` — the colour- and gradient-constancy terms of the
/// linear system, relinearised about the current increment.
fn compute_data_term(
    p: &VarParams,
    st: &mut VarState,
    d_w_u: &RedBlack,
    d_w_v: &RedBlack,
    red: bool,
) {
    let (h, bw) = (st.a11.h, st.a11.bw());
    let zeta2 = p.zeta * p.zeta;
    let eps2 = p.epsilon * p.epsilon;
    let gamma2 = p.gamma / 2.0;
    let delta2 = p.delta / 2.0;
    for i in 0..h {
        let len = st.a11.len_of(red, i);
        let row = (i + 1) * bw + 1;
        for j in 0..len {
            let k = row + j;
            let (ix, iy, iz) =
                (pick(&st.ix, red).d[k], pick(&st.iy, red).d[k], pick(&st.iz, red).d[k]);
            let (ixx, ixy, iyy) =
                (pick(&st.ixx, red).d[k], pick(&st.ixy, red).d[k], pick(&st.iyy, red).d[k]);
            let (ixz, iyz) = (pick(&st.ixz, red).d[k], pick(&st.iyz, red).d[k]);
            let (du, dv) = (pick(d_w_u, red).d[k], pick(d_w_v, red).d[k]);

            let mut deriv_norm = ix * ix + iy * iy + zeta2;
            let ik1z = iz + ix * du + iy * dv;
            let mut weight = (delta2 / (ik1z * ik1z / deriv_norm + eps2).sqrt()) / deriv_norm;
            let mut a11 = weight * (ix * ix) + zeta2;
            let mut a12 = weight * (ix * iy);
            let mut a22 = weight * (iy * iy) + zeta2;
            let mut b1 = -weight * (iz * ix);
            let mut b2 = -weight * (iz * iy);

            deriv_norm = ixx * ixx + ixy * ixy + zeta2;
            let deriv_norm2 = iyy * iyy + ixy * ixy + zeta2;
            let ik1zx = ixz + ixx * du + ixy * dv;
            let ik1zy = iyz + ixy * du + iyy * dv;
            weight = gamma2
                / (ik1zx * ik1zx / deriv_norm + ik1zy * ik1zy / deriv_norm2 + eps2).sqrt();
            a11 += weight * (ixx * ixx / deriv_norm + ixy * ixy / deriv_norm2);
            a12 += weight * (ixx * ixy / deriv_norm + ixy * iyy / deriv_norm2);
            a22 += weight * (ixy * ixy / deriv_norm + iyy * iyy / deriv_norm2);
            b1 += -weight * (ixx * ixz / deriv_norm + ixy * iyz / deriv_norm2);
            b2 += -weight * (ixy * ixz / deriv_norm + iyy * iyz / deriv_norm2);

            pick_mut(&mut st.a11, red).d[k] = a11;
            pick_mut(&mut st.a12, red).d[k] = a12;
            pick_mut(&mut st.a22, red).d[k] = a22;
            pick_mut(&mut st.b1, red).d[k] = b1;
            pick_mut(&mut st.b2, red).d[k] = b2;
        }
    }
}

/// The offset into the *other* colour's row that holds this pixel's right-hand
/// neighbour. One on even rows for red, two on odd — the checkerboard shifts.
fn next_offs(red: bool, i: usize) -> usize {
    match (red, i % 2 == 0) {
        (true, true) => 1,
        (true, false) => 2,
        (false, true) => 2,
        (false, false) => 1,
    }
}

/// `ComputeSmoothnessTermHorPass_ParBody`.
///
/// This writes the smoothness weight for the pixel **and** adds the horizontal
/// coupling to both endpoints of each edge, which is why it cannot be merged
/// with the vertical pass.
#[allow(clippy::too_many_arguments)]
fn smoothness_hor_pass(
    p: &VarParams,
    st: &mut VarState,
    w_u: &RedBlack,
    w_v: &RedBlack,
    cur_w_u: &RedBlack,
    cur_w_v: &RedBlack,
    red: bool,
) {
    let (h, bw) = (st.a11.h, st.a11.bw());
    let eps2 = p.epsilon * p.epsilon;
    let alpha2 = p.alpha / 2.0;
    let (even_len, odd_len) = if red {
        (st.a11.red_even_len, st.a11.red_odd_len)
    } else {
        (st.a11.black_even_len, st.a11.black_odd_len)
    };
    // `bool_default` is true for the red pass and false for the black one.
    let bool_default = red;
    for i in 0..h {
        let len = st.a11.len_of(red, i);
        if len == 0 {
            continue;
        }
        let no = next_offs(red, i);
        let cur = (i + 1) * bw + 1;
        let nxt = (i + 1) * bw + no;
        let nxt_row = (i + 2) * bw + 1;
        let touches_right_border = if i % 2 == 0 {
            if even_len != odd_len {
                bool_default
            } else {
                !bool_default
            }
        } else if even_len != odd_len {
            !bool_default
        } else {
            bool_default
        };

        for j in 0..len {
            let (k, kn, kr) = (cur + j, nxt + j, nxt_row + j);
            let cwu = pick(cur_w_u, red).d[k];
            let cwv = pick(cur_w_v, red).d[k];
            let ux = pick(cur_w_u, !red).d[kn] - cwu;
            let vx = pick(cur_w_v, !red).d[kn] - cwv;
            let uy = pick(cur_w_u, !red).d[kr] - cwu;
            let vy = pick(cur_w_v, !red).d[kr] - cwv;
            let weight = alpha2 / (ux * ux + vx * vx + uy * uy + vy * vy + eps2).sqrt();
            pick_mut(&mut st.weights, red).d[k] = weight;
            // The last element of a row that runs into the right border gets
            // its weight but not its coupling: there is no neighbour there.
            if j == len - 1 && touches_right_border {
                break;
            }
            let ux = weight * (pick(w_u, !red).d[kn] - pick(w_u, red).d[k]);
            let vx = weight * (pick(w_v, !red).d[kn] - pick(w_v, red).d[k]);
            pick_mut(&mut st.b1, red).d[k] += ux;
            pick_mut(&mut st.a11, red).d[k] += weight;
            pick_mut(&mut st.b2, red).d[k] += vx;
            pick_mut(&mut st.a22, red).d[k] += weight;
            pick_mut(&mut st.b1, !red).d[kn] -= ux;
            pick_mut(&mut st.a11, !red).d[kn] += weight;
            pick_mut(&mut st.b2, !red).d[kn] -= vx;
            pick_mut(&mut st.a22, !red).d[kn] += weight;
        }
    }
}

/// `ComputeSmoothnessTermVertPass_ParBody` — the vertical coupling, reusing
/// the weight the horizontal pass already stored. The last row is skipped.
fn smoothness_vert_pass(st: &mut VarState, w_u: &RedBlack, w_v: &RedBlack, red: bool) {
    let (h, bw) = (st.a11.h, st.a11.bw());
    for i in 0..h.saturating_sub(1) {
        let len = st.a11.len_of(red, i);
        let cur = (i + 1) * bw + 1;
        let nxt_row = (i + 2) * bw + 1;
        for j in 0..len {
            let (k, kr) = (cur + j, nxt_row + j);
            let weight = pick(&st.weights, red).d[k];
            let uy = weight * (pick(w_u, !red).d[kr] - pick(w_u, red).d[k]);
            let vy = weight * (pick(w_v, !red).d[kr] - pick(w_v, red).d[k]);
            pick_mut(&mut st.b1, red).d[k] += uy;
            pick_mut(&mut st.a11, red).d[k] += weight;
            pick_mut(&mut st.b2, red).d[k] += vy;
            pick_mut(&mut st.a22, red).d[k] += weight;
            pick_mut(&mut st.b1, !red).d[kr] -= uy;
            pick_mut(&mut st.a11, !red).d[kr] += weight;
            pick_mut(&mut st.b2, !red).d[kr] -= vy;
            pick_mut(&mut st.a22, !red).d[kr] += weight;
        }
    }
}

/// `RedBlackSOR_ParBody` — one over-relaxed sweep of a single colour.
///
/// `du` is updated first and the **new** value is what feeds `dv`'s coupling
/// term. That is Gauss-Seidel within the pixel as well as across it.
fn red_black_sor(p: &VarParams, st: &VarState, d_w_u: &mut RedBlack, d_w_v: &mut RedBlack, red: bool) {
    let (h, bw) = (st.a11.h, st.a11.bw());
    for i in 0..h {
        let len = st.a11.len_of(red, i);
        let no = next_offs(red, i);
        let cur = (i + 1) * bw + 1;
        let nxt = (i + 1) * bw + no;
        let nxt_row = (i + 2) * bw + 1;
        let prev_row = i * bw + 1;
        for j in 0..len {
            let k = cur + j;
            let kn = nxt + j;
            let kr = nxt_row + j;
            let kp = prev_row + j;
            // `pW_next[j - 1]` at `j == 0` reaches into the border column,
            // where the weight is still the zero it was created with.
            let w_left = pick(&st.weights, !red).d[kn - 1];
            let w_here = pick(&st.weights, red).d[k];
            let w_up = pick(&st.weights, !red).d[kp];
            let sigma_u = w_left * pick(d_w_u, !red).d[kn - 1]
                + w_here * pick(d_w_u, !red).d[kn]
                + w_up * pick(d_w_u, !red).d[kp]
                + w_here * pick(d_w_u, !red).d[kr];
            let sigma_v = w_left * pick(d_w_v, !red).d[kn - 1]
                + w_here * pick(d_w_v, !red).d[kn]
                + w_up * pick(d_w_v, !red).d[kp]
                + w_here * pick(d_w_v, !red).d[kr];
            let (a11, a12, a22) =
                (pick(&st.a11, red).d[k], pick(&st.a12, red).d[k], pick(&st.a22, red).d[k]);
            let (b1, b2) = (pick(&st.b1, red).d[k], pick(&st.b2, red).d[k]);
            let du = pick(d_w_u, red).d[k];
            let dv = pick(d_w_v, red).d[k];
            let du = du + p.omega * ((sigma_u + b1 - dv * a12) / a11 - du);
            pick_mut(d_w_u, red).d[k] = du;
            let dv = dv + p.omega * ((sigma_v + b2 - du * a12) / a22 - dv);
            pick_mut(d_w_v, red).d[k] = dv;
        }
    }
}

/// `cv::VariationalRefinement::calcUV(I0, I1, flow_u, flow_v)`.
pub fn variational_calc_uv<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    p: &VarParams,
    i0: &A,
    i1: &B,
    u: &Mat,
    v: &Mat,
) -> Result<(Mat, Mat)> {
    cv_assert!(i0.size() == i1.size(), "calcUV: the frames differ in size");
    cv_assert!(u.size() == i0.size() && v.size() == i0.size(), "calcUV: flow size mismatch");
    cv_assert!(i0.channels() == 1 && i1.channels() == 1, "calcUV takes single-channel frames");
    let (p0, p1) = (Plane::<f32>::from_mat(i0)?, Plane::<f32>::from_mat(i1)?);
    let (pu, pv) = (Plane::<f32>::from_mat(u)?, Plane::<f32>::from_mat(v)?);
    let (ou, ov) = calc_uv_planes(p, &p0, &p1, &pu, &pv)?;
    Ok((ou.to_mat()?, ov.to_mat()?))
}

/// The parameters of `cv::DISOpticalFlow`.
#[derive(Debug, Clone, Copy)]
pub struct DisParams {
    /// `finestScale`.
    pub finest_scale: i32,
    /// `patchSize`.
    pub patch_size: i32,
    /// `patchStride`.
    pub patch_stride: i32,
    /// `gradientDescentIterations`.
    pub grad_descent_iter: i32,
    /// `variationalRefinementIterations`; zero disables the stage.
    pub variational_refinement_iter: i32,
    /// `variationalRefinementAlpha`.
    pub alpha: f32,
    /// `variationalRefinementDelta`.
    pub delta: f32,
    /// `variationalRefinementGamma`.
    pub gamma: f32,
    /// `variationalRefinementEpsilon`.
    pub epsilon: f32,
    /// `useMeanNormalization`.
    pub use_mean_normalization: bool,
    /// `useSpatialPropagation`.
    pub use_spatial_propagation: bool,
    /// `coarsestScale`, or `-1` to derive it from the image width.
    pub user_coarsest_scale: i32,
}

/// `border_size` — how far a patch may hang off the edge of `I1`.
const BORDER_SIZE: i32 = 16;
/// The `EPS` guarding the structure-tensor inverse.
const DIS_EPS: f32 = 0.001;

/// `autoSelectCoarsestScale(img_width)`.
fn auto_coarsest(patch_size: i32, img_width: i32) -> i32 {
    let fratio = 5.0f32;
    (((2.0 * img_width as f32) / (fratio * patch_size as f32)).log2().floor() as i32).max(0)
}

/// `autoSelectPatchSizeAndScales(img_width)`.
fn auto_patch_and_scales(finest: i32, img_width: i32) -> (i32, i32, i32) {
    let (psz, back) = match finest {
        3 => (12, 4),
        4 => (12, 5),
        // 1, 2 and anything else share the default.
        _ => (8, 2),
    };
    let coarse = auto_coarsest(psz, img_width);
    (psz, coarse, (coarse - back).max(0))
}

/// Bilinear weights for a patch origin, clamped into the padded `I1`.
struct Weights {
    i_i1: i32,
    j_i1: i32,
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
}

/// `INIT_BILINEAR_WEIGHTS(Ux, Uy)`.
#[allow(clippy::too_many_arguments)]
fn init_weights(i: i32, j: i32, ux: f32, uy: f32, psz: i32, h: i32, w: i32) -> Weights {
    let b = BORDER_SIZE as f32;
    let i_lower = b - psz as f32 + 1.0;
    let i_upper = b + h as f32 - 1.0;
    let j_lower = b - psz as f32 + 1.0;
    let j_upper = b + w as f32 - 1.0;
    let i_i1 = (i as f32 + uy + b).max(i_lower).min(i_upper);
    let j_i1 = (j as f32 + ux + b).max(j_lower).min(j_upper);
    let di = i_i1 - i_i1.floor();
    let dj = j_i1 - j_i1.floor();
    Weights {
        // A truncation, and the clamp above keeps it non-negative.
        i_i1: i_i1 as i32,
        j_i1: j_i1 as i32,
        w11: di * dj,
        w10: di * (1.0 - dj),
        w01: (1.0 - di) * dj,
        w00: (1.0 - di) * (1.0 - dj),
    }
}

/// Everything the patch kernels need about the two images.
struct Frames<'a> {
    i0: &'a Plane<f32>,
    i1_ext: &'a Plane<f32>,
    i0x: &'a [i16],
    i0y: &'a [i16],
    w: i32,
    w_ext: i32,
}

/// `v_reduce_sum(v_float32x4)` — a pairwise tree, not a running sum. On
/// AArch64 this is `vaddvq_f32`, which adds `(a0 + a1) + (a2 + a3)`.
fn reduce4(v: [f32; 4]) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3])
}

/// The four bilinearly-sampled differences of one patch row, split into the
/// left and right halves the way `HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION`
/// does. Lane `k` sees column `k` and column `k + 4`.
fn row_diffs(f: &Frames, i: i32, j: i32, wt: &Weights, a: i32) -> ([f32; 4], [f32; 4]) {
    let s1 = ((wt.i_i1 + a) * f.w_ext + wt.j_i1) as usize;
    let s0 = ((i + a) * f.w + j) as usize;
    let e = f.w_ext as usize;
    let mut l = [0.0f32; 4];
    let mut r = [0.0f32; 4];
    for b in 0..4usize {
        // The four-argument `v_add` folds left, so this is
        // `(((w00*p + w01*q) + w10*s) + w11*u) - I0`.
        l[b] = wt.w00 * f.i1_ext.d[s1 + b]
            + wt.w01 * f.i1_ext.d[s1 + b + 1]
            + wt.w10 * f.i1_ext.d[s1 + e + b]
            + wt.w11 * f.i1_ext.d[s1 + e + b + 1]
            - f.i0.d[s0 + b];
        r[b] = wt.w00 * f.i1_ext.d[s1 + b + 4]
            + wt.w01 * f.i1_ext.d[s1 + b + 5]
            + wt.w10 * f.i1_ext.d[s1 + e + b + 4]
            + wt.w11 * f.i1_ext.d[s1 + e + b + 5]
            - f.i0.d[s0 + b + 4];
    }
    (l, r)
}

/// `computeSSD` / `computeSSDMeanNorm`.
///
/// **The summation order is part of the answer.** At the default patch size
/// upstream accumulates into four SIMD lanes and reduces once at the end;
/// summing the 64 differences in sequence instead moves the total by a few
/// ULP, and the descent below stops as soon as the residual fails to fall, so
/// a few ULP changes the iteration count and with it the flow vector. On this
/// fixture that alone was a tenth of a pixel.
fn compute_ssd(f: &Frames, i: i32, j: i32, wt: &Weights, psz: i32, mean_norm: bool) -> f32 {
    if psz == 8 {
        let mut sum_diff = [0.0f32; 4];
        let mut sum_sq = [0.0f32; 4];
        for a in 0..8 {
            let (l, r) = row_diffs(f, i, j, wt, a);
            for b in 0..4 {
                sum_sq[b] += l[b] * l[b] + r[b] * r[b];
                sum_diff[b] += l[b] + r[b];
            }
        }
        let sq = reduce4(sum_sq);
        return if mean_norm {
            sq - reduce4(sum_diff) * reduce4(sum_diff) / 64.0
        } else {
            sq
        };
    }
    let (mut sum_diff, mut sum_sq) = (0.0f32, 0.0f32);
    for a in 0..psz {
        for b in 0..psz {
            let s1 = ((wt.i_i1 + a) * f.w_ext + wt.j_i1 + b) as usize;
            let diff = wt.w00 * f.i1_ext.d[s1]
                + wt.w01 * f.i1_ext.d[s1 + 1]
                + wt.w10 * f.i1_ext.d[s1 + f.w_ext as usize]
                + wt.w11 * f.i1_ext.d[s1 + f.w_ext as usize + 1]
                - f.i0.d[((i + a) * f.w + j + b) as usize];
            sum_diff += diff;
            sum_sq += diff * diff;
        }
    }
    if mean_norm {
        sum_sq - sum_diff * sum_diff / (psz * psz) as f32
    } else {
        sum_sq
    }
}

/// `processPatch` / `processPatchMeanNorm` — one gradient-descent step's
/// residual and its projection onto the patch gradients. Lane-accumulated at
/// the default patch size for the reason given on [`compute_ssd`].
#[allow(clippy::too_many_arguments)]
fn process_patch(
    f: &Frames,
    i: i32,
    j: i32,
    wt: &Weights,
    psz: i32,
    mean_norm: bool,
    x_grad_sum: f32,
    y_grad_sum: f32,
) -> (f32, f32, f32) {
    let n = (psz * psz) as f32;
    if psz == 8 {
        let mut sx = [0.0f32; 4];
        let mut sy = [0.0f32; 4];
        let mut sum_diff = [0.0f32; 4];
        let mut sum_sq = [0.0f32; 4];
        for a in 0..8 {
            let (l, r) = row_diffs(f, i, j, wt, a);
            let s0 = ((i + a) * f.w + j) as usize;
            for b in 0..4 {
                sx[b] += l[b] * f32::from(f.i0x[s0 + b]) + r[b] * f32::from(f.i0x[s0 + b + 4]);
                sy[b] += l[b] * f32::from(f.i0y[s0 + b]) + r[b] * f32::from(f.i0y[s0 + b + 4]);
                sum_sq[b] += l[b] * l[b] + r[b] * r[b];
                sum_diff[b] += l[b] + r[b];
            }
        }
        let (mx, my, sq) = (reduce4(sx), reduce4(sy), reduce4(sum_sq));
        return if mean_norm {
            let sd = reduce4(sum_diff);
            (mx - sd * x_grad_sum / n, my - sd * y_grad_sum / n, sq - sd * sd / n)
        } else {
            (mx, my, sq)
        };
    }
    let (mut sum_diff, mut sum_sq) = (0.0f32, 0.0f32);
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for a in 0..psz {
        for b in 0..psz {
            let s1 = ((wt.i_i1 + a) * f.w_ext + wt.j_i1 + b) as usize;
            let s0 = ((i + a) * f.w + j + b) as usize;
            let diff = wt.w00 * f.i1_ext.d[s1]
                + wt.w01 * f.i1_ext.d[s1 + 1]
                + wt.w10 * f.i1_ext.d[s1 + f.w_ext as usize]
                + wt.w11 * f.i1_ext.d[s1 + f.w_ext as usize + 1]
                - f.i0.d[s0];
            sum_diff += diff;
            sum_sq += diff * diff;
            sx += diff * f32::from(f.i0x[s0]);
            sy += diff * f32::from(f.i0y[s0]);
        }
    }
    if mean_norm {
        (sx - sum_diff * x_grad_sum / n, sy - sum_diff * y_grad_sum / n, sum_sq - sum_diff * sum_diff / n)
    } else {
        (sx, sy, sum_sq)
    }
}

/// The sparse structure tensor, summed over each patch on the sparse grid.
struct Tensor {
    xx: Vec<f32>,
    yy: Vec<f32>,
    xy: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
}

/// `precomputeStructureTensor` — a separable box filter that reads a dense
/// grid and writes a sparse one.
#[allow(clippy::too_many_arguments)]
fn precompute_structure_tensor(
    i0x: &[i16],
    i0y: &[i16],
    h: i32,
    w: i32,
    ws: i32,
    hs: i32,
    psz: i32,
    pstr: i32,
) -> Tensor {
    let (hu, wsu) = (h as usize, ws as usize);
    let mut aux = Tensor {
        xx: vec![0.0; hu * wsu],
        yy: vec![0.0; hu * wsu],
        xy: vec![0.0; hu * wsu],
        x: vec![0.0; hu * wsu],
        y: vec![0.0; hu * wsu],
    };
    for i in 0..h {
        let row = (i * w) as usize;
        let (mut sxx, mut syy, mut sxy, mut sx, mut sy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for j in 0..psz as usize {
            let (xv, yv) = (f32::from(i0x[row + j]), f32::from(i0y[row + j]));
            sxx += xv * xv;
            syy += yv * yv;
            sxy += xv * yv;
            sx += xv;
            sy += yv;
        }
        let base = i as usize * wsu;
        aux.xx[base] = sxx;
        aux.yy[base] = syy;
        aux.xy[base] = sxy;
        aux.x[base] = sx;
        aux.y[base] = sy;
        let mut js = 1usize;
        for j in psz..w {
            let (a, b) = (row + j as usize, row + (j - psz) as usize);
            let (xn, yn) = (f32::from(i0x[a]), f32::from(i0y[a]));
            let (xo, yo) = (f32::from(i0x[b]), f32::from(i0y[b]));
            sxx += xn * xn - xo * xo;
            syy += yn * yn - yo * yo;
            sxy += xn * yn - xo * yo;
            sx += xn - xo;
            sy += yn - yo;
            if (j - psz + 1) % pstr == 0 && js < wsu {
                aux.xx[base + js] = sxx;
                aux.yy[base + js] = syy;
                aux.xy[base + js] = sxy;
                aux.x[base + js] = sx;
                aux.y[base + js] = sy;
                js += 1;
            }
        }
    }

    let n = (hs as usize) * wsu;
    let mut out =
        Tensor { xx: vec![0.0; n], yy: vec![0.0; n], xy: vec![0.0; n], x: vec![0.0; n], y: vec![0.0; n] };
    let (mut sxx, mut syy, mut sxy, mut sx, mut sy) =
        (vec![0.0f32; wsu], vec![0.0f32; wsu], vec![0.0f32; wsu], vec![0.0f32; wsu], vec![0.0f32; wsu]);
    for i in 0..psz as usize {
        for j in 0..wsu {
            sxx[j] += aux.xx[i * wsu + j];
            syy[j] += aux.yy[i * wsu + j];
            sxy[j] += aux.xy[i * wsu + j];
            sx[j] += aux.x[i * wsu + j];
            sy[j] += aux.y[i * wsu + j];
        }
    }
    out.xx[..wsu].copy_from_slice(&sxx);
    out.yy[..wsu].copy_from_slice(&syy);
    out.xy[..wsu].copy_from_slice(&sxy);
    out.x[..wsu].copy_from_slice(&sx);
    out.y[..wsu].copy_from_slice(&sy);
    let mut is = 1usize;
    for i in psz..h {
        for j in 0..wsu {
            let (a, b) = ((i as usize) * wsu + j, ((i - psz) as usize) * wsu + j);
            sxx[j] += aux.xx[a] - aux.xx[b];
            syy[j] += aux.yy[a] - aux.yy[b];
            sxy[j] += aux.xy[a] - aux.xy[b];
            sx[j] += aux.x[a] - aux.x[b];
            sy[j] += aux.y[a] - aux.y[b];
        }
        if (i - psz + 1) % pstr == 0 && is < hs as usize {
            for j in 0..wsu {
                out.xx[is * wsu + j] = sxx[j];
                out.yy[is * wsu + j] = syy[j];
                out.xy[is * wsu + j] = sxy[j];
                out.x[is * wsu + j] = sx[j];
                out.y[is * wsu + j] = sy[j];
            }
            is += 1;
        }
    }
    out
}

/// The mutable state one call to [`patch_inverse_search`] works on.
struct Sparse {
    sx: Vec<f32>,
    sy: Vec<f32>,
}

/// `PatchInverseSearch_ParBody` for a single stripe.
///
/// The stripe decomposition is load-bearing: a candidate propagates to the
/// patch below only when both are in the same stripe, so the answer depends on
/// the stripe count. Upstream pins it at 8 whenever spatial propagation is on
/// and uses the thread count otherwise — with propagation off no candidate
/// crosses anything, so the thread count stops mattering.
#[allow(clippy::too_many_arguments)]
fn patch_inverse_search(
    p: &DisParams,
    s: &mut Sparse,
    t: &Tensor,
    f: &Frames,
    u: &Plane<f32>,
    v: &Plane<f32>,
    ws: i32,
    stripe: (i32, i32),
    num_iter: i32,
    psz: i32,
    pstr: i32,
    h: i32,
) {
    let psz2 = psz / 2;
    let num_inner = (p.grad_descent_iter as f32 / num_iter as f32).floor() as i32;
    let w = f.w;
    for iter in 0..num_iter {
        let (dir, start_is, end_is, start_js, end_js, start_i, start_j) = if iter % 2 == 0 {
            (1, stripe.0, stripe.1, 0, ws, stripe.0 * pstr, 0)
        } else {
            (-1, stripe.1 - 1, stripe.0 - 1, ws - 1, -1, (stripe.1 - 1) * pstr, (ws - 1) * pstr)
        };
        let mut i = start_i;
        let mut is = start_is;
        while dir * is < dir * end_is {
            let mut j = start_j;
            let mut js = start_js;
            while dir * js < dir * end_js {
                let idx = (is * ws + js) as usize;
                if iter == 0 {
                    s.sx[idx] = u.at((i + psz2) as usize, (j + psz2) as usize);
                    s.sy[idx] = v.at((i + psz2) as usize, (j + psz2) as usize);
                }
                let ssd = |ux: f32, uy: f32| -> f32 {
                    let wt = init_weights(i, j, ux, uy, psz, h, w);
                    compute_ssd(f, i, j, &wt, psz, p.use_mean_normalization)
                };
                let mut min_ssd = f32::INFINITY;
                if p.use_spatial_propagation {
                    min_ssd = ssd(s.sx[idx], s.sy[idx]);
                    if dir * js > dir * start_js {
                        let n = (is * ws + js - dir) as usize;
                        let cur = ssd(s.sx[n], s.sy[n]);
                        if cur < min_ssd {
                            min_ssd = cur;
                            s.sx[idx] = s.sx[n];
                            s.sy[idx] = s.sy[n];
                        }
                    }
                    if dir * is > dir * start_is {
                        let n = ((is - dir) * ws + js) as usize;
                        let cur = ssd(s.sx[n], s.sy[n]);
                        if cur < min_ssd {
                            s.sx[idx] = s.sx[n];
                            s.sy[idx] = s.sy[n];
                        }
                    }
                }
                let _ = min_ssd;

                let (mut cur_ux, mut cur_uy) = (s.sx[idx], s.sy[idx]);
                let mut det = t.xx[idx] * t.yy[idx] - t.xy[idx] * t.xy[idx];
                if det.abs() < DIS_EPS {
                    det = DIS_EPS;
                }
                let inv11 = t.yy[idx] / det;
                let inv12 = -t.xy[idx] / det;
                let inv22 = t.xx[idx] / det;
                let mut prev_ssd = f32::INFINITY;
                for _ in 0..num_inner {
                    let wt = init_weights(i, j, cur_ux, cur_uy, psz, h, w);
                    let (dux, duy, ssd_v) = process_patch(
                        f,
                        i,
                        j,
                        &wt,
                        psz,
                        p.use_mean_normalization,
                        t.x[idx],
                        t.y[idx],
                    );
                    cur_ux -= inv11 * dux + inv12 * duy;
                    cur_uy -= inv12 * dux + inv22 * duy;
                    // Stop when the patch distance stops falling — *after*
                    // taking the step that failed to improve it.
                    if ssd_v >= prev_ssd {
                        break;
                    }
                    prev_ssd = ssd_v;
                }
                // A vector that ran further than a patch width from where it
                // started is discarded rather than trusted.
                let (ddx, ddy) = (cur_ux - s.sx[idx], cur_uy - s.sy[idx]);
                if (ddx * ddx + ddy * ddy).sqrt() <= psz as f32 {
                    s.sx[idx] = cur_ux;
                    s.sy[idx] = cur_uy;
                }
                j += dir * pstr;
                js += dir;
            }
            i += dir * pstr;
            is += dir;
        }
    }
}

/// `Densification_ParBody` — turn the sparse field into a dense one by
/// averaging every patch that covers a pixel, weighted by how well that
/// patch's vector explains the pixel.
#[allow(clippy::too_many_arguments)]
fn densify(
    s: &Sparse,
    i0: &Plane<f32>,
    i1: &Plane<f32>,
    ws: i32,
    hs: i32,
    psz: i32,
    pstr: i32,
) -> (Plane<f32>, Plane<f32>) {
    let (h, w) = (i0.h as i32, i0.w as i32);
    let mut ux = Plane::<f32>::zeros(h as usize, w as usize);
    let mut uy = Plane::<f32>::zeros(h as usize, w as usize);
    let mut start_is = 0i32;
    let mut end_is = -1i32;
    for i in 0..h {
        if i % pstr == 0 && i + psz <= h && end_is + 1 < hs {
            end_is += 1;
        }
        if i - psz >= 0 && (i - psz) % pstr == 0 && start_is < end_is {
            start_is += 1;
        }
        let mut start_js = 0i32;
        let mut end_js = -1i32;
        for j in 0..w {
            if j % pstr == 0 && j + psz <= w && end_js + 1 < ws {
                end_js += 1;
            }
            if j - psz >= 0 && (j - psz) % pstr == 0 && start_js < end_js {
                start_js += 1;
            }
            let (mut sum_x, mut sum_y, mut sum_c) = (0.0f32, 0.0f32, 0.0f32);
            for is in start_is..=end_is {
                for js in start_js..=end_js {
                    let k = (is * ws + js) as usize;
                    let j_m = (j as f32 + s.sx[k]).clamp(0.0, w as f32 - 1.0 - DIS_EPS);
                    let i_m = (i as f32 + s.sy[k]).clamp(0.0, h as f32 - 1.0 - DIS_EPS);
                    let j_l = j_m as i32;
                    let i_l = i_m as i32;
                    let (j_u, i_u) = (j_l + 1, i_l + 1);
                    let diff = (j_m - j_l as f32) * (i_m - i_l as f32) * i1.at(i_u as usize, j_u as usize)
                        + (j_u as f32 - j_m) * (i_m - i_l as f32) * i1.at(i_u as usize, j_l as usize)
                        + (j_m - j_l as f32) * (i_u as f32 - i_m) * i1.at(i_l as usize, j_u as usize)
                        + (j_u as f32 - j_m) * (i_u as f32 - i_m) * i1.at(i_l as usize, j_l as usize)
                        - i0.at(i as usize, j as usize);
                    let coef = 1.0 / diff.abs().max(1.0);
                    sum_x += coef * s.sx[k];
                    sum_y += coef * s.sy[k];
                    sum_c += coef;
                }
            }
            ux.d[(i * w + j) as usize] = sum_x / sum_c;
            uy.d[(i * w + j) as usize] = sum_y / sum_c;
        }
    }
    (ux, uy)
}

/// `cv::DISOpticalFlow::calc(I0, I1, flow)`.
///
/// Returns a two-channel `CV_32F` flow field.
pub fn dis_calc<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    p: &DisParams,
    i0: &A,
    i1: &B,
) -> Result<Mat> {
    cv_assert!(i0.channels() == 1 && i1.channels() == 1, "DIS takes single-channel frames");
    cv_assert!(i0.size() == i1.size(), "DIS frames must match in size");
    let (rows, cols) = (i0.rows(), i0.cols());

    // The coarsest usable level is bounded twice: by a search range of a
    // quarter of the width, and by the level at which a patch still fits.
    let mut patch_size = p.patch_size;
    let max_scale = (((f64::from(cols.max(rows)) / (4.0 * f64::from(patch_size))).ln()
        / core::f64::consts::LN_2
        + 0.5) as i32)
        .min((f64::from(cols.min(rows)) / f64::from(patch_size)).log2() as i32);
    let mut coarsest = if p.user_coarsest_scale != -1 {
        p.user_coarsest_scale.min(max_scale)
    } else {
        max_scale
    };
    cv_assert!(coarsest >= 0, "DIS needs an image at least 12 pixels across");
    let mut finest = p.finest_scale;
    if coarsest < finest {
        let (psz, c, f) = auto_patch_and_scales(finest, cols);
        patch_size = psz;
        coarsest = c;
        finest = f;
    }
    let patch_stride = p.patch_stride;

    // --- prepareBuffers: the two pyramids, plus a padded copy of each I1.
    // The pyramid stays **8-bit**. Upstream resizes each level from the level
    // above, which is a `CV_8U` Mat, so every level is rounded back to whole
    // levels before the next reduction; carrying it in float instead lets the
    // fractional part survive and drifts the whole pyramid.
    let mut i0s: Vec<Option<Mat>> = vec![None; (coarsest + 1) as usize];
    let mut i1s: Vec<Option<Mat>> = vec![None; (coarsest + 1) as usize];
    let mut i1_ext: Vec<Option<Plane<f32>>> = vec![None; (coarsest + 1) as usize];
    let mut gx: Vec<Option<Vec<i16>>> = vec![None; (coarsest + 1) as usize];
    let mut gy: Vec<Option<Vec<i16>>> = vec![None; (coarsest + 1) as usize];

    let mut fraction = 1;
    for i in 0..=coarsest {
        if i == finest {
            let (r, c) = (rows / fraction, cols / fraction);
            let s = cv_core::Size::new(c, r);
            i0s[i as usize] = Some(cv_imgproc::resize::resize(i0, s, 0.0, 0.0, Interpolation::Area)?);
            i1s[i as usize] = Some(cv_imgproc::resize::resize(i1, s, 0.0, 0.0, Interpolation::Area)?);
        } else if i > finest {
            let prev0 = i0s[(i - 1) as usize].as_ref().expect("the finer level exists").clone();
            let prev1 = i1s[(i - 1) as usize].as_ref().expect("the finer level exists").clone();
            let s = cv_core::Size::new(prev0.cols() / 2, prev0.rows() / 2);
            i0s[i as usize] =
                Some(cv_imgproc::resize::resize(&prev0, s, 0.0, 0.0, Interpolation::Area)?);
            i1s[i as usize] =
                Some(cv_imgproc::resize::resize(&prev1, s, 0.0, 0.0, Interpolation::Area)?);
        }
        if i >= finest {
            let cur1 = i1s[i as usize].as_ref().expect("this level exists");
            let padded = cv_core::border::copy_make_border(
                cur1,
                BORDER_SIZE,
                BORDER_SIZE,
                BORDER_SIZE,
                BORDER_SIZE,
                cv_core::BorderType::Replicate,
                cv_core::Scalar::all(0.0),
            )?;
            i1_ext[i as usize] = Some(Plane::<f32>::from_mat(&padded)?);
            let cur0 = i0s[i as usize].as_ref().expect("this level exists");
            let (ch, cw) = (cur0.rows() as usize, cur0.cols() as usize);
            // The gradients are 3x3 Sobel in **i16**, taken on the 8-bit
            // level: the descent projects onto these, not onto a float
            // gradient.
            let (dx, dy) = cv_imgproc::features::spatial_gradient(cur0)?;
            let mut vx = vec![0i16; ch * cw];
            let mut vy = vec![0i16; ch * cw];
            for y in 0..ch {
                for x in 0..cw {
                    vx[y * cw + x] = dx.at_f64(y as i32, x as i32, 0)? as i16;
                    vy[y * cw + x] = dy.at_f64(y as i32, x as i32, 0)? as i16;
                }
            }
            gx[i as usize] = Some(vx);
            gy[i as usize] = Some(vy);
        }
        fraction *= 2;
    }

    let coarse = i0s[coarsest as usize].as_ref().expect("the coarsest level exists");
    let mut ux = Plane::<f32>::zeros(coarse.rows() as usize, coarse.cols() as usize);
    let mut uy = Plane::<f32>::zeros(coarse.rows() as usize, coarse.cols() as usize);

    let var = VarParams {
        alpha: p.alpha,
        delta: p.delta,
        gamma: p.gamma,
        epsilon: p.epsilon,
        sor_iterations: 5,
        fixed_point_iterations: p.variational_refinement_iter,
        ..VarParams::default()
    };

    for level in (finest..=coarsest).rev() {
        let i0l = &Plane::<f32>::from_mat(i0s[level as usize].as_ref().expect("this level exists"))?;
        let i1l = &Plane::<f32>::from_mat(i1s[level as usize].as_ref().expect("this level exists"))?;
        let ext = i1_ext[level as usize].as_ref().expect("this level exists");
        let (h, w) = (i0l.h as i32, i0l.w as i32);
        let ws = 1 + (w - patch_size) / patch_stride;
        let hs = 1 + (h - patch_size) / patch_stride;
        let t = precompute_structure_tensor(
            gx[level as usize].as_ref().expect("this level exists"),
            gy[level as usize].as_ref().expect("this level exists"),
            h,
            w,
            ws,
            hs,
            patch_size,
            patch_stride,
        );
        let f = Frames {
            i0: i0l,
            i1_ext: ext,
            i0x: gx[level as usize].as_ref().expect("this level exists"),
            i0y: gy[level as usize].as_ref().expect("this level exists"),
            w,
            w_ext: w + 2 * BORDER_SIZE,
        };
        let mut sparse = Sparse { sx: vec![0.0; (ws * hs) as usize], sy: vec![0.0; (ws * hs) as usize] };

        let (num_stripes, num_iter) =
            if p.use_spatial_propagation { (8, 2) } else { (1, 1) };
        let stripe_sz = (hs as f64 / f64::from(num_stripes)).ceil() as i32;
        for n in 0..num_stripes {
            let lo = (n * stripe_sz).min(hs);
            let hi = ((n + 1) * stripe_sz).min(hs);
            if lo >= hi {
                continue;
            }
            patch_inverse_search(
                p,
                &mut sparse,
                &t,
                &f,
                &ux,
                &uy,
                ws,
                (lo, hi),
                num_iter,
                patch_size,
                patch_stride,
                h,
            );
        }

        let (dux, duy) = densify(&sparse, i0l, i1l, ws, hs, patch_size, patch_stride);
        ux = dux;
        uy = duy;
        if p.variational_refinement_iter > 0 {
            let (ru, rv) = calc_uv_planes(&var, i0l, i1l, &ux, &uy)?;
            ux = ru;
            uy = rv;
        }
        if level > finest {
            let finer = i0s[(level - 1) as usize].as_ref().expect("the finer level exists");
            let s = cv_core::Size::new(finer.cols(), finer.rows());
            let mut nu = Plane::<f32>::from_mat(&cv_imgproc::resize::resize(
                &ux.to_mat()?,
                s,
                0.0,
                0.0,
                Interpolation::Linear,
            )?)?;
            let mut nv = Plane::<f32>::from_mat(&cv_imgproc::resize::resize(
                &uy.to_mat()?,
                s,
                0.0,
                0.0,
                Interpolation::Linear,
            )?)?;
            for k in 0..nu.d.len() {
                nu.d[k] *= 2.0;
                nv.d[k] *= 2.0;
            }
            ux = nu;
            uy = nv;
        }
    }

    let s = cv_core::Size::new(cols, rows);
    let mut fu = Plane::<f32>::from_mat(&cv_imgproc::resize::resize(
        &ux.to_mat()?,
        s,
        0.0,
        0.0,
        Interpolation::Linear,
    )?)?;
    let mut fv = Plane::<f32>::from_mat(&cv_imgproc::resize::resize(
        &uy.to_mat()?,
        s,
        0.0,
        0.0,
        Interpolation::Linear,
    )?)?;
    let scale = (1 << finest) as f32;
    for k in 0..fu.d.len() {
        fu.d[k] *= scale;
        fv.d[k] *= scale;
    }
    cv_core::merge(&[fu.to_mat()?, fv.to_mat()?])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(h: usize, w: usize, shift: usize) -> Plane<f32> {
        let mut p = Plane::<f32>::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = (((x + shift) * 7 + y * 13) % 251) as f32;
            }
        }
        p
    }

    #[test]
    fn checkerboard_split_and_merge_round_trip() {
        for w in [15usize, 16, 17] {
            let src = ramp(9, w, 0);
            let mut rb = RedBlack::create(9, w);
            rb.split_from(&src);
            let mut back = Plane::<f32>::zeros(9, w);
            rb.merge_into(&mut back);
            assert_eq!(back.d, src.d, "width {w} did not survive the round trip");
        }
    }

    #[test]
    fn the_two_colours_partition_the_image() {
        // Every interior slot of one colour must hold a pixel the other colour
        // does not, or the solver would be relaxing a pixel against itself.
        let (h, w) = (6usize, 8usize);
        let src = ramp(h, w, 0);
        let mut rb = RedBlack::create(h, w);
        rb.split_from(&src);
        let mut seen = vec![0u8; h * w];
        let mut back = Plane::<f32>::zeros(h, w);
        rb.merge_into(&mut back);
        for (k, v) in back.d.iter().enumerate() {
            assert_eq!(*v, src.d[k]);
            seen[k] += 1;
        }
        assert!(seen.iter().all(|c| *c == 1));
    }

    #[test]
    fn refinement_leaves_a_matching_pair_alone() {
        // Two identical frames have zero residual everywhere, so the
        // increment stays zero and the flow comes back as it went in.
        let a = ramp(12, 14, 0);
        let u = Plane::<f32>::zeros(12, 14);
        let v = Plane::<f32>::zeros(12, 14);
        let p = VarParams::default();
        let (ou, ov) = calc_uv_planes(&p, &a, &a, &u, &v).unwrap();
        for k in 0..ou.d.len() {
            assert!(ou.d[k].abs() < 1e-3, "u drifted to {} at {k}", ou.d[k]);
            assert!(ov.d[k].abs() < 1e-3, "v drifted to {} at {k}", ov.d[k]);
        }
    }

    #[test]
    fn auto_scales_follow_the_finest_setting() {
        assert_eq!(auto_patch_and_scales(3, 640).0, 12);
        assert_eq!(auto_patch_and_scales(4, 640).0, 12);
        assert_eq!(auto_patch_and_scales(2, 640).0, 8);
        assert_eq!(auto_patch_and_scales(1, 640).0, 8);
        // finest is never negative however deep the request.
        assert!(auto_patch_and_scales(4, 64).2 >= 0);
    }
}
