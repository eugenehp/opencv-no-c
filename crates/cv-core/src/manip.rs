//! Array reshaping and channel manipulation
//! (`modules/core/src/matrix_operations.cpp`, `copy.cpp`, `convert.dispatch.cpp`).

use crate::arithm::set_channel;
use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::MatType;
use crate::{cv_assert, cv_error};

/// `cv::ReduceTypes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceType {
    /// `REDUCE_SUM`.
    Sum = 0,
    /// `REDUCE_AVG`.
    Avg = 1,
    /// `REDUCE_MAX`.
    Max = 2,
    /// `REDUCE_MIN`.
    Min = 3,
    /// `REDUCE_SUM2` — sum of squares.
    Sum2 = 4,
}

/// `cv::reduce(src, dst, dim, rtype, dtype)`.
///
/// `dim == 0` collapses rows into a single row; `dim == 1` collapses columns
/// into a single column. Accumulation is in `f64` regardless of depth, matching
/// upstream's use of a wider working type.
pub fn reduce<S: MatLike + ?Sized>(
    src: &S,
    dim: i32,
    rtype: ReduceType,
    dtype: Option<crate::types::Depth>,
) -> Result<Mat> {
    cv_assert!(dim == 0 || dim == 1, "reduce dim must be 0 or 1, got {dim}");
    cv_assert!(!src.is_empty(), "reduce requires a non-empty source");
    let cn = src.channels();
    let depth = dtype.unwrap_or_else(|| src.depth());
    let typ = MatType::new(depth, cn);

    let (out_rows, out_cols, n) = if dim == 0 {
        (1, src.cols(), src.rows())
    } else {
        (src.rows(), 1, src.cols())
    };
    let mut dst = Mat::zeros(out_rows, out_cols, typ)?;

    for i in 0..(if dim == 0 { src.cols() } else { src.rows() }) {
        for c in 0..cn {
            let mut acc = match rtype {
                ReduceType::Max => f64::NEG_INFINITY,
                ReduceType::Min => f64::INFINITY,
                _ => 0.0,
            };
            for k in 0..n {
                let v = if dim == 0 { src.at_f64(k, i, c)? } else { src.at_f64(i, k, c)? };
                acc = match rtype {
                    ReduceType::Sum | ReduceType::Avg => acc + v,
                    ReduceType::Sum2 => acc + v * v,
                    ReduceType::Max => acc.max(v),
                    ReduceType::Min => acc.min(v),
                };
            }
            if rtype == ReduceType::Avg {
                acc /= n as f64;
            }
            let (y, x) = if dim == 0 { (0, i) } else { (i, 0) };
            set_channel(&mut dst, y, x, c, acc)?;
        }
    }
    Ok(dst)
}

/// `cv::repeat(src, ny, nx, dst)` — tile `src` into an `ny x nx` grid.
pub fn repeat<S: MatLike + ?Sized>(src: &S, ny: i32, nx: i32) -> Result<Mat> {
    cv_assert!(ny > 0 && nx > 0, "repeat counts must be positive, got {ny}x{nx}");
    let (h, w) = (src.rows(), src.cols());
    let esz = src.elem_size();
    let mut dst = Mat::zeros(h * ny, w * nx, src.mat_type())?;
    for ty in 0..ny {
        for y in 0..h {
            let srow = src.row_bytes(y)?.to_vec();
            let drow = dst.row_bytes_mut(ty * h + y)?;
            for tx in 0..nx {
                let off = (tx * w) as usize * esz;
                drow[off..off + srow.len()].copy_from_slice(&srow);
            }
        }
    }
    Ok(dst)
}

/// `cv::hconcat(src, dst)` — concatenate left to right.
pub fn hconcat(parts: &[Mat]) -> Result<Mat> {
    cv_assert!(!parts.is_empty(), "hconcat requires at least one matrix");
    let first = &parts[0];
    let mut total_cols = 0;
    for p in parts {
        if p.rows() != first.rows() || p.mat_type() != first.mat_type() {
            return cv_error!(
                Code::StsUnmatchedSizes,
                "hconcat parts must share row count and type"
            );
        }
        total_cols += p.cols();
    }
    let esz = first.elem_size();
    let mut dst = Mat::zeros(first.rows(), total_cols, first.mat_type())?;
    for y in 0..first.rows() {
        let mut off = 0usize;
        // Collect first so `dst` is not borrowed while reading the parts.
        let rows: Vec<Vec<u8>> = parts.iter().map(|p| p.row_bytes(y).map(<[u8]>::to_vec)).collect::<Result<_>>()?;
        let drow = dst.row_bytes_mut(y)?;
        for r in rows {
            drow[off..off + r.len()].copy_from_slice(&r);
            off += r.len();
        }
        debug_assert_eq!(off, total_cols as usize * esz);
    }
    Ok(dst)
}

/// `cv::vconcat(src, dst)` — concatenate top to bottom.
pub fn vconcat(parts: &[Mat]) -> Result<Mat> {
    cv_assert!(!parts.is_empty(), "vconcat requires at least one matrix");
    let first = &parts[0];
    let mut total_rows = 0;
    for p in parts {
        if p.cols() != first.cols() || p.mat_type() != first.mat_type() {
            return cv_error!(
                Code::StsUnmatchedSizes,
                "vconcat parts must share column count and type"
            );
        }
        total_rows += p.rows();
    }
    let mut dst = Mat::zeros(total_rows, first.cols(), first.mat_type())?;
    let mut y0 = 0;
    for p in parts {
        for y in 0..p.rows() {
            let r = p.row_bytes(y)?.to_vec();
            dst.row_bytes_mut(y0 + y)?.copy_from_slice(&r);
        }
        y0 += p.rows();
    }
    Ok(dst)
}

/// `cv::extractChannel(src, dst, coi)`.
pub fn extract_channel<S: MatLike + ?Sized>(src: &S, coi: i32) -> Result<Mat> {
    cv_assert!(
        coi >= 0 && coi < src.channels(),
        "channel {coi} out of range 0..{}",
        src.channels()
    );
    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type().with_channels(1))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let v = src.at_f64(y, x, coi)?;
            set_channel(&mut dst, y, x, 0, v)?;
        }
    }
    Ok(dst)
}

/// `cv::insertChannel(src, dst, coi)` — write a single-channel plane into `dst`.
pub fn insert_channel<S: MatLike + ?Sized>(src: &S, dst: &mut Mat, coi: i32) -> Result<()> {
    cv_assert!(src.channels() == 1, "insertChannel source must be single-channel");
    cv_assert!(
        coi >= 0 && coi < dst.channels(),
        "channel {coi} out of range 0..{}",
        dst.channels()
    );
    if src.size() != dst.size() {
        return cv_error!(Code::StsUnmatchedSizes, "insertChannel sizes must match");
    }
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let v = src.at_f64(y, x, 0)?;
            set_channel(dst, y, x, coi, v)?;
        }
    }
    Ok(())
}

/// `cv::mixChannels(src, dst, fromTo)` for a single source and destination.
///
/// `from_to` is a flat list of `(src_channel, dst_channel)` pairs.
pub fn mix_channels<S: MatLike + ?Sized>(
    src: &S,
    dst: &mut Mat,
    from_to: &[(i32, i32)],
) -> Result<()> {
    if src.size() != dst.size() {
        return cv_error!(Code::StsUnmatchedSizes, "mixChannels sizes must match");
    }
    for &(from, to) in from_to {
        cv_assert!(
            from >= 0 && from < src.channels(),
            "source channel {from} out of range"
        );
        cv_assert!(to >= 0 && to < dst.channels(), "destination channel {to} out of range");
    }
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for &(from, to) in from_to {
                let v = src.at_f64(y, x, from)?;
                set_channel(dst, y, x, to, v)?;
            }
        }
    }
    Ok(())
}

/// `cv::SortFlags`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortFlags {
    /// `SORT_EVERY_ROW` when true, `SORT_EVERY_COLUMN` when false.
    pub every_row: bool,
    /// `SORT_DESCENDING` when true, `SORT_ASCENDING` when false.
    pub descending: bool,
}

/// `cv::sort(src, dst, flags)` — single-channel only, as upstream.
pub fn sort<S: MatLike + ?Sized>(src: &S, flags: SortFlags) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "sort requires a single-channel matrix");
    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    let (outer, inner) =
        if flags.every_row { (src.rows(), src.cols()) } else { (src.cols(), src.rows()) };

    for i in 0..outer {
        let mut vals: Vec<f64> = Vec::with_capacity(inner as usize);
        for k in 0..inner {
            vals.push(if flags.every_row { src.at_f64(i, k, 0)? } else { src.at_f64(k, i, 0)? });
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        if flags.descending {
            vals.reverse();
        }
        for (k, v) in vals.iter().enumerate() {
            let (y, x) = if flags.every_row { (i, k as i32) } else { (k as i32, i) };
            set_channel(&mut dst, y, x, 0, *v)?;
        }
    }
    Ok(dst)
}

/// `cv::sortIdx(src, dst, flags)` — the permutation that would sort each
/// row or column, as `CV_32S`.
pub fn sort_idx<S: MatLike + ?Sized>(src: &S, flags: SortFlags) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "sortIdx requires a single-channel matrix");
    let mut dst = Mat::zeros(src.rows(), src.cols(), crate::types::CV_32SC1)?;
    let (outer, inner) =
        if flags.every_row { (src.rows(), src.cols()) } else { (src.cols(), src.rows()) };

    for i in 0..outer {
        let mut idx: Vec<i32> = (0..inner).collect();
        let mut vals: Vec<f64> = Vec::with_capacity(inner as usize);
        for k in 0..inner {
            vals.push(if flags.every_row { src.at_f64(i, k, 0)? } else { src.at_f64(k, i, 0)? });
        }
        idx.sort_by(|&a, &b| {
            vals[a as usize]
                .partial_cmp(&vals[b as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        if flags.descending {
            idx.reverse();
        }
        for (k, &v) in idx.iter().enumerate() {
            let (y, x) = if flags.every_row { (i, k as i32) } else { (k as i32, i) };
            dst.pixel_mut::<i32>(y, x)?[0] = v;
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Depth, CV_8UC1, CV_8UC3};

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    #[test]
    fn reduce_type_values_match_opencv() {
        assert_eq!(ReduceType::Sum as i32, 0);
        assert_eq!(ReduceType::Avg as i32, 1);
        assert_eq!(ReduceType::Max as i32, 2);
        assert_eq!(ReduceType::Min as i32, 3);
        assert_eq!(ReduceType::Sum2 as i32, 4);
    }

    #[test]
    fn reduce_dim0_collapses_rows() {
        let m = u8m(3, 2, &[1, 2, 3, 4, 5, 6]);
        let s = reduce(&m, 0, ReduceType::Sum, Some(Depth::F32)).unwrap();
        assert_eq!(s.size(), crate::Size::new(2, 1));
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![9.0, 12.0]);
    }

    #[test]
    fn reduce_dim1_collapses_columns() {
        let m = u8m(3, 2, &[1, 2, 3, 4, 5, 6]);
        let s = reduce(&m, 1, ReduceType::Sum, Some(Depth::F32)).unwrap();
        assert_eq!(s.size(), crate::Size::new(1, 3));
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![3.0, 7.0, 11.0]);
    }

    /// `cv2.reduce(m, dim, cv2.REDUCE_SUM2)` on the same matrix.
    #[test]
    fn reduce_sum2_accumulates_squares() {
        let m = u8m(2, 3, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            reduce(&m, 0, ReduceType::Sum2, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![17.0, 29.0, 45.0]
        );
        assert_eq!(
            reduce(&m, 1, ReduceType::Sum2, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![14.0, 77.0]
        );
    }

    #[test]
    fn reduce_avg_min_max() {
        let m = u8m(3, 2, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            reduce(&m, 0, ReduceType::Avg, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![3.0, 4.0]
        );
        assert_eq!(
            reduce(&m, 0, ReduceType::Max, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![5.0, 6.0]
        );
        assert_eq!(
            reduce(&m, 0, ReduceType::Min, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn reduce_sum_can_exceed_the_source_depth() {
        // Summing 8-bit values into 8-bit saturates; into f32 it does not.
        let m = u8m(3, 1, &[200, 200, 200]);
        assert_eq!(reduce(&m, 0, ReduceType::Sum, None).unwrap().to_vec::<u8>().unwrap(), vec![255]);
        assert_eq!(
            reduce(&m, 0, ReduceType::Sum, Some(Depth::F32)).unwrap().to_vec::<f32>().unwrap(),
            vec![600.0]
        );
    }

    #[test]
    fn reduce_is_multichannel_aware() {
        let m = Mat::from_vec(2, 1, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        let s = reduce(&m, 0, ReduceType::Sum, Some(Depth::F32)).unwrap();
        assert_eq!(s.channels(), 3);
        assert_eq!(s.pixel::<f32>(0, 0).unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn repeat_tiles_the_source() {
        let m = u8m(1, 2, &[1, 2]);
        let r = repeat(&m, 2, 3).unwrap();
        assert_eq!(r.size(), crate::Size::new(6, 2));
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    }

    #[test]
    fn repeat_once_is_a_copy() {
        let m = u8m(2, 2, &[1, 2, 3, 4]);
        assert_eq!(repeat(&m, 1, 1).unwrap().to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn repeat_preserves_channels() {
        let m = Mat::from_vec(1, 1, 3, &[7u8, 8, 9]).unwrap();
        let r = repeat(&m, 2, 2).unwrap();
        assert_eq!(r.pixel::<u8>(1, 1).unwrap(), &[7, 8, 9]);
    }

    #[test]
    fn hconcat_joins_left_to_right() {
        let a = u8m(2, 2, &[1, 2, 3, 4]);
        let b = u8m(2, 1, &[5, 6]);
        let c = hconcat(&[a, b]).unwrap();
        assert_eq!(c.size(), crate::Size::new(3, 2));
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![1, 2, 5, 3, 4, 6]);
    }

    #[test]
    fn vconcat_joins_top_to_bottom() {
        let a = u8m(1, 3, &[1, 2, 3]);
        let b = u8m(2, 3, &[4, 5, 6, 7, 8, 9]);
        let c = vconcat(&[a, b]).unwrap();
        assert_eq!(c.size(), crate::Size::new(3, 3));
        assert_eq!(c.to_vec::<u8>().unwrap(), (1u8..10).collect::<Vec<_>>());
    }

    #[test]
    fn concat_rejects_mismatched_parts() {
        let a = u8m(2, 2, &[0; 4]);
        let b = u8m(3, 2, &[0; 6]);
        assert_eq!(hconcat(&[a.clone(), b.clone()]).unwrap_err().code, Code::StsUnmatchedSizes);
        let c = u8m(2, 3, &[0; 6]);
        assert_eq!(vconcat(&[a, c]).unwrap_err().code, Code::StsUnmatchedSizes);
    }

    #[test]
    fn extract_and_insert_channel_roundtrip() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        let g = extract_channel(&m, 1).unwrap();
        assert_eq!(g.channels(), 1);
        assert_eq!(g.to_vec::<u8>().unwrap(), vec![2, 5]);

        let mut dst = Mat::zeros(1, 2, CV_8UC3).unwrap();
        insert_channel(&g, &mut dst, 2).unwrap();
        assert_eq!(dst.pixel::<u8>(0, 0).unwrap(), &[0, 0, 2]);
        assert_eq!(dst.pixel::<u8>(0, 1).unwrap(), &[0, 0, 5]);
    }

    #[test]
    fn extract_channel_rejects_a_bad_index() {
        let m = Mat::zeros(1, 1, CV_8UC3).unwrap();
        assert!(extract_channel(&m, 3).is_err());
        assert!(extract_channel(&m, -1).is_err());
    }

    #[test]
    fn mix_channels_swaps_red_and_blue() {
        let src = Mat::from_vec(1, 1, 3, &[1u8, 2, 3]).unwrap();
        let mut dst = Mat::zeros(1, 1, CV_8UC3).unwrap();
        mix_channels(&src, &mut dst, &[(0, 2), (1, 1), (2, 0)]).unwrap();
        assert_eq!(dst.pixel::<u8>(0, 0).unwrap(), &[3, 2, 1]);
    }

    #[test]
    fn sort_orders_each_row() {
        let m = u8m(2, 3, &[3, 1, 2, 9, 7, 8]);
        let s = sort(&m, SortFlags { every_row: true, descending: false }).unwrap();
        assert_eq!(s.to_vec::<u8>().unwrap(), vec![1, 2, 3, 7, 8, 9]);

        let d = sort(&m, SortFlags { every_row: true, descending: true }).unwrap();
        assert_eq!(d.to_vec::<u8>().unwrap(), vec![3, 2, 1, 9, 8, 7]);
    }

    #[test]
    fn sort_orders_each_column() {
        let m = u8m(3, 2, &[5, 6, 1, 4, 3, 2]);
        let s = sort(&m, SortFlags { every_row: false, descending: false }).unwrap();
        assert_eq!(s.to_vec::<u8>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn sort_idx_returns_the_permutation() {
        let m = u8m(1, 4, &[30, 10, 40, 20]);
        let i = sort_idx(&m, SortFlags { every_row: true, descending: false }).unwrap();
        assert_eq!(i.mat_type(), crate::types::CV_32SC1);
        assert_eq!(i.to_vec::<i32>().unwrap(), vec![1, 3, 0, 2]);
    }

    #[test]
    fn sort_rejects_multichannel() {
        let m = Mat::zeros(2, 2, CV_8UC3).unwrap();
        assert!(sort(&m, SortFlags { every_row: true, descending: false }).is_err());
    }

    #[test]
    fn reduce_accepts_a_strided_view() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let v = m.roi(crate::Rect::new(1, 1, 2, 2)).unwrap();
        assert_eq!(v.mat_type(), CV_8UC1);
        let s = reduce(&v, 0, ReduceType::Sum, Some(Depth::F32)).unwrap();
        // Columns of the view are (6,11) and (7,12).
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![17.0, 19.0]);
    }
}
