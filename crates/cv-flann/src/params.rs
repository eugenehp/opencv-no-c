//! FLANN's index, distance and parameter enumerations
//! (`modules/flann/include/opencv2/flann/defines.h`, `miniflann.hpp`).
//!
//! # What these actually select
//!
//! FLANN is *approximate* nearest neighbours: every index but
//! [`IndexType::Linear`] trades exactness for speed, and these enumerations are
//! how a caller picks a point on that trade. The choice is not free — an LSH
//! index over float descriptors, or a k-d tree over 128 dimensions, will run
//! and return neighbours that are not the nearest ones. The doc on each
//! variant says what it is for.

/// `cv::flann::flann_algorithm_t` — which index structure to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexType {
    /// `cv::flann::FLANN_INDEX_LINEAR` / `cv::flann::LINEAR` — exhaustive
    /// search. The only exact one, and the right choice below a few thousand
    /// points.
    #[default]
    Linear,
    /// `cv::flann::FLANN_INDEX_KDTREE` / `cv::flann::KDTREE` — randomised k-d
    /// forest. Good to roughly 100 dimensions; beyond that the trees stop
    /// pruning anything and it degenerates to linear search with overhead.
    KdTree,
    /// `cv::flann::FLANN_INDEX_KMEANS` / `cv::flann::KMEANS` — hierarchical
    /// k-means. Handles higher dimensions than the k-d forest, builds slower.
    KMeans,
    /// `cv::flann::FLANN_INDEX_COMPOSITE` / `cv::flann::COMPOSITE` — both of
    /// the above, searched together.
    Composite,
    /// `cv::flann::FLANN_INDEX_KDTREE_SINGLE` — one exact k-d tree, for low
    /// dimensions.
    KdTreeSingle,
    /// `cv::flann::FLANN_INDEX_HIERARCHICAL`
    Hierarchical,
    /// `cv::flann::FLANN_INDEX_LSH` — locality-sensitive hashing. For **binary**
    /// descriptors under a Hamming distance; meaningless on floats.
    Lsh,
    /// `cv::flann::FLANN_INDEX_SAVED` / `cv::flann::SAVED` — load a built index.
    Saved,
    /// `cv::flann::FLANN_INDEX_AUTOTUNED` / `cv::flann::AUTOTUNED` — measure and
    /// choose, given a target precision.
    Autotuned,
}

/// `cv::flann::flann_distance_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceType {
    /// `cv::flann::FLANN_DIST_EUCLIDEAN` / `cv::flann::FLANN_DIST_L2` /
    /// `cv::flann::EUCLIDEAN` — the default.
    #[default]
    Euclidean,
    /// `cv::flann::FLANN_DIST_MANHATTAN` / `cv::flann::FLANN_DIST_L1` /
    /// `cv::flann::MANHATTAN`
    Manhattan,
    /// `cv::flann::FLANN_DIST_MINKOWSKI` — needs an order parameter.
    Minkowski,
    /// `cv::flann::FLANN_DIST_MAX` — Chebyshev, the largest coordinate gap.
    Max,
    /// `cv::flann::FLANN_DIST_HIST_INTERSECT` — for histograms.
    HistIntersect,
    /// `cv::flann::FLANN_DIST_HELLINGER` — for histograms; the Euclidean
    /// distance between element-wise square roots.
    Hellinger,
    /// `cv::flann::FLANN_DIST_CHI_SQUARE` / `cv::flann::FLANN_DIST_CS`
    ChiSquare,
    /// `cv::flann::FLANN_DIST_KULLBACK_LEIBLER` / `cv::flann::FLANN_DIST_KL` —
    /// asymmetric, so it is not a metric.
    KullbackLeibler,
    /// `cv::flann::FLANN_DIST_HAMMING` — differing bits, for binary descriptors.
    Hamming,
    /// `cv::flann::FLANN_DIST_DNAMMING` — Hamming over 2-bit DNA codes.
    DnaHamming,
}

impl DistanceType {
    /// Whether this distance obeys the triangle inequality.
    ///
    /// The tree indices assume it; using one that does not is how a search
    /// silently starts missing neighbours it has already pruned away.
    #[must_use]
    pub fn is_metric(self) -> bool {
        !matches!(
            self,
            DistanceType::KullbackLeibler | DistanceType::HistIntersect | DistanceType::ChiSquare
        )
    }

    /// Whether this distance is defined on binary data rather than on numbers.
    #[must_use]
    pub fn is_binary(self) -> bool {
        matches!(self, DistanceType::Hamming | DistanceType::DnaHamming)
    }

    /// Distance between two float vectors, where that is meaningful.
    ///
    /// Returns `None` for the binary distances, which have no float reading.
    #[must_use]
    pub fn distance(self, a: &[f32], b: &[f32]) -> Option<f64> {
        if self.is_binary() || a.len() != b.len() {
            return None;
        }
        let z = |x: &f32, y: &f32| (f64::from(*x), f64::from(*y));
        Some(match self {
            DistanceType::Euclidean => {
                a.iter().zip(b).map(|(x, y)| { let (x, y) = z(x, y); (x - y) * (x - y) }).sum::<f64>()
            }
            DistanceType::Manhattan => {
                a.iter().zip(b).map(|(x, y)| { let (x, y) = z(x, y); (x - y).abs() }).sum()
            }
            DistanceType::Max => a
                .iter()
                .zip(b)
                .map(|(x, y)| { let (x, y) = z(x, y); (x - y).abs() })
                .fold(0.0f64, f64::max),
            DistanceType::Minkowski => {
                // Order 3 unless a caller says otherwise; upstream carries the
                // order in the index parameters rather than the enum.
                a.iter()
                    .zip(b)
                    .map(|(x, y)| { let (x, y) = z(x, y); (x - y).abs().powi(3) })
                    .sum::<f64>()
            }
            DistanceType::HistIntersect => {
                a.iter().zip(b).map(|(x, y)| { let (x, y) = z(x, y); x.min(y) }).sum()
            }
            DistanceType::Hellinger => a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let (x, y) = z(x, y);
                    let d = x.max(0.0).sqrt() - y.max(0.0).sqrt();
                    d * d
                })
                .sum(),
            DistanceType::ChiSquare => a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let (x, y) = z(x, y);
                    let s = x + y;
                    if s.abs() < 1e-12 { 0.0 } else { (x - y) * (x - y) / s }
                })
                .sum(),
            DistanceType::KullbackLeibler => a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let (x, y) = z(x, y);
                    if x > 0.0 && y > 0.0 { x * (x / y).ln() } else { 0.0 }
                })
                .sum(),
            DistanceType::Hamming | DistanceType::DnaHamming => unreachable!("filtered above"),
        })
    }
}

/// `cv::flann::flann_centers_init_t` — how k-means picks its starting centres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CentersInit {
    /// `cv::flann::FLANN_CENTERS_RANDOM` / `cv::flann::CENTERS_RANDOM`
    #[default]
    Random,
    /// `cv::flann::FLANN_CENTERS_GONZALES` / `cv::flann::CENTERS_GONZALES` —
    /// each new centre is the point furthest from those already chosen.
    Gonzales,
    /// `cv::flann::FLANN_CENTERS_KMEANSPP` / `cv::flann::CENTERS_KMEANSPP` —
    /// k-means++, which spreads centres probabilistically. Usually the best.
    KMeansPp,
    /// `cv::flann::FLANN_CENTERS_GROUPWISE`
    Groupwise,
}

/// `cv::flann::flann_datatype_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlannDataType {
    /// `cv::flann::FLANN_INT8` / `cv::flann::FLANN_INDEX_TYPE_8S`
    Int8,
    /// `cv::flann::FLANN_INT16` / `cv::flann::FLANN_INDEX_TYPE_16S`
    Int16,
    /// `cv::flann::FLANN_INT32` / `cv::flann::FLANN_INDEX_TYPE_32S`
    Int32,
    /// `cv::flann::FLANN_INT64` / `cv::flann::FLANN_INDEX_TYPE_64S`
    Int64,
    /// `cv::flann::FLANN_UINT8` / `cv::flann::FLANN_INDEX_TYPE_8U`
    Uint8,
    /// `cv::flann::FLANN_UINT16` / `cv::flann::FLANN_INDEX_TYPE_16U`
    Uint16,
    /// `cv::flann::FLANN_UINT32` / `cv::flann::FLANN_INDEX_TYPE_32U`
    Uint32,
    /// `cv::flann::FLANN_UINT64` / `cv::flann::FLANN_INDEX_TYPE_64U`
    Uint64,
    /// `cv::flann::FLANN_FLOAT32` / `cv::flann::FLANN_INDEX_TYPE_32F`
    Float32,
    /// `cv::flann::FLANN_FLOAT64` / `cv::flann::FLANN_INDEX_TYPE_64F`
    Float64,
    /// `cv::flann::FLANN_INDEX_TYPE_STRING`
    String,
    /// `cv::flann::FLANN_INDEX_TYPE_BOOL`
    Bool,
    /// `cv::flann::FLANN_INDEX_TYPE_ALGORITHM`
    Algorithm,
}

/// `cv::flann::SearchParams` — how hard a query tries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// Leaves to visit. Higher is more accurate and slower.
    pub checks: i32,
    /// Search radius, for the radius queries.
    pub eps: f32,
    /// Sort the results by distance.
    pub sorted: bool,
}

impl SearchParams {
    /// `cv::flann::FLANN_CHECKS_UNLIMITED` — visit every leaf, making the
    /// search exact at the cost of the index's whole point.
    pub const CHECKS_UNLIMITED: i32 = -1;
    /// `cv::flann::FLANN_CHECKS_AUTOTUNED`
    pub const CHECKS_AUTOTUNED: i32 = -2;
}

impl Default for SearchParams {
    /// Upstream's defaults: 32 checks, no epsilon, sorted.
    fn default() -> Self {
        Self { checks: 32, eps: 0.0, sorted: true }
    }
}

/// `cv::flann::KDTreeIndexParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdTreeIndexParams {
    /// Parallel randomised trees. More trees means better recall, linearly
    /// more memory.
    pub trees: i32,
}

impl Default for KdTreeIndexParams {
    /// Upstream's default is 4 trees.
    fn default() -> Self {
        Self { trees: 4 }
    }
}

/// `cv::flann::KMeansIndexParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansIndexParams {
    /// Children per node.
    pub branching: i32,
    /// k-means iterations per split; -1 runs to convergence.
    pub iterations: i32,
    /// How the starting centres are chosen.
    pub centers_init: CentersInit,
    /// Leaf-size cutoff, as a fraction of the branching factor.
    pub cb_index: f32,
}

impl Default for KMeansIndexParams {
    /// Upstream's defaults: branching 32, 11 iterations, random centres.
    fn default() -> Self {
        Self { branching: 32, iterations: 11, centers_init: CentersInit::Random, cb_index: 0.2 }
    }
}

/// `cv::flann::LshIndexParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LshIndexParams {
    /// Hash tables. More tables raises recall and memory together.
    pub table_number: i32,
    /// Bits per hash key.
    pub key_size: i32,
    /// Bits flipped when probing neighbouring buckets; 0 disables multi-probe.
    pub multi_probe_level: i32,
}

impl Default for LshIndexParams {
    /// A common starting point: 12 tables, 20-bit keys, 2-bit probing.
    fn default() -> Self {
        Self { table_number: 12, key_size: 20, multi_probe_level: 2 }
    }
}

/// `cv::flann::AutotunedIndexParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutotunedIndexParams {
    /// Recall to aim for, in `0..=1`.
    pub target_precision: f32,
    /// How much build time is worth trading for query time.
    pub build_weight: f32,
    /// How much memory is worth trading for speed.
    pub memory_weight: f32,
    /// Fraction of the data used while measuring.
    pub sample_fraction: f32,
}

impl Default for AutotunedIndexParams {
    /// Upstream's defaults: 80% precision, build weight 0.01.
    fn default() -> Self {
        Self {
            target_precision: 0.8,
            build_weight: 0.01,
            memory_weight: 0.0,
            sample_fraction: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_match_upstream() {
        assert_eq!(IndexType::default(), IndexType::Linear);
        assert_eq!(DistanceType::default(), DistanceType::Euclidean);
        assert_eq!(CentersInit::default(), CentersInit::Random);
        assert_eq!(SearchParams::default().checks, 32);
        assert!(SearchParams::default().sorted);
        assert_eq!(KdTreeIndexParams::default().trees, 4);
        let k = KMeansIndexParams::default();
        assert_eq!((k.branching, k.iterations), (32, 11));
        assert_eq!(AutotunedIndexParams::default().target_precision, 0.8);
        assert_eq!(SearchParams::CHECKS_UNLIMITED, -1);
        assert_eq!(SearchParams::CHECKS_AUTOTUNED, -2);
    }

    #[test]
    fn the_metric_and_binary_flags_are_right() {
        // The tree indices prune on the triangle inequality, so it matters
        // which of these obey it.
        for d in [
            DistanceType::Euclidean,
            DistanceType::Manhattan,
            DistanceType::Max,
            DistanceType::Hellinger,
        ] {
            assert!(d.is_metric(), "{d:?} should be a metric");
        }
        for d in [
            DistanceType::KullbackLeibler,
            DistanceType::HistIntersect,
            DistanceType::ChiSquare,
        ] {
            assert!(!d.is_metric(), "{d:?} is not a metric");
        }
        assert!(DistanceType::Hamming.is_binary());
        assert!(DistanceType::DnaHamming.is_binary());
        assert!(!DistanceType::Euclidean.is_binary());
    }

    #[test]
    fn the_float_distances_compute_what_they_claim() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];
        // Euclidean is reported squared, as FLANN does internally.
        assert_eq!(DistanceType::Euclidean.distance(&a, &b), Some(25.0));
        assert_eq!(DistanceType::Manhattan.distance(&a, &b), Some(7.0));
        assert_eq!(DistanceType::Max.distance(&a, &b), Some(4.0));
        // Every distance is zero between a vector and itself.
        for d in [
            DistanceType::Euclidean,
            DistanceType::Manhattan,
            DistanceType::Max,
            DistanceType::Minkowski,
            DistanceType::Hellinger,
            DistanceType::ChiSquare,
        ] {
            assert_eq!(d.distance(&a, &a), Some(0.0), "{d:?} on identical input");
        }
    }

    #[test]
    fn a_binary_distance_has_no_float_reading() {
        let a = [1.0f32, 2.0];
        assert_eq!(DistanceType::Hamming.distance(&a, &a), None);
        assert_eq!(DistanceType::DnaHamming.distance(&a, &a), None);
        // And mismatched lengths are refused rather than truncated.
        assert_eq!(DistanceType::Euclidean.distance(&a, &[1.0]), None);
    }

    #[test]
    fn kullback_leibler_is_asymmetric() {
        // Which is exactly why it is not a metric — worth pinning, because a
        // caller who assumes symmetry gets subtly wrong neighbours.
        // Not a reversed pair: `[0.5, 0.3, 0.2]` against `[0.2, 0.3, 0.5]`
        // gives the *same* divergence both ways, because the terms swap
        // exactly. A genuinely asymmetric pair is needed to show the property.
        let p = [0.7f32, 0.2, 0.1];
        let q = [0.1f32, 0.3, 0.6];
        let a = DistanceType::KullbackLeibler.distance(&p, &q).unwrap();
        let b = DistanceType::KullbackLeibler.distance(&q, &p).unwrap();
        assert!(a > 0.0 && b > 0.0);
        assert!((a - b).abs() > 1e-9, "KL should differ by direction: {a} vs {b}");
        assert!(!DistanceType::KullbackLeibler.is_metric());
    }

    #[test]
    fn hellinger_is_the_euclidean_distance_of_square_roots() {
        let a = [4.0f32, 9.0];
        let b = [1.0f32, 4.0];
        // (2-1)^2 + (3-2)^2 = 2.
        assert_eq!(DistanceType::Hellinger.distance(&a, &b), Some(2.0));
    }
}
