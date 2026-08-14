//! `cv::flann` for `opencv-no-c` — nearest-neighbour indices.
//!
//! # What an index buys
//!
//! Brute force is `O(n)` per query and always exact. A KD-tree is `O(log n)`
//! on low-dimensional data and still exact when searched with backtracking;
//! LSH is approximate but survives high dimensions where a KD-tree degenerates
//! to a linear scan. All three are here because which one is right depends
//! entirely on the data, and picking wrong is a silent performance cliff
//! rather than an error.
//!
//! Every index reports whether its answers are exact, so a caller can tell.

#![warn(missing_docs)]

pub mod constants;
pub mod params;

use cv_core::{cv_assert, Result};

/// Distance metric — `cv::flann::flann_distance_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distance {
    /// `cvflann::L2` — squared Euclidean. Squared, not rooted: monotone in the
    /// true distance and cheaper, which is what upstream reports too.
    #[default]
    L2,
    /// `cvflann::L1` — sum of absolute differences.
    L1,
    /// `cvflann::HAMMING` — differing bits, over `u8` features.
    Hamming,
}

impl Distance {
    /// Distance between two equal-length feature vectors.
    #[must_use]
    pub fn between(self, a: &[f32], b: &[f32]) -> f64 {
        match self {
            Distance::L2 => a
                .iter()
                .zip(b)
                .map(|(p, q)| {
                    let d = f64::from(*p) - f64::from(*q);
                    d * d
                })
                .sum(),
            Distance::L1 => a.iter().zip(b).map(|(p, q)| f64::from((*p - *q).abs())).sum(),
            Distance::Hamming => a
                .iter()
                .zip(b)
                .map(|(p, q)| f64::from((*p as u8 ^ *q as u8).count_ones()))
                .sum(),
        }
    }
}

/// One LSH table: which bit positions form the key, and the buckets.
type LshTable = (Vec<usize>, std::collections::HashMap<u64, Vec<usize>>);

/// One neighbour returned by a search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbour {
    /// Row index into the indexed feature set.
    pub index: usize,
    /// Distance under the index's metric.
    pub distance: f64,
}

/// A KD-tree node: either a split or a bucket of point indices.
enum Node {
    Leaf(Vec<usize>),
    Split {
        /// Which dimension the split is on.
        dim: usize,
        /// Value the dimension is split at.
        at: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// Which structure an [`Index`] is using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// `cv::flann::LinearIndexParams` — exhaustive, always exact.
    Linear,
    /// `cv::flann::KDTreeIndexParams` — exact when `checks` is unlimited.
    KdTree,
    /// `cv::flann::LshIndexParams` — approximate.
    Lsh,
}

/// `cv::flann::Index`.
pub struct Index {
    data: Vec<Vec<f32>>,
    dim: usize,
    metric: Distance,
    algorithm: Algorithm,
    tree: Option<Node>,
    /// LSH: one hash table per set of bit positions.
    tables: Vec<LshTable>,
}

impl std::fmt::Debug for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Index")
            .field("points", &self.data.len())
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("algorithm", &self.algorithm)
            .finish()
    }
}

impl Index {
    /// `cv::flann::Index::build(features, params, distType)`.
    ///
    /// `features` is one row per point; every row must share a length.
    pub fn build(features: &[Vec<f32>], algorithm: Algorithm, metric: Distance) -> Result<Self> {
        cv_assert!(!features.is_empty(), "an index needs at least one point");
        let dim = features[0].len();
        cv_assert!(dim > 0, "features must have at least one dimension");
        for (i, f) in features.iter().enumerate() {
            cv_assert!(f.len() == dim, "row {i} has {} values, expected {dim}", f.len());
        }
        let mut idx = Index {
            data: features.to_vec(),
            dim,
            metric,
            algorithm,
            tree: None,
            tables: Vec::new(),
        };
        match algorithm {
            Algorithm::Linear => {}
            Algorithm::KdTree => {
                let all: Vec<usize> = (0..idx.data.len()).collect();
                idx.tree = Some(idx.build_kd(all, 0));
            }
            Algorithm::Lsh => idx.build_lsh(),
        }
        Ok(idx)
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the index holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Feature dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether searches on this index return exact nearest neighbours.
    ///
    /// True for [`Algorithm::Linear`] and [`Algorithm::KdTree`] — the KD-tree
    /// search here backtracks fully rather than budgeting `checks`. False for
    /// [`Algorithm::Lsh`], which can miss.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        !matches!(self.algorithm, Algorithm::Lsh)
    }

    /// Recursively split on the widest-spread dimension at its median.
    fn build_kd(&self, idxs: Vec<usize>, depth: usize) -> Node {
        const LEAF: usize = 8;
        if idxs.len() <= LEAF || depth > 32 {
            return Node::Leaf(idxs);
        }
        // Widest spread, not round-robin: round-robin splits a flat axis and
        // wastes a level, which on skewed data compounds badly.
        let mut best = (0usize, 0.0f64);
        for d in 0..self.dim {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &i in &idxs {
                let v = f64::from(self.data[i][d]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if hi - lo > best.1 {
                best = (d, hi - lo);
            }
        }
        if best.1 <= 0.0 {
            return Node::Leaf(idxs);
        }
        let dim = best.0;
        let mut sorted = idxs.clone();
        sorted.sort_by(|&a, &b| {
            self.data[a][dim]
                .partial_cmp(&self.data[b][dim])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mid = sorted.len() / 2;
        let at = f64::from(self.data[sorted[mid]][dim]);
        let (l, r): (Vec<usize>, Vec<usize>) =
            idxs.into_iter().partition(|&i| f64::from(self.data[i][dim]) < at);
        // A degenerate split (everything on one side) would recurse forever.
        if l.is_empty() || r.is_empty() {
            return Node::Leaf(sorted);
        }
        Node::Split {
            dim,
            at,
            left: Box::new(self.build_kd(l, depth + 1)),
            right: Box::new(self.build_kd(r, depth + 1)),
        }
    }

    /// Build LSH tables by sampling random bit positions of the feature bytes.
    fn build_lsh(&mut self) {
        const TABLES: usize = 8;
        const KEY_BITS: usize = 12;
        let mut rng = cv_core::rng::Rng::new(0xF1A_1234);
        let total_bits = self.dim * 8;
        for _ in 0..TABLES {
            let bits: Vec<usize> =
                (0..KEY_BITS).map(|_| (rng.next_u32() as usize) % total_bits).collect();
            let mut table: std::collections::HashMap<u64, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, row) in self.data.iter().enumerate() {
                table.entry(lsh_key(row, &bits)).or_default().push(i);
            }
            self.tables.push((bits, table));
        }
    }

    /// `cv::flann::Index::knnSearch(query, indices, dists, knn, params)`.
    pub fn knn_search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbour>> {
        cv_assert!(
            query.len() == self.dim,
            "query has {} values, the index has {}",
            query.len(),
            self.dim
        );
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut best: Vec<Neighbour> = Vec::new();
        match self.algorithm {
            Algorithm::Linear => {
                for (i, row) in self.data.iter().enumerate() {
                    push_best(&mut best, Neighbour { index: i, distance: self.metric.between(query, row) }, k);
                }
            }
            Algorithm::KdTree => {
                if let Some(t) = &self.tree {
                    self.search_kd(t, query, k, &mut best);
                }
            }
            Algorithm::Lsh => {
                // Union of the buckets the query lands in.
                let mut seen = std::collections::HashSet::new();
                for (bits, table) in &self.tables {
                    if let Some(bucket) = table.get(&lsh_key(query, bits)) {
                        for &i in bucket {
                            if seen.insert(i) {
                                push_best(
                                    &mut best,
                                    Neighbour {
                                        index: i,
                                        distance: self.metric.between(query, &self.data[i]),
                                    },
                                    k,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(best)
    }

    /// Exact KD-tree search: descend to the query's leaf, then backtrack into
    /// any subtree whose splitting plane is closer than the current worst.
    fn search_kd(&self, node: &Node, query: &[f32], k: usize, best: &mut Vec<Neighbour>) {
        match node {
            Node::Leaf(idxs) => {
                for &i in idxs {
                    push_best(
                        best,
                        Neighbour { index: i, distance: self.metric.between(query, &self.data[i]) },
                        k,
                    );
                }
            }
            Node::Split { dim, at, left, right } => {
                let q = f64::from(query[*dim]);
                let (near, far) = if q < *at { (left, right) } else { (right, left) };
                self.search_kd(near, query, k, best);
                // Distance to the plane, in the index's own metric units.
                let plane = match self.metric {
                    Distance::L2 => (q - at) * (q - at),
                    _ => (q - at).abs(),
                };
                if best.len() < k || plane <= best.last().map_or(f64::INFINITY, |n| n.distance) {
                    self.search_kd(far, query, k, best);
                }
            }
        }
    }

    /// `cv::flann::Index::radiusSearch(query, indices, dists, radius, maxResults, params)`.
    ///
    /// `radius` is in the index's metric, so for [`Distance::L2`] it is a
    /// *squared* distance — the same convention upstream uses and the same one
    /// that trips people up.
    pub fn radius_search(&self, query: &[f32], radius: f64, max: usize) -> Result<Vec<Neighbour>> {
        cv_assert!(query.len() == self.dim, "query dimensionality must match the index");
        let mut out: Vec<Neighbour> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, row)| Neighbour { index: i, distance: self.metric.between(query, row) })
            .filter(|n| n.distance <= radius)
            .collect();
        out.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(core::cmp::Ordering::Equal).then(a.index.cmp(&b.index))
        });
        if max > 0 && out.len() > max {
            out.truncate(max);
        }
        Ok(out)
    }

    /// Batch [`Index::knn_search`] over several queries.
    pub fn knn_search_many(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<Neighbour>>> {
        queries.iter().map(|q| self.knn_search(q, k)).collect()
    }
}

/// Pack the selected bit positions of a feature row into a hash key.
fn lsh_key(row: &[f32], bits: &[usize]) -> u64 {
    let mut key = 0u64;
    for (n, &b) in bits.iter().enumerate() {
        let byte = row.get(b / 8).copied().unwrap_or(0.0) as u8;
        if byte >> (b % 8) & 1 == 1 {
            key |= 1 << n;
        }
    }
    key
}

/// Insert into a sorted top-`k` list, keeping it ascending by distance.
fn push_best(best: &mut Vec<Neighbour>, n: Neighbour, k: usize) {
    let pos = best
        .iter()
        .position(|b| n.distance < b.distance || (n.distance == b.distance && n.index < b.index))
        .unwrap_or(best.len());
    if pos >= k {
        return;
    }
    best.insert(pos, n);
    best.truncate(k);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of points, so nearest neighbours are obvious by inspection.
    fn grid() -> Vec<Vec<f32>> {
        let mut v = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                v.push(vec![x as f32, y as f32]);
            }
        }
        v
    }

    #[test]
    fn linear_finds_the_exact_nearest() {
        let idx = Index::build(&grid(), Algorithm::Linear, Distance::L2).unwrap();
        let r = idx.knn_search(&[3.1, 4.2], 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(idx.data[r[0].index], vec![3.0, 4.0]);
    }

    #[test]
    fn kdtree_agrees_with_linear_on_every_query() {
        let data = grid();
        let lin = Index::build(&data, Algorithm::Linear, Distance::L2).unwrap();
        let kd = Index::build(&data, Algorithm::KdTree, Distance::L2).unwrap();
        assert!(kd.is_exact(), "a fully-backtracking KD-tree is exact");
        for qx in 0..20 {
            for qy in 0..20 {
                let q = vec![qx as f32 * 0.5 + 0.13, qy as f32 * 0.5 - 0.27];
                let a = lin.knn_search(&q, 4).unwrap();
                let b = kd.knn_search(&q, 4).unwrap();
                let da: Vec<f64> = a.iter().map(|n| n.distance).collect();
                let db: Vec<f64> = b.iter().map(|n| n.distance).collect();
                assert_eq!(da, db, "KD-tree disagreed with brute force at {q:?}");
            }
        }
    }

    #[test]
    fn knn_results_are_sorted_and_capped() {
        let idx = Index::build(&grid(), Algorithm::KdTree, Distance::L2).unwrap();
        let r = idx.knn_search(&[5.0, 5.0], 7).unwrap();
        assert_eq!(r.len(), 7);
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance, "results must be ascending");
        }
        assert_eq!(r[0].distance, 0.0, "the query sits on a grid point");
    }

    #[test]
    fn knn_caps_at_the_index_size() {
        let idx = Index::build(&[vec![0.0f32, 0.0], vec![1.0, 1.0]], Algorithm::Linear, Distance::L2)
            .unwrap();
        assert_eq!(idx.knn_search(&[0.0, 0.0], 10).unwrap().len(), 2);
        assert!(idx.knn_search(&[0.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn l2_distances_are_squared() {
        let idx = Index::build(&[vec![0.0f32, 0.0]], Algorithm::Linear, Distance::L2).unwrap();
        let r = idx.knn_search(&[3.0, 4.0], 1).unwrap();
        assert_eq!(r[0].distance, 25.0, "L2 reports the squared distance, as upstream does");
    }

    #[test]
    fn l1_and_hamming_metrics() {
        let d = Index::build(&[vec![0.0f32, 0.0]], Algorithm::Linear, Distance::L1).unwrap();
        assert_eq!(d.knn_search(&[3.0, 4.0], 1).unwrap()[0].distance, 7.0);

        let h = Index::build(&[vec![0.0f32, 0.0]], Algorithm::Linear, Distance::Hamming).unwrap();
        // 0b1111 has four set bits, 0b1 has one.
        assert_eq!(h.knn_search(&[15.0, 1.0], 1).unwrap()[0].distance, 5.0);
    }

    #[test]
    fn radius_search_respects_the_radius_and_the_cap() {
        let idx = Index::build(&grid(), Algorithm::Linear, Distance::L2).unwrap();
        // Squared radius of 2 catches the point itself plus the four adjacent.
        let r = idx.radius_search(&[5.0, 5.0], 1.0, 0).unwrap();
        assert_eq!(r.len(), 5, "centre plus four orthogonal neighbours, got {r:?}");
        for n in &r {
            assert!(n.distance <= 1.0);
        }
        assert_eq!(idx.radius_search(&[5.0, 5.0], 1.0, 3).unwrap().len(), 3);
    }

    #[test]
    fn radius_search_can_return_nothing() {
        let idx = Index::build(&grid(), Algorithm::Linear, Distance::L2).unwrap();
        assert!(idx.radius_search(&[100.0, 100.0], 1.0, 0).unwrap().is_empty());
    }

    #[test]
    fn lsh_finds_its_own_points_and_admits_it_is_approximate() {
        // Binary-ish features, which is what LSH is for.
        let mut rng = cv_core::rng::Rng::new(7);
        let data: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..16).map(|_| rng.uniform_int(0, 256) as f32).collect())
            .collect();
        let idx = Index::build(&data, Algorithm::Lsh, Distance::Hamming).unwrap();
        assert!(!idx.is_exact(), "LSH must report itself as approximate");
        // Querying with an indexed point finds that point at distance 0.
        let mut hits = 0;
        for probe in data.iter().take(50) {
            let r = idx.knn_search(probe, 1).unwrap();
            if r.first().is_some_and(|n| n.distance == 0.0) {
                hits += 1;
            }
        }
        assert!(hits >= 45, "LSH should find most exact duplicates, got {hits}/50");
    }

    #[test]
    fn ragged_or_empty_input_is_rejected() {
        assert!(Index::build(&[], Algorithm::Linear, Distance::L2).is_err());
        let ragged = vec![vec![0.0f32, 1.0], vec![2.0]];
        assert!(Index::build(&ragged, Algorithm::Linear, Distance::L2).is_err());
        assert!(Index::build(&[vec![]], Algorithm::Linear, Distance::L2).is_err());
    }

    #[test]
    fn a_query_of_the_wrong_width_is_rejected() {
        let idx = Index::build(&grid(), Algorithm::Linear, Distance::L2).unwrap();
        assert!(idx.knn_search(&[1.0, 2.0, 3.0], 1).is_err());
        assert!(idx.radius_search(&[1.0], 1.0, 0).is_err());
    }

    #[test]
    fn kdtree_survives_duplicate_points() {
        // Every point identical: the split is degenerate and must not recurse
        // forever.
        let data: Vec<Vec<f32>> = (0..50).map(|_| vec![1.0f32, 2.0]).collect();
        let idx = Index::build(&data, Algorithm::KdTree, Distance::L2).unwrap();
        let r = idx.knn_search(&[1.0, 2.0], 3).unwrap();
        assert_eq!(r.len(), 3);
        assert!(r.iter().all(|n| n.distance == 0.0));
    }

    #[test]
    fn kdtree_handles_high_dimensional_data() {
        let mut rng = cv_core::rng::Rng::new(3);
        let data: Vec<Vec<f32>> = (0..80)
            .map(|_| (0..32).map(|_| rng.uniform_f64(-1.0, 1.0) as f32).collect())
            .collect();
        let lin = Index::build(&data, Algorithm::Linear, Distance::L2).unwrap();
        let kd = Index::build(&data, Algorithm::KdTree, Distance::L2).unwrap();
        let q: Vec<f32> = (0..32).map(|_| rng.uniform_f64(-1.0, 1.0) as f32).collect();
        let a = lin.knn_search(&q, 5).unwrap();
        let b = kd.knn_search(&q, 5).unwrap();
        assert_eq!(
            a.iter().map(|n| n.index).collect::<Vec<_>>(),
            b.iter().map(|n| n.index).collect::<Vec<_>>(),
            "the KD-tree must stay exact even where it stops being fast"
        );
    }

    #[test]
    fn batch_search_matches_one_at_a_time() {
        let idx = Index::build(&grid(), Algorithm::KdTree, Distance::L2).unwrap();
        let qs = vec![vec![1.2f32, 3.4], vec![7.7, 0.1]];
        let batch = idx.knn_search_many(&qs, 2).unwrap();
        assert_eq!(batch.len(), 2);
        for (b, q) in batch.iter().zip(qs.iter()) {
            assert_eq!(*b, idx.knn_search(q, 2).unwrap());
        }
    }
}

/// Internal tuning constants of the FLANN index builders
/// (`modules/flann/include/opencv2/flann/kdtree_index.h`, `lsh_table.h`).
///
/// These are not settings a caller passes; they are the fixed choices the
/// upstream builders were written around, and they are declared because code
/// translated from C++ refers to them by name.
pub mod tuning {
    /// `SAMPLE_MEAN` — how many points the split mean is estimated from.
    ///
    /// A randomised kd-tree splits on a high-variance dimension. Estimating
    /// that variance from a sample rather than the whole node is what keeps
    /// the build near-linear; 100 is upstream's figure.
    pub const SAMPLE_MEAN: usize = 100;

    /// `RAND_DIM` — how many of the top-variance dimensions the split is drawn
    /// from.
    ///
    /// Always splitting on the single highest-variance dimension would make
    /// every tree in the forest identical and the randomisation pointless.
    /// Picking uniformly among the top five is what decorrelates them.
    pub const RAND_DIM: usize = 5;

    /// `LshTable::SpeedLevel` — how an LSH bucket table is stored.
    ///
    /// The choice is made from the key size: a short key can index a dense
    /// array directly, a medium one needs a bitset to test occupancy before
    /// hashing, and a long one has to hash.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum SpeedLevel {
        /// `kArray` — buckets in a flat array, indexed by the key.
        #[default]
        KArray = 0,
        /// `kBitsetHash` — a bitset guards a hash map, so a miss costs one bit test.
        KBitsetHash = 1,
        /// `kHash` — a plain hash map.
        KHash = 2,
    }

    /// `CV_RESTRICT` — Rust's `&mut` already carries the no-aliasing guarantee
    /// this asks the C compiler for, so there is nothing to spell.
    pub const CV_RESTRICT: &str = "";

    /// `CV_THROW_IF_TYPE_MISMATCH(src, dst)` — check a dynamic cast.
    ///
    /// Upstream's `any` is a type-erased box that must check its tag before
    /// handing back a reference. Rust has no such box here — an `Index`
    /// knows its own element type statically — so this returns an error rather
    /// than throwing, for a caller that has erased a type itself.
    #[macro_export]
    macro_rules! cv_throw_if_type_mismatch {
        ($src:expr, $dst:expr) => {
            if $src != $dst {
                return Err(cv_core::Error::new(
                    cv_core::Code::StsBadArg,
                    format!("bad any_cast: stored {:?} but asked for {:?}", $src, $dst),
                ));
            }
        };
    }
}

impl Index {
    /// `cv::flann::Index::getAlgorithm()` — which structure was built.
    #[must_use]
    pub fn get_algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// `cv::flann::Index::getDistance()` — the metric searches use.
    #[must_use]
    pub fn get_distance(&self) -> Distance {
        self.metric
    }

    /// The points the index was built from, one row each.
    #[must_use]
    pub fn points(&self) -> &[Vec<f32>] {
        &self.data
    }

    /// `cv::flann::Index::release()` — drop the points and the structure.
    ///
    /// The index is left empty and searchable-but-fruitless rather than
    /// invalid, so a caller that releases and then searches gets no neighbours
    /// instead of a panic. The build settings survive so it can be rebuilt.
    pub fn release(&mut self) {
        // The dimension is kept: a search against a released index should
        // report no neighbours, not reject the query for having the wrong
        // width. The shape is part of the index's identity, the points are not.
        self.data.clear();
        self.tree = None;
        self.tables.clear();
    }

    /// `cv::flann::Index::save(filename)`.
    ///
    /// Writes the points and the build settings, not the tree: rebuilding on
    /// load is fast and cannot disagree with the data, whereas a serialised
    /// tree can be loaded against points it was not built from.
    pub fn save(&self, filename: &str) -> Result<()> {
        let mut fs = cv_core::FileStorage::new();
        fs.set_int("algorithm", self.get_algorithm() as i64);
        fs.set_int("distance", self.get_distance() as i64);
        fs.set_int("dim", self.dim() as i64);
        fs.set_int("count", self.len() as i64);
        let flat: Vec<f64> =
            self.points().iter().flat_map(|p| p.iter().map(|&v| f64::from(v))).collect();
        if flat.is_empty() {
            fs.set("points", cv_core::Node::Seq(Vec::new()));
        } else {
            let m = cv_core::Mat::from_vec(self.len() as i32, self.dim() as i32, 1, &flat)?;
            fs.set_mat("points", &m)?;
        }
        fs.write_file(filename)
    }

    /// `cv::flann::Index::load(features, filename)`.
    ///
    /// Reads the settings written by [`Index::save`] and rebuilds.
    pub fn load(filename: &str) -> Result<Index> {
        use cv_core::MatLike as _;
        let fs = cv_core::FileStorage::read_file(filename)?;
        let get = |k: &str| -> Result<i64> {
            fs.get(k).and_then(cv_core::Node::as_int).ok_or_else(|| {
                cv_core::Error::new(
                    cv_core::Code::StsParseError,
                    format!("{filename} has no '{k}' entry; it was not written by Index::save"),
                )
            })
        };
        let algorithm = match get("algorithm")? {
            0 => Algorithm::Linear,
            1 => Algorithm::KdTree,
            2 => Algorithm::Lsh,
            other => {
                return Err(cv_core::Error::new(
                    cv_core::Code::StsParseError,
                    format!("{filename} names algorithm {other}, which is not one of 0..=2"),
                ))
            }
        };
        let metric = match get("distance")? {
            0 => Distance::L2,
            1 => Distance::L1,
            2 => Distance::Hamming,
            other => {
                return Err(cv_core::Error::new(
                    cv_core::Code::StsParseError,
                    format!("{filename} names distance {other}, which is not one of 0..=2"),
                ))
            }
        };
        let count = get("count")? as usize;
        if count == 0 {
            // An index saved after `release`: rebuild the shell at the
            // recorded width, not the tree.
            let dim = get("dim")?.max(1) as usize;
            let mut empty = Index::build(&[vec![0.0f32; dim]], algorithm, metric)?;
            empty.release();
            return Ok(empty);
        }
        let m = fs
            .get("points")
            .and_then(cv_core::Node::as_mat)
            .ok_or_else(|| {
                cv_core::Error::new(cv_core::Code::StsParseError, format!("{filename} has no points"))
            })?;
        let dim = m.cols() as usize;
        let mut features = Vec::with_capacity(count);
        for r in 0..m.rows() {
            let mut row = Vec::with_capacity(dim);
            for c in 0..m.cols() {
                row.push(m.at_f64(r, c, 0)? as f32);
            }
            features.push(row);
        }
        Index::build(&features, algorithm, metric)
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    fn sample() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![5.0, 6.0],
        ]
    }

    #[test]
    fn the_build_settings_are_reported_back() {
        let i = Index::build(&sample(), Algorithm::KdTree, Distance::L2).unwrap();
        assert_eq!(i.get_algorithm(), Algorithm::KdTree);
        assert_eq!(i.get_distance(), Distance::L2);
        let j = Index::build(&sample(), Algorithm::Linear, Distance::L1).unwrap();
        assert_eq!(j.get_algorithm(), Algorithm::Linear);
        assert_eq!(j.get_distance(), Distance::L1);
    }

    #[test]
    fn an_index_round_trips_through_a_file() {
        let path = std::env::temp_dir().join("cv_no_c_flann_index.yml");
        let p = path.to_str().unwrap();
        let orig = Index::build(&sample(), Algorithm::KdTree, Distance::L2).unwrap();
        orig.save(p).unwrap();

        let back = Index::load(p).unwrap();
        assert_eq!(back.len(), orig.len());
        assert_eq!(back.dim(), orig.dim());
        assert_eq!(back.get_algorithm(), Algorithm::KdTree);
        assert_eq!(back.get_distance(), Distance::L2);

        // The rebuilt index answers the same query.
        let q = [4.9f32, 5.1];
        let a = orig.knn_search(&q, 2).unwrap();
        let b = back.knn_search(&q, 2).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].index, b[0].index);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn release_empties_the_index_without_breaking_it() {
        let mut i = Index::build(&sample(), Algorithm::Linear, Distance::L2).unwrap();
        assert_eq!(i.len(), 5);
        i.release();
        assert!(i.is_empty());
        assert_eq!(i.len(), 0);
        // Still usable: a search simply finds nothing.
        assert!(i.knn_search(&[0.0, 0.0], 3).unwrap().is_empty());
        // The settings survive, so it can be rebuilt in place.
        assert_eq!(i.get_algorithm(), Algorithm::Linear);
    }

    #[test]
    fn loading_a_file_that_is_not_an_index_is_rejected() {
        let path = std::env::temp_dir().join("cv_no_c_flann_bad.yml");
        let p = path.to_str().unwrap();
        let mut fs = cv_core::FileStorage::new();
        fs.set_int("something_else", 1);
        fs.write_file(p).unwrap();
        let e = Index::load(p).unwrap_err();
        assert!(e.msg.contains("algorithm"), "{}", e.msg);
        std::fs::remove_file(p).ok();
        assert!(Index::load("/nonexistent/index.yml").is_err());
    }

    #[test]
    fn the_tuning_constants_match_upstream() {
        use tuning::{SpeedLevel, RAND_DIM, SAMPLE_MEAN};
        assert_eq!(SAMPLE_MEAN, 100);
        assert_eq!(RAND_DIM, 5);
        assert_eq!(SpeedLevel::KArray as i32, 0);
        assert_eq!(SpeedLevel::KBitsetHash as i32, 1);
        assert_eq!(SpeedLevel::KHash as i32, 2);
        assert_eq!(SpeedLevel::default(), SpeedLevel::KArray);
    }
}
