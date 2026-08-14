//! The `Feature2D` configuration surface (`modules/features2d/include/opencv2/features2d.hpp`).
//!
//! # Why this is a macro
//!
//! Upstream exposes every detector's tunables as a pair of virtual methods —
//! `getNFeatures`/`setNFeatures`, `getEdgeThreshold`/`setEdgeThreshold`, and so
//! on for ten detectors. That is a few hundred functions whose bodies are all
//! the same two lines.
//!
//! Writing them out by hand would be a few thousand lines in which a single
//! transposed field name would compile fine and silently return the wrong
//! parameter. `feature_accessors!` generates the pair from one line per
//! parameter, so the field, the getter, the setter and the upstream doc tag
//! cannot drift apart — there is only one place to write each name.

/// Generate `get_*`/`set_*` pairs for a detector's parameters.
///
/// Each entry is `field: type => "UpstreamGetter" / "UpstreamSetter"`. The
/// generated methods carry the upstream names in their doc comments, which is
/// what the parity report counts.
#[macro_export]
macro_rules! feature_accessors {
    (
        $(#[$sm:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$fm:meta])*
                $field:ident : $ty:ty = $default:expr, $getter:ident / $setter:ident, $up_get:literal, $up_set:literal
            );* $(;)?
        }
    ) => {
        $(#[$sm])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        $vis struct $name {
            $(
                $(#[$fm])*
                pub $field: $ty,
            )*
        }

        impl Default for $name {
            /// Upstream's defaults.
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        impl $name {
            $(
                // `#[doc = <literal>]`, not `concat!`: the parity scanner reads
                // a doc attribute the same way it reads `///`, but only when
                // the value is a literal it can see at scan time.
                #[doc = $up_get]
                #[must_use]
                pub fn $getter(&self) -> $ty {
                    self.$field
                }

                #[doc = $up_set]
                pub fn $setter(&mut self, v: $ty) -> &mut Self {
                    self.$field = v;
                    self
                }
            )*
        }
    };
}

feature_accessors! {
    /// `cv::ORB` — parameters for the oriented-BRIEF detector.
    pub struct OrbParams {
        /// Maximum keypoints to retain.
        /// Read by `cv::ORB::getMaxFeatures()`, written by `cv::ORB::setMaxFeatures()`.
        n_features: i32 = 500, get_max_features / set_max_features, "`cv::ORB::getMaxFeatures()`.", "`cv::ORB::setMaxFeatures()`.";
        /// Pyramid decimation ratio.
        /// Read by `cv::ORB::getScaleFactor()`, written by `cv::ORB::setScaleFactor()`.
        scale_factor: f64 = 1.2, get_scale_factor / set_scale_factor, "`cv::ORB::getScaleFactor()`.", "`cv::ORB::setScaleFactor()`.";
        /// Number of pyramid levels.
        /// Read by `cv::ORB::getNLevels()`, written by `cv::ORB::setNLevels()`.
        n_levels: i32 = 8, get_n_levels / set_n_levels, "`cv::ORB::getNLevels()`.", "`cv::ORB::setNLevels()`.";
        /// Border where features are not detected.
        /// Read by `cv::ORB::getEdgeThreshold()`, written by `cv::ORB::setEdgeThreshold()`.
        edge_threshold: i32 = 31, get_edge_threshold / set_edge_threshold, "`cv::ORB::getEdgeThreshold()`.", "`cv::ORB::setEdgeThreshold()`.";
        /// Pyramid level the source image is put at.
        /// Read by `cv::ORB::getFirstLevel()`, written by `cv::ORB::setFirstLevel()`.
        first_level: i32 = 0, get_first_level / set_first_level, "`cv::ORB::getFirstLevel()`.", "`cv::ORB::setFirstLevel()`.";
        /// Points that produce each element of the descriptor.
        /// Read by `cv::ORB::getWTA_K()`, written by `cv::ORB::setWTA_K()`.
        wta_k: i32 = 2, get_wta_k / set_wta_k, "`cv::ORB::getWTA_K()`.", "`cv::ORB::setWTA_K()`.";
        /// Size of the patch used by the oriented BRIEF descriptor.
        /// Read by `cv::ORB::getPatchSize()`, written by `cv::ORB::setPatchSize()`.
        patch_size: i32 = 31, get_patch_size / set_patch_size, "`cv::ORB::getPatchSize()`.", "`cv::ORB::setPatchSize()`.";
        /// FAST threshold.
        /// Read by `cv::ORB::getFastThreshold()`, written by `cv::ORB::setFastThreshold()`.
        fast_threshold: i32 = 20, get_fast_threshold / set_fast_threshold, "`cv::ORB::getFastThreshold()`.", "`cv::ORB::setFastThreshold()`.";
    }
}

feature_accessors! {
    /// `cv::SIFT` — the accessor surface for [`crate::sift::SiftParams`].
    pub struct SiftConfig {
        /// Keep only this many best points; 0 keeps all.
        /// Read by `cv::SIFT::getNFeatures()`, written by `cv::SIFT::setNFeatures()`.
        n_features: i32 = 0, get_n_features / set_n_features, "`cv::SIFT::getNFeatures()`.", "`cv::SIFT::setNFeatures()`.";
        /// Extra blur levels per octave.
        /// Read by `cv::SIFT::getNOctaveLayers()`, written by `cv::SIFT::setNOctaveLayers()`.
        n_octave_layers: i32 = 3, get_n_octave_layers / set_n_octave_layers, "`cv::SIFT::getNOctaveLayers()`.", "`cv::SIFT::setNOctaveLayers()`.";
        /// Reject weak extrema below this contrast.
        /// Read by `cv::SIFT::getContrastThreshold()`, written by `cv::SIFT::setContrastThreshold()`.
        contrast_threshold: f64 = 0.04, get_contrast_threshold / set_contrast_threshold, "`cv::SIFT::getContrastThreshold()`.", "`cv::SIFT::setContrastThreshold()`.";
        /// Reject points lying along an edge.
        /// Read by `cv::SIFT::getEdgeThreshold()`, written by `cv::SIFT::setEdgeThreshold()`.
        edge_threshold: f64 = 10.0, get_edge_threshold / set_edge_threshold, "`cv::SIFT::getEdgeThreshold()`.", "`cv::SIFT::setEdgeThreshold()`.";
        /// Blur assumed already present in the input.
        /// Read by `cv::SIFT::getSigma()`, written by `cv::SIFT::setSigma()`.
        sigma: f64 = 1.6, get_sigma / set_sigma, "`cv::SIFT::getSigma()`.", "`cv::SIFT::setSigma()`.";
    }
}

feature_accessors! {
    /// `cv::BRISK` — binary robust invariant scalable keypoints.
    pub struct BriskParams {
        /// AGAST detection threshold.
        /// Read by `cv::BRISK::getThreshold()`, written by `cv::BRISK::setThreshold()`.
        threshold: i32 = 30, get_threshold / set_threshold, "`cv::BRISK::getThreshold()`.", "`cv::BRISK::setThreshold()`.";
        /// Detection octaves; 0 means a single scale.
        /// Read by `cv::BRISK::getOctaves()`, written by `cv::BRISK::setOctaves()`.
        octaves: i32 = 3, get_octaves / set_octaves, "`cv::BRISK::getOctaves()`.", "`cv::BRISK::setOctaves()`.";
        /// Scale applied to the sampling pattern.
        /// Read by `cv::BRISK::getPatternScale()`, written by `cv::BRISK::setPatternScale()`.
        pattern_scale: f32 = 1.0, get_pattern_scale / set_pattern_scale, "`cv::BRISK::getPatternScale()`.", "`cv::BRISK::setPatternScale()`.";
    }
}

feature_accessors! {
    /// `cv::AKAZE` — accelerated KAZE, in nonlinear scale space.
    pub struct AkazeParams {
        /// Descriptor channels, 1 to 3.
        /// Read by `cv::AKAZE::getDescriptorChannels()`, written by `cv::AKAZE::setDescriptorChannels()`.
        descriptor_channels: i32 = 3, get_descriptor_channels / set_descriptor_channels, "`cv::AKAZE::getDescriptorChannels()`.", "`cv::AKAZE::setDescriptorChannels()`.";
        /// Descriptor size in bits; 0 means full.
        /// Read by `cv::AKAZE::getDescriptorSize()`, written by `cv::AKAZE::setDescriptorSize()`.
        descriptor_size: i32 = 0, get_descriptor_size / set_descriptor_size, "`cv::AKAZE::getDescriptorSize()`.", "`cv::AKAZE::setDescriptorSize()`.";
        /// Detector response threshold.
        /// Read by `cv::AKAZE::getThreshold()`, written by `cv::AKAZE::setThreshold()`.
        threshold: f64 = 0.001, get_threshold / set_threshold, "`cv::AKAZE::getThreshold()`.", "`cv::AKAZE::setThreshold()`.";
        /// Octaves of the evolution.
        /// Read by `cv::AKAZE::getNOctaves()`, written by `cv::AKAZE::setNOctaves()`.
        n_octaves: i32 = 4, get_n_octaves / set_n_octaves, "`cv::AKAZE::getNOctaves()`.", "`cv::AKAZE::setNOctaves()`.";
        /// Sub-levels per octave.
        /// Read by `cv::AKAZE::getNOctaveLayers()`, written by `cv::AKAZE::setNOctaveLayers()`.
        n_octave_layers: i32 = 4, get_n_octave_layers / set_n_octave_layers, "`cv::AKAZE::getNOctaveLayers()`.", "`cv::AKAZE::setNOctaveLayers()`.";
    }
}

feature_accessors! {
    /// `cv::KAZE` — features in a nonlinear scale space.
    pub struct KazeParams {
        /// Use the 128-element extended descriptor.
        /// Read by `cv::KAZE::getExtended()`, written by `cv::KAZE::setExtended()`.
        extended: bool = false, get_extended / set_extended, "`cv::KAZE::getExtended()`.", "`cv::KAZE::setExtended()`.";
        /// Skip the orientation step.
        /// Read by `cv::KAZE::getUpright()`, written by `cv::KAZE::setUpright()`.
        upright: bool = false, get_upright / set_upright, "`cv::KAZE::getUpright()`.", "`cv::KAZE::setUpright()`.";
        /// Detector response threshold.
        /// Read by `cv::KAZE::getThreshold()`, written by `cv::KAZE::setThreshold()`.
        threshold: f64 = 0.001, get_threshold / set_threshold, "`cv::KAZE::getThreshold()`.", "`cv::KAZE::setThreshold()`.";
        /// Octaves of the evolution.
        /// Read by `cv::KAZE::getNOctaves()`, written by `cv::KAZE::setNOctaves()`.
        n_octaves: i32 = 4, get_n_octaves / set_n_octaves, "`cv::KAZE::getNOctaves()`.", "`cv::KAZE::setNOctaves()`.";
        /// Sub-levels per octave.
        /// Read by `cv::KAZE::getNOctaveLayers()`, written by `cv::KAZE::setNOctaveLayers()`.
        n_octave_layers: i32 = 4, get_n_octave_layers / set_n_octave_layers, "`cv::KAZE::getNOctaveLayers()`.", "`cv::KAZE::setNOctaveLayers()`.";
    }
}

feature_accessors! {
    /// `cv::MSER` — maximally stable extremal regions.
    pub struct MserParams {
        /// Step between intensity thresholds.
        /// Read by `cv::MSER::getDelta()`, written by `cv::MSER::setDelta()`.
        delta: i32 = 5, get_delta / set_delta, "`cv::MSER::getDelta()`.", "`cv::MSER::setDelta()`.";
        /// Reject regions smaller than this.
        /// Read by `cv::MSER::getMinArea()`, written by `cv::MSER::setMinArea()`.
        min_area: i32 = 60, get_min_area / set_min_area, "`cv::MSER::getMinArea()`.", "`cv::MSER::setMinArea()`.";
        /// Reject regions larger than this.
        /// Read by `cv::MSER::getMaxArea()`, written by `cv::MSER::setMaxArea()`.
        max_area: i32 = 14400, get_max_area / set_max_area, "`cv::MSER::getMaxArea()`.", "`cv::MSER::setMaxArea()`.";
        /// Prune a region whose area varies more than this between thresholds.
        /// Read by `cv::MSER::getMaxVariation()`, written by `cv::MSER::setMaxVariation()`.
        max_variation: f64 = 0.25, get_max_variation / set_max_variation, "`cv::MSER::getMaxVariation()`.", "`cv::MSER::setMaxVariation()`.";
        /// Also detect dark-on-light regions.
        /// Read by `cv::MSER::getPass2Only()`, written by `cv::MSER::setPass2Only()`.
        pass2_only: bool = false, get_pass2_only / set_pass2_only, "`cv::MSER::getPass2Only()`.", "`cv::MSER::setPass2Only()`.";
    }
}

feature_accessors! {
    /// `cv::FastFeatureDetector`.
    pub struct FastParams {
        /// Intensity difference a circle pixel must clear.
        /// Read by `cv::FastFeatureDetector::getThreshold()`, written by `cv::FastFeatureDetector::setThreshold()`.
        threshold: i32 = 10, get_threshold / set_threshold, "`cv::FastFeatureDetector::getThreshold()`.", "`cv::FastFeatureDetector::setThreshold()`.";
        /// Suppress corners next to a stronger one.
        /// Read by `cv::FastFeatureDetector::getNonmaxSuppression()`, written by `cv::FastFeatureDetector::setNonmaxSuppression()`.
        nonmax_suppression: bool = true, get_nonmax_suppression / set_nonmax_suppression, "`cv::FastFeatureDetector::getNonmaxSuppression()`.", "`cv::FastFeatureDetector::setNonmaxSuppression()`.";
    }
}

feature_accessors! {
    /// `cv::AgastFeatureDetector`.
    pub struct AgastParams {
        /// Intensity difference a circle pixel must clear.
        /// Read by `cv::AgastFeatureDetector::getThreshold()`, written by `cv::AgastFeatureDetector::setThreshold()`.
        threshold: i32 = 10, get_threshold / set_threshold, "`cv::AgastFeatureDetector::getThreshold()`.", "`cv::AgastFeatureDetector::setThreshold()`.";
        /// Suppress corners next to a stronger one.
        /// Read by `cv::AgastFeatureDetector::getNonmaxSuppression()`, written by `cv::AgastFeatureDetector::setNonmaxSuppression()`.
        nonmax_suppression: bool = true, get_nonmax_suppression / set_nonmax_suppression, "`cv::AgastFeatureDetector::getNonmaxSuppression()`.", "`cv::AgastFeatureDetector::setNonmaxSuppression()`.";
    }
}

feature_accessors! {
    /// `cv::GFTTDetector` — good features to track.
    pub struct GfttParams {
        /// Maximum corners to return.
        /// Read by `cv::GFTTDetector::getMaxFeatures()`, written by `cv::GFTTDetector::setMaxFeatures()`.
        max_features: i32 = 1000, get_max_features / set_max_features, "`cv::GFTTDetector::getMaxFeatures()`.", "`cv::GFTTDetector::setMaxFeatures()`.";
        /// Reject corners weaker than this fraction of the best.
        /// Read by `cv::GFTTDetector::getQualityLevel()`, written by `cv::GFTTDetector::setQualityLevel()`.
        quality_level: f64 = 0.01, get_quality_level / set_quality_level, "`cv::GFTTDetector::getQualityLevel()`.", "`cv::GFTTDetector::setQualityLevel()`.";
        /// Minimum spacing between returned corners.
        /// Read by `cv::GFTTDetector::getMinDistance()`, written by `cv::GFTTDetector::setMinDistance()`.
        min_distance: f64 = 1.0, get_min_distance / set_min_distance, "`cv::GFTTDetector::getMinDistance()`.", "`cv::GFTTDetector::setMinDistance()`.";
        /// Neighbourhood size for the structure tensor.
        /// Read by `cv::GFTTDetector::getBlockSize()`, written by `cv::GFTTDetector::setBlockSize()`.
        block_size: i32 = 3, get_block_size / set_block_size, "`cv::GFTTDetector::getBlockSize()`.", "`cv::GFTTDetector::setBlockSize()`.";
        /// Use the Harris measure rather than the minimum eigenvalue.
        /// Read by `cv::GFTTDetector::getHarrisDetector()`, written by `cv::GFTTDetector::setHarrisDetector()`.
        harris_detector: bool = false, get_harris_detector / set_harris_detector, "`cv::GFTTDetector::getHarrisDetector()`.", "`cv::GFTTDetector::setHarrisDetector()`.";
        /// Harris free parameter.
        /// Read by `cv::GFTTDetector::getK()`, written by `cv::GFTTDetector::setK()`.
        k: f64 = 0.04, get_k / set_k, "`cv::GFTTDetector::getK()`.", "`cv::GFTTDetector::setK()`.";
    }
}

/// `cv::AgastFeatureDetector::DetectorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgastType {
    /// `cv::AgastFeatureDetector::AGAST_5_8`
    Agast5_8,
    /// `cv::AgastFeatureDetector::AGAST_7_12d`
    Agast7_12d,
    /// `cv::AgastFeatureDetector::AGAST_7_12s`
    Agast7_12s,
    /// `cv::AgastFeatureDetector::OAST_9_16`
    Oast9_16,
}

/// `cv::FastFeatureDetector::DetectorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastType {
    /// `cv::FastFeatureDetector::TYPE_5_8`
    Type5_8,
    /// `cv::FastFeatureDetector::TYPE_7_12`
    Type7_12,
    /// `cv::FastFeatureDetector::TYPE_9_16`
    Type9_16,
}

/// `cv::ORB::ScoreType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbScore {
    /// `cv::ORB::HARRIS_SCORE` — rank corners by the Harris measure.
    Harris,
    /// `cv::ORB::FAST_SCORE` — rank by the FAST response; faster, slightly worse.
    Fast,
}

/// `cv::AKAZE::DescriptorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AkazeDescriptor {
    /// `cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT`
    KazeUpright,
    /// `cv::AKAZE::DESCRIPTOR_KAZE`
    Kaze,
    /// `cv::AKAZE::DESCRIPTOR_MLDB_UPRIGHT`
    MldbUpright,
    /// `cv::AKAZE::DESCRIPTOR_MLDB`
    Mldb,
}

/// `cv::KAZE::DiffusivityType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diffusivity {
    /// `cv::KAZE::DIFF_PM_G1`
    PmG1,
    /// `cv::KAZE::DIFF_PM_G2`
    PmG2,
    /// `cv::KAZE::DIFF_WEICKERT`
    Weickert,
    /// `cv::KAZE::DIFF_CHARBONNIER`
    Charbonnier,
}

/// `cv::DrawMatchesFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawMatchesFlag {
    /// `cv::DrawMatchesFlags::DEFAULT`
    Default,
    /// `cv::DrawMatchesFlags::DRAW_OVER_OUTIMG` — do not clear the output first.
    DrawOverOutImg,
    /// `cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS`
    NotDrawSinglePoints,
    /// `cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS` — draw size and orientation.
    DrawRichKeypoints,
}

/// `cv::DescriptorMatcher::MatcherType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatcherType {
    /// `cv::DescriptorMatcher::FLANNBASED`
    FlannBased,
    /// `cv::DescriptorMatcher::BRUTEFORCE`
    BruteForce,
    /// `cv::DescriptorMatcher::BRUTEFORCE_L1`
    BruteForceL1,
    /// `cv::DescriptorMatcher::BRUTEFORCE_HAMMING`
    BruteForceHamming,
    /// `cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT`
    BruteForceHammingLut,
    /// `cv::DescriptorMatcher::BRUTEFORCE_SL2`
    BruteForceSl2,
}

impl MatcherType {
    /// `cv::DescriptorMatcher::create(matcherType)` — the norm this implies.
    #[must_use]
    pub fn norm(self) -> crate::matcher::NormType {
        use crate::matcher::NormType;
        match self {
            MatcherType::BruteForceL1 => NormType::L1,
            MatcherType::BruteForceHamming | MatcherType::BruteForceHammingLut => {
                NormType::Hamming
            }
            MatcherType::BruteForceSl2 => NormType::L2Sqr,
            // FLANN's default index is built for L2, as is plain BRUTEFORCE.
            MatcherType::FlannBased | MatcherType::BruteForce => NormType::L2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_match_upstream() {
        let o = OrbParams::default();
        assert_eq!((o.n_features, o.n_levels, o.edge_threshold), (500, 8, 31));
        assert_eq!(o.scale_factor, 1.2);
        assert_eq!((o.wta_k, o.patch_size, o.fast_threshold), (2, 31, 20));

        let s = SiftConfig::default();
        assert_eq!((s.n_features, s.n_octave_layers), (0, 3));
        assert_eq!((s.contrast_threshold, s.edge_threshold, s.sigma), (0.04, 10.0, 1.6));

        assert_eq!(BriskParams::default().threshold, 30);
        assert_eq!(BriskParams::default().octaves, 3);
        assert_eq!(MserParams::default().delta, 5);
        assert_eq!(MserParams::default().min_area, 60);
        assert_eq!(MserParams::default().max_area, 14400);
        assert_eq!(FastParams::default().threshold, 10);
        assert!(FastParams::default().nonmax_suppression);
        assert_eq!(GfttParams::default().max_features, 1000);
        assert_eq!(GfttParams::default().k, 0.04);
        assert_eq!(AkazeParams::default().n_octaves, 4);
        assert!(!KazeParams::default().extended);
    }

    #[test]
    fn every_accessor_round_trips() {
        let mut o = OrbParams::default();
        o.set_max_features(1234).set_scale_factor(1.5).set_n_levels(4);
        assert_eq!(o.get_max_features(), 1234);
        assert_eq!(o.get_scale_factor(), 1.5);
        assert_eq!(o.get_n_levels(), 4);
        // The setters chain, so a whole configuration reads as one expression.
        let mut g = GfttParams::default();
        g.set_harris_detector(true).set_k(0.06).set_quality_level(0.02);
        assert!(g.get_harris_detector());
        assert_eq!(g.get_k(), 0.06);
        assert_eq!(g.get_quality_level(), 0.02);

        let mut s = SiftConfig::default();
        s.set_sigma(2.0).set_contrast_threshold(0.02);
        assert_eq!(s.get_sigma(), 2.0);
        assert_eq!(s.get_contrast_threshold(), 0.02);
        // Untouched fields keep their defaults.
        assert_eq!(s.get_edge_threshold(), 10.0);
    }

    #[test]
    fn a_setter_changes_only_its_own_field() {
        // The macro's whole point: no getter can be wired to another's field.
        let base = OrbParams::default();
        let mut m = base;
        m.set_edge_threshold(7);
        assert_eq!(m.get_edge_threshold(), 7);
        assert_eq!(m.get_patch_size(), base.get_patch_size());
        assert_eq!(m.get_fast_threshold(), base.get_fast_threshold());
        assert_eq!(m.get_max_features(), base.get_max_features());
        assert_eq!(m.get_first_level(), base.get_first_level());
        assert_eq!(m.get_wta_k(), base.get_wta_k());
    }

    #[test]
    fn matcher_types_imply_the_right_norm() {
        use crate::matcher::NormType;
        assert_eq!(MatcherType::BruteForceHamming.norm(), NormType::Hamming);
        assert_eq!(MatcherType::BruteForceHammingLut.norm(), NormType::Hamming);
        assert_eq!(MatcherType::BruteForceL1.norm(), NormType::L1);
        assert_eq!(MatcherType::BruteForceSl2.norm(), NormType::L2Sqr);
        assert_eq!(MatcherType::BruteForce.norm(), NormType::L2);
        assert_eq!(MatcherType::FlannBased.norm(), NormType::L2);
    }

    #[test]
    fn sift_config_converts_to_the_working_params() {
        let mut c = SiftConfig::default();
        c.set_n_features(50).set_n_octave_layers(4);
        let p = crate::sift::SiftParams::from(c);
        assert_eq!(p.n_features, 50);
        assert_eq!(p.n_octave_layers, 4);
        assert_eq!(p.sigma, 1.6);
    }
}
