//! Detection, tracking, and camera-control vocabulary.
//!
//! Trait definitions for the detector -> tracker -> panner -> director
//! chain. Implementations live in reco-detect (detector backends) and reco-autocam
//! (trackers, panners).

pub mod detector;
pub mod director;
pub mod panner;
pub mod pipeline_event;
pub mod tracker;

/// Shared interface for types that accept detection/tracking/panning
/// configuration. Implemented by [`StitchCore`](crate::core::StitchCore),
/// [`StitchSession`](crate::session::StitchSession), and
/// [`MonoStitchCore`](crate::core::mono::MonoStitchCore), so consumers like
/// `reco_autocam::setup_autocam` can configure any of them without
/// duplication.
pub trait DetectionTarget {
    /// Attach a detector backend.
    fn set_detector(&mut self, detector: Box<dyn detector::UnifiedDetector>);
    /// Set the detection interval (run every N frames).
    fn set_detection_interval(&mut self, interval: u64);
    /// Attach a ball tracker.
    fn set_ball_tracker(&mut self, tracker: Box<dyn tracker::Tracker>);
    /// Attach a player tracker.
    fn set_player_tracker(&mut self, tracker: Box<dyn tracker::Tracker>);
    /// Attach a panner that resolves viewport pose from tracked state.
    fn set_panner(&mut self, panner: Box<dyn panner::Panner>);
    /// Input frame dimensions as `(width, height)`.
    ///
    /// Was `fn pipeline(&self) -> &StitchPipeline` before `MonoStitchCore`
    /// (which has no `StitchPipeline` - it wraps
    /// [`MonoPipeline`](crate::render::mono_pipeline::MonoPipeline)
    /// instead) needed to implement this trait too; every caller only
    /// ever used `pipeline().source_info()`, so the trait now exposes
    /// that directly instead of the concrete stereo pipeline type.
    fn source_info(&self) -> (u32, u32);
    /// Shared reference to the GPU context.
    fn gpu(&self) -> &crate::gpu::GpuContext;
}
