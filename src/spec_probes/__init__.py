"""Phase 6 frozen spectrum-encoder physical-recovery probes.

A bounded review instrument (architecture.md decision 12), and the
encoder-quality half of the spectra gate: benchmarks how well frozen
spectrum encoders' embeddings support recovery of spec-z (spectral class
recovery is gated behind `labels.spectral_class.enabled`, currently always
false -- see spec_probes.spectra_data's module docstring), before any
encoder is used in a retrieval pipeline. This package has zero import
coupling to aion_reimp's training/retrieval/model modules; it reuses only
aion_reimp.utils, aion_reimp.artifacts, and the generic split/fingerprint
helpers in aion_reimp.manifest.
"""

__version__ = "0.1.0"
