# Research Limitations & Honest Boundaries

**Project**: ProofCausality AI — Phase 2  
**Date**: October 2026  

---

## 1. Perceptual Boundaries

1. **Planar Monocular Projection**: ProofCausality AI currently reasons over 2D pixel coordinates and camera-frame bounding boxes. While depth separation can be bounded by scale drop and occlusion masks, full 3D metric reasoning requires calibrated stereo or depth sensors.
2. **Lighting and Shadow Artefacts**: Drastic global illumination changes (e.g., sudden strobe or room blackout) can trigger unobservability abstentions due to high Laplacian variance fluctuations.
3. **Static Speed Threshold**: The physical teleportation limit ($v_{\max} = 500\text{ px/frame}$) is currently calibrated to typical tabletop camera frame rates (30 fps). Dynamic camera motion requires IMU odometry integration.

---

## 2. Experimental Boundaries

1. **Tabletop Scope**: The primary real-world benchmark contains 12 photographed tabletop pairs. While ground-truth verified by dual annotators with CC-BY-4.0 provenance, scaling to thousands of diverse industrial scenes remains future work.
2. **Computational Overhead on Lightweight Nodes**: As documented in H02, directed graph tracking adds minor overhead on sub-millisecond mathematical operations. Its benefits scale with computationally expensive deep perception models.
