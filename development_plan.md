# Development stages
## Pipe-cleaner e2e

### Goals

- Functional MVP
- Can train, gradients run e2e
- Single person only data
- Reasonable RPS
- Window size training only (don't carry long scenes across batches)

### Non-Goals

- High quality performance
- Training optimizations

### Model state

Basic image encoder, use one level of feature pyramid, auxiliary 2d-detection head (use YOLO, no backprop), sparse-head style positional encoding, no denoising train branch

### Tooling

- Data loader: batch input visualize & perf test
- View all gradients
- View number of memory-persisted queries

### Training

- Losses without assignment due to only 1 person
- Focal/class loss for person detection
- L1 3d loss for keypoints
- Backprop only 1 image, only current timestep
- Draw small number of outputs during training showing: image grid, auxiliary 2d detections overlaid on image, other relevant debug info from network


# Misc ToDos (not now, later, just for record keeping):
1. Handle image distortion correctly
2. Optimize floating point sizes of inputs/weights/etc
3. Consider trying MLN
4. Fix missing camera issue
5. Fix incorrect offset issue (confirmed on some sequences)
6. Generate training examples using subsets of complete camera set
7. Fix the kludge of adding the human prior repo to the path of unified perception
8. Fix the odd 1000x1002 resolution on some videos
9. Fix the uneven data distribution between workers
10. Make training resumable
11. Profile forward/backward pass
12. 


# Next Experiments
1. Try increasing the YOLO backbone size