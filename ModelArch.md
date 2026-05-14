Input grayscale image
[B, 1, H, W]
        |
        |-----------------------------|
        |                             |
        v                             v
Frozen Teacher Backbone          Trainable branches
PDN / WideResNet / ResNet         Student + Autoencoder
        |                             |
        v                             v
Dense teacher patch features      Predicted normal features
[B, C, h, w]                       [B, C, h, w]
        |
        |-----------------------------|
        |                             |
        v                             v
PatchCore memory distance      Student/AE feature error
        |                             |
        |------------- fuse ----------|
                      |
                      v
              Final anomaly map
              [B, 1, H, W]
                      |
                      v
              Image anomaly score



                       ┌─────────────────────────────┐
                       │   Input grayscale image      │
                       │        B x 1 x H x W         │
                       └──────────────┬──────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               │                      │                      │
               v                      v                      v
┌────────────────────────┐ ┌───────────────────────┐ ┌────────────────────────┐
│ Frozen Teacher Branch  │ │ Student Branch        │ │ Autoencoder Branch     │
│ PDN / ResNet / WRN     │ │ Trainable CNN         │ │ Trainable bottleneck   │
└───────────┬────────────┘ └───────────┬───────────┘ └───────────┬────────────┘
            │                          │                         │
            v                          v                         v
┌────────────────────────┐ ┌───────────────────────┐ ┌────────────────────────┐
│ Teacher patch features │ │ Student prediction    │ │ AE feature prediction  │
│       T                │ │       S               │ │       A                │
│   B x C x h x w        │ │   B x C x h x w       │ │   B x C x h x w        │
└───────────┬────────────┘ └───────────┬───────────┘ └───────────┬────────────┘
            │                          │                         │
            │                          v                         v
            │              ┌───────────────────────┐ ┌────────────────────────┐
            │              │ Student error map     │ │ AE error map           │
            │              │ mean((S - T)^2)       │ │ mean((A - T)^2)        │
            │              └───────────┬───────────┘ └───────────┬────────────┘
            │                          │                         │
            v                          │                         │
┌────────────────────────┐             │                         │
│ Patch memory bank      │             │                         │
│ Normal patch features  │             │                         │
└───────────┬────────────┘             │                         │
            │                          │                         │
            v                          │                         │
┌────────────────────────┐             │                         │
│ PatchCore distance map │             │                         │
│ nearest normal patch   │             │                         │
└───────────┬────────────┘             │                         │
            │                          │                         │
            └──────────────┬───────────┴─────────────┬───────────┘
                           v                         v
                  ┌────────────────────────────────────────┐
                  │ Calibrated weighted map fusion          │
                  │ 0.60 PatchCore + 0.25 Student + 0.15 AE │
                  └────────────────────┬───────────────────┘
                                       v
                          ┌────────────────────────┐
                          │ Final anomaly heatmap  │
                          │      B x 1 x H x W     │
                          └───────────┬────────────┘
                                      v
                          ┌────────────────────────┐
                          │ Image anomaly score    │
                          │ top-k mean heatmap     │
                          └────────────────────────┘