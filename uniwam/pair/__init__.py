"""PAIR: Plug-in Adaptive Interface Router (training and deployment tools).

Pipeline, all with the UniW(A)M backbone frozen:

1. ``uniwam.pair.windows``   sample four-chunk windows with train/val/test splits
2. ``uniwam.inference.fourway --windows-json``   per-chunk action MAE of every interface (labels)
3. ``uniwam.pair.features``  per-chunk router features from the frozen backbone
4. ``uniwam.pair.train``     train one router seed on features + labels
5. ``uniwam.pair.evaluate``  offline routing quality of one router or a seed ensemble
6. ``uniwam.pair.gate.PAIRGate``  per-chunk interface selection at deployment
"""
