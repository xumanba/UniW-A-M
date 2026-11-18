// All numbers are taken from the UniW(A)M paper.
window.UNIWAM = {
  interfaces: ["Joint-WAM", "IDM-WAM", "Action-Centric-WAM", "Action-Only-WAM"],
  tasks: ["Folding Clothes", "Folding Box", "Tabletop Cleaning"],

  // Table 2: real-world success (%), 20 trials per task
  realWorld: [
    { name: "π0.5", kind: "base", v: [60, 65, 70], avg: 65.0 },
    { name: "DreamZero", kind: "base", v: [45, 55, 60], avg: 53.3 },
    { name: "WorldScape Policy 2.0", kind: "base", v: [75, 75, 80], avg: 76.7 },
    { name: "Joint-WAM", kind: "joint", v: [90, 80, 90], avg: 86.7 },
    { name: "IDM-WAM", kind: "idm", v: [90, 85, 90], avg: 88.3 },
    { name: "Action-Centric-WAM", kind: "ac", v: [85, 75, 85], avg: 81.7 },
    { name: "Action-Only-WAM", kind: "ao", v: [85, 70, 85], avg: 80.0 }
  ],

  // Training data
  data: [
    { task: "Tabletop Cleaning", episodes: 2426, frames: 1528831, hours: 14.16 },
    { task: "Folding Box", episodes: 1217, frames: 1768220, hours: 16.37 },
    { task: "Folding Clothes", episodes: 605, frames: 3110732, hours: 28.80 }
  ],

  libero: {
    cols: ["Spatial", "Object", "Goal", "Long", "Average"],
    groups: [
      { label: "Vision-Language-Action Models", rows: [
        ["OpenVLA", 84.7, 88.4, 79.2, 53.7, 76.5],
        ["π0", 96.8, 98.8, 95.8, 85.2, 94.1],
        ["π0.5", 98.8, 98.2, 98.0, 92.4, 96.9]
      ]},
      { label: "World Action Models", rows: [
        ["Motus", 96.8, 99.8, 96.6, 97.6, 97.7],
        ["LingBot-VA", 98.5, 99.6, 97.2, 98.5, 98.5],
        ["Fast-WAM", 98.2, 100.0, 97.0, 95.2, 97.6]
      ]},
      { label: null, ours: true, rows: [["UniW(A)M", 99.2, 99.0, 96.6, 95.2, 97.5]] }
    ]
  },

  robotwin: [
    { name: "π0 (single-task)", c2c: 46.42, c2r: 16.34 },
    { name: "π0.5", c2c: 70.7, c2r: 46.0 },
    { name: "Fast-WAM", c2c: 77.8, c2r: 1.9 },
    { name: "WorldScape Policy 2.0", c2c: 84.92, c2r: 38.5 },
    { name: "UniW(A)M", c2c: 84.38, c2r: 66.60, ours: true }
  ],

  worldArena: {
    metrics: ["JEPA Similarity", "Motion Smoothness", "Interaction Quality", "Trajectory Accuracy", "Instruction Following"],
    tasks: {
      "Folding Clothes": {
        "Cosmos-Predict 2.5": [21.86, 60.70, 64.00, 7.20, 56.50, 48.08],
        "CtrlWorld": [17.57, 78.21, 63.50, 18.56, 51.50, 53.09],
        "UniW(A)M": [44.39, 87.77, 80.00, 22.87, 93.50, 61.51]
      },
      "Folding Box": {
        "Cosmos-Predict 2.5": [40.27, 52.47, 76.50, 10.91, 95.00, 51.07],
        "CtrlWorld": [63.81, 66.56, 64.50, 14.73, 78.00, 55.48],
        "UniW(A)M": [65.41, 68.69, 70.50, 17.28, 84.00, 55.89]
      },
      "Tabletop Cleaning": {
        "Cosmos-Predict 2.5": [4.63, 48.49, 54.00, 2.88, 41.00, 31.75],
        "CtrlWorld": [33.76, 57.93, 60.00, 5.94, 59.00, 50.23],
        "UniW(A)M": [50.30, 63.10, 66.50, 7.26, 68.50, 52.53]
      }
    }
  },

  // Table 4: OOD success (%)
  ood: [
    { name: "Joint-WAM", kind: "joint", v: [65, 55, 70], avg: 63.3 },
    { name: "IDM-WAM", kind: "idm", v: [65, 50, 75], avg: 63.3 },
    { name: "Action-Centric-WAM", kind: "ac", v: [50, 40, 55], avg: 48.3 },
    { name: "Action-Only-WAM", kind: "ao", v: [50, 45, 55], avg: 50.0 },
    { name: "PAIR", kind: "ours", v: [75, 70, 85], avg: 76.7 }
  ],

  // Folding Clothes OOD: adaptation controls
  adaptation: [
    ["Phase-based heuristic routing", 45.0, null],
    ["Joint-WAM + LoRA", 65.0, 65.0],
    ["IDM-WAM + LoRA", 65.0, 65.0],
    ["Action-Centric-WAM + LoRA", 50.0, 55.0],
    ["Action-Only-WAM + LoRA", 50.0, 50.0],
    ["PAIR (router only, backbone frozen)", 75.0, null]
  ],

  // Figure 6: PAIR selection share (%) per phase: [Joint, IDM, AC, AO]
  phases: ["Reach", "Contact", "Core"],
  selection: {
    "Folding Clothes": [[16, 10, 30, 44], [30, 35, 20, 15], [50, 40, 5, 5]],
    "Folding Box": [[20, 20, 40, 20], [35, 35, 10, 20], [35, 50, 5, 10]],
    "Tabletop Cleaning": [[15, 15, 30, 40], [25, 35, 20, 20], [35, 35, 20, 10]]
  },

  // Figure 7
  pareto: [
    { name: "Action-Centric-WAM", kind: "ac", t: 245, s: 48.3 },
    { name: "Action-Only-WAM", kind: "ao", t: 267.5, s: 50.0 },
    { name: "Joint-WAM", kind: "joint", t: 299, s: 63.3 },
    { name: "IDM-WAM", kind: "idm", t: 414, s: 63.3 },
    { name: "UniW(A)M + PAIR", kind: "ours", t: 308.9, s: 76.7 }
  ],
  latency: { "Joint-WAM": 428.1, "IDM-WAM": 532.7, "Action-Centric-WAM": 154.3, "Action-Only-WAM": 225.8 },

  // Table 5: ablations, average success over three tasks
  ablation: {
    configs: ["Single DiT (Dense)", "MoE, sample-level routing", "CATR, w/o AC-WM", "CATR, all five (Ours)"],
    avg: {
      "Joint-WAM": [13.3, 51.7, 71.7, 86.7],
      "IDM-WAM": [16.7, 60.0, 73.3, 88.3],
      "Action-Centric-WAM": [3.3, 10.0, 50.0, 81.7],
      "Action-Only-WAM": [1.7, 10.0, 36.7, 80.0]
    },
    full: {
      "Single DiT (Dense)": [[10, 10, 20], [20, 0, 30], [0, 0, 10], [0, 0, 5]],
      "MoE, sample-level routing": [[50, 45, 60], [80, 30, 70], [10, 0, 20], [10, 0, 20]],
      "CATR, w/o AC-WM": [[75, 60, 80], [80, 60, 80], [55, 30, 65], [50, 10, 50]],
      "CATR, all five (Ours)": [[90, 80, 90], [90, 85, 90], [85, 75, 85], [85, 70, 85]]
    }
  },

  // Progressive co-training on Folding Clothes
  progressive: {
    steps: ["Joint", "+ AC-WM", "+ IDM", "+ Action-Centric", "+ Action-Only"],
    "Joint-WAM": [50, 60, 75, 75, 90],
    "IDM-WAM": [null, null, 60, 85, 90],
    "Action-Centric-WAM": [null, null, null, 60, 85],
    "Action-Only-WAM": [null, null, null, null, 85]
  },

  // Expert removal (Folding Clothes, 20K continued steps)
  removal: [
    { iface: "Joint-WAM", kind: "joint", full: 90, world: ["VGM", 70], action: ["IDM", 20] },
    { iface: "IDM-WAM", kind: "idm", full: 95, world: ["VGM", 50], action: ["IDM", 35] },
    { iface: "Action-Centric-WAM", kind: "ac", full: 85, world: ["WM", 55], action: ["VLA", 30] },
    { iface: "Action-Only-WAM", kind: "ao", full: 85, world: ["VGM", 60], action: ["VLA", 25] }
  ],

  // Gate-weighted utilization (%): shared vs atomic
  utilization: [
    ["Joint-WAM", "Video", "VGM", 78.4], ["Joint-WAM", "Action", "IDM", 63.2],
    ["IDM-WAM", "Video", "VGM", 74.9], ["IDM-WAM", "Action", "IDM", 66.7],
    ["Action-Centric-WAM", "Video", "WM", 81.3], ["Action-Centric-WAM", "Action", "VLA", 59.5],
    ["Action-Only-WAM", "Video", "VGM", 72.1], ["Action-Only-WAM", "Action", "VLA", 65.8],
    ["AC-WM", "Video", "WM", 69.6]
  ],

  dense: [
    ["Dense (Wan2.1-14B)", 10.0, 20.0, 0.0, 0.0, 7.5],
    ["Dense (Wan2.2-5B)", 12.5, 0.0, 0.0, 0.0, 3.1],
    ["UniW(A)M", 90.0, 90.0, 85.0, 85.0, 87.5]
  ],

  // Interface explorer. Tokens: f0 (observed frame), f1..fh (future video), a1..ah (future actions)
  explorer: {
    "AC-WM": {
      kind: "acwm", role: "Action-conditioned world model",
      caps: ["WM"], train: ["Shared", "WM"], infer: ["Shared", "WM"],
      desc: "Predicts future video given externally supplied actions. Actions are clean conditions rather than prediction targets, so the interface simulates the world without generating actions.",
      deploy: "World simulation / evaluation",
      mask: { trainTokens: ["f0", "f1", "fh", "a1", "ah"], trainRows: ["f0", "f1", "fh"], train: [[1,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]],
              inferTokens: ["f0", "f1", "fh", "a1", "ah"], inferRows: ["f0", "f1", "fh"], infer: [[1,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]] }
    },
    "Joint-WAM": {
      kind: "joint", role: "Mutual visibility",
      caps: ["VGM", "IDM"], train: ["Shared", "VGM", "IDM"], infer: ["Shared", "VGM", "IDM"],
      desc: "Denoises future video and actions jointly with bidirectional attention, so the two evolving representations inform one another.",
      deploy: "428.1 ms / step · 86.7% real-world",
      mask: { trainTokens: ["f0","f1","fh","a1","ah"], trainRows: ["f0","f1","fh","a1","ah"],
              train: [[1,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
              inferTokens: ["f0","f1","fh","a1","ah"], inferRows: ["f0","f1","fh","a1","ah"],
              infer: [[1,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] }
    },
    "IDM-WAM": {
      kind: "idm", role: "Sequential vision → action",
      caps: ["VGM", "IDM"], train: ["Shared", "VGM", "IDM"], infer: ["Shared", "VGM", "IDM"],
      desc: "VGM first imagines a visual rollout; IDM then infers the actions that realize it. Action tokens see the predicted video, video tokens never see future actions.",
      deploy: "532.7 ms / step · 88.3% real-world",
      mask: { trainTokens: ["f0","f1","fh","a1","ah"], trainRows: ["f0","f1","fh","a1","ah"],
              train: [[1,0,0,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[1,1,1,1,1]],
              inferTokens: ["f0","f1","fh","a1","ah"], inferRows: ["f0","f1","fh","a1","ah"],
              infer: [[1,0,0,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[1,1,1,1,1]] }
    },
    "Action-Centric-WAM": {
      kind: "ac", role: "Action-centric with video supervision",
      caps: ["VLA", "WM"], train: ["Shared", "VLA", "WM"], infer: ["Shared", "VLA"],
      desc: "VLA predicts actions without seeing future video, while WM learns their visual consequences as auxiliary supervision. Future video is dropped at deployment.",
      deploy: "154.3 ms / step · 81.7% real-world",
      mask: { trainTokens: ["f0","f1","fh","a1","ah"], trainRows: ["f0","f1","fh","a1","ah"],
              train: [[1,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,0,0,1,1],[1,0,0,1,1]],
              inferTokens: ["f0","a1","ah"], inferRows: ["f0","a1","ah"],
              infer: [[1,0,0],[1,1,1],[1,1,1]] }
    },
    "Action-Only-WAM": {
      kind: "ao", role: "Action generation with auxiliary video",
      caps: ["VLA", "VGM"], train: ["Shared", "VLA", "VGM"], infer: ["Shared", "VLA", "VGM*"],
      desc: "Both future branches predict from the observed context only; cross-modal future attention is disabled. Inference keeps only context and action tokens (*VGM processes the conditioning frame).",
      deploy: "225.8 ms / step · 80.0% real-world",
      mask: { trainTokens: ["f0","f1","fh","a1","ah"], trainRows: ["f0","f1","fh","a1","ah"],
              train: [[1,0,0,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,0,0,1,1],[1,0,0,1,1]],
              inferTokens: ["f0","a1","ah"], inferRows: ["f0","a1","ah"],
              infer: [[1,0,0],[1,1,1],[1,1,1]] }
    }
  }
};
