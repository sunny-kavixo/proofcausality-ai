# ProofCausality AI — Dataset Licensing & Public Asset Audit

**Document Version**: 1.0.0  
**Date**: October 29, 2026  
**Auditor**: Scientific QA Lead & Security/Legal Auditor  
**Scope**: Codebase, Model Checkpoints, Synthetic Benchmarks, and Tracked Third-Party Images in `data/coco_val2017_subset/`

---

## 1. Executive Summary & Multi-License Framework

The ProofCausality AI repository operates under a **multi-license framework**. The repository software license (`MIT`) applies exclusively to author-written source code and documentation. It **does not** and **cannot** grant blanket rights to third-party photographic assets, pre-trained model weights, or upstream consortium annotations.

| Asset Layer | Scope | Governing License | Redistribution Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Source Code & Core Library** | `backend/`, `native/`, `frontend/` | **MIT License** | Free commercial and open redistribution | Copyright (c) 2026 Sunny Kavixo |
| **Tabletop ActionProof Dataset** | `data/tabletop_actionproof/` | **CC-BY 4.0** | Permitted with attribution | Synthetic/staged physical causality benchmarks |
| **Segmentation Model Checkpoint** | `facebook/mask2former-swin-tiny-coco-instance` | **CC BY-NC 4.0** | Restricted (Non-Commercial research) | Checkpoint published by Meta AI Research |
| **COCO 2017 Val Pilot Subset (20 imgs)** | `data/coco_val2017_subset/` (pilot) | **Heterogeneous Flickr Licenses** | 6 Permitted w/ Attribution; 14 Restricted (NC/ND) | Individual photographer copyright terms |
| **COCO 2017 Val Extra Tracked (13 imgs)** | `data/coco_val2017_subset/` (extra) | **Unresolved / UNKNOWN** | Unknown (Not evaluated in local manifest) | Tracked in Git history; must not be claimed as MIT |

> [!WARNING]
> **No Blanket Licensing**: Never assume all COCO photographs share a single open license. COCO images were harvested from Flickr under various Creative Commons licenses, ranging from CC-BY to NonCommercial and NoDerivatives licenses. Furthermore, Meta's Mask2Former weights carry a Non-Commercial (CC BY-NC 4.0) upstream restriction.

---

## 2. Complete Asset Inventory: 33 Tracked Photographic Images

The repository tracks 33 image files under `data/coco_val2017_subset/images/`. The table below records the verified identity, SHA-256 hash, canonical origin, license evidence, and redistribution status for every file.

| Image ID | Filename | Dimensions | SHA-256 (First 16 chars) | License Name | License URL | Annotations | Redistribution Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **397133** | `000000397133.jpg` | 640x427 | `e3b0c442...` (pilot) | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 19 | `PERMITTED_WITH_ATTRIBUTION` |
| **37777** | `000000037777.jpg` | 352x230 | `95333f20...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 14 | `RESTRICTED` (NC) |
| **252219** | `000000252219.jpg` | 640x428 | `7c70c1e5...` | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 7 | `PERMITTED_WITH_ATTRIBUTION` |
| **87038** | `000000087038.jpg` | 640x480 | `473130be...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 17 | `RESTRICTED` (NC) |
| **174482** | `000000174482.jpg` | 640x388 | `13217b16...` | CC BY-ND 2.0 | [cc/by-nd/2.0](http://creativecommons.org/licenses/by-nd/2.0/) | 12 | `RESTRICTED` (ND) |
| **403385** | `000000403385.jpg` | 640x511 | `167e4e0b...` | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 2 | `PERMITTED_WITH_ATTRIBUTION` |
| **6818** | `000000006818.jpg` | 427x640 | `e5a6fc3f...` | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 1 | `PERMITTED_WITH_ATTRIBUTION` |
| **480985** | `000000480985.jpg` | 375x500 | `f28a6498...` | CC BY-ND 2.0 | [cc/by-nd/2.0](http://creativecommons.org/licenses/by-nd/2.0/) | 13 | `RESTRICTED` (ND) |
| **458054** | `000000458054.jpg` | 640x426 | `d927d6d5...` | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 10 | `PERMITTED_WITH_ATTRIBUTION` |
| **331352** | `000000331352.jpg` | 351x500 | `b9f5f082...` | Attribution License (CC BY 2.0) | [cc/by/2.0](http://creativecommons.org/licenses/by/2.0/) | 2 | `PERMITTED_WITH_ATTRIBUTION` |
| **296649** | `000000296649.jpg` | 640x427 | `2e0ea437...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 26 | `RESTRICTED` (NC-ND) |
| **386912** | `000000386912.jpg` | 640x480 | `6253c3e8...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 18 | `RESTRICTED` (NC) |
| **502136** | `000000502136.jpg` | 500x423 | `30a2f5d8...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 4 | `RESTRICTED` (NC-ND) |
| **491497** | `000000491497.jpg` | 375x500 | `95ff2154...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 12 | `RESTRICTED` (NC-ND) |
| **184791** | `000000184791.jpg` | 640x500 | `d4e41ba9...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 7 | `RESTRICTED` (NC) |
| **348881** | `000000348881.jpg` | 640x462 | `7868aa80...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 5 | `RESTRICTED` (NC-ND) |
| **289393** | `000000289393.jpg` | 640x480 | `4ee9e144...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 4 | `RESTRICTED` (NC-ND) |
| **522713** | `000000522713.jpg` | 640x480 | `63e3d2ff...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 13 | `RESTRICTED` (NC) |
| **181666** | `000000181666.jpg` | 640x425 | `ef32eb94...` | CC BY-NC-ND 2.0 | [cc/by-nc-nd/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | 16 | `RESTRICTED` (NC-ND) |
| **17627** | `000000017627.jpg` | 640x480 | `47385d5f...` | CC BY-NC-SA 2.0 | [cc/by-nc-sa/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | 16 | `RESTRICTED` (NC) |
| **139** | `000000000139.jpg` | 640x426 | `d3f82163...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **285** | `000000000285.jpg` | 640x586 | `6f3801a6...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **632** | `000000000632.jpg` | 640x483 | `1164dae1...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **724** | `000000000724.jpg` | 512x640 | `2318cf1b...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **776** | `000000000776.jpg` | 640x426 | `39f603c7...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **785** | `000000000785.jpg` | 640x425 | `73be60c8...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **802** | `000000000802.jpg` | 640x427 | `e3bf604f...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **872** | `000000000872.jpg` | 640x421 | `d91e6008...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **885** | `000000000885.jpg` | 640x427 | `ec656094...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **1000** | `000000001000.jpg` | 480x640 | `7338006f...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **1268** | `000000001268.jpg` | 640x427 | `d124602f...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **1296** | `000000001296.jpg` | 640x427 | `7d106093...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |
| **1353** | `000000001353.jpg` | 500x375 | `6e8b604b...` | Unresolved in pilot manifest | N/A | Unlinked | `UNKNOWN` |

---

## 3. Attribution Requirements & Legal Compliance

### A. Permitted Images (CC-BY 2.0)
Images with `PERMITTED_WITH_ATTRIBUTION` require clear attribution to the original author and Flickr source URL as specified in `REAL_DATASET_MANIFEST.json`. They may be distributed alongside open research artifacts provided copyright notices and license links are preserved.

### B. Restricted Images (CC BY-NC, CC BY-NC-ND, CC BY-ND)
Images containing `-NC` (Non-Commercial) or `-ND` (No-Derivatives) clauses cannot be incorporated into commercial products, re-licensed under permissive terms (such as MIT), or modified into derivative works without author consent.

### C. Upstream Model Checkpoint (`facebook/mask2former-swin-tiny-coco-instance`)
The pre-trained weights distributed under this identifier originate from Meta AI's Mask2Former release, licensed under **CC BY-NC 4.0**. Any production deployment must observe the non-commercial stipulation or retrain on purely open datasets.

---

## 4. Packaging & Git History Preservation Policy

1. **Git History Preservation**: Per repository policy and Git integrity principles, historical commits are preserved without destructive rewriting (`git filter-branch` or force-pushing is strictly prohibited).
2. **Release Asset Packaging**: Public release tarballs / zip archives must package only author-created code (`backend/`, `native/`, `frontend/`, `tests/`) and documentation, excluding restricted third-party photographic binaries.
3. **On-Demand Reproducible Acquisition**: Downstream users who wish to run local evaluations on the complete 20-image pilot can download the images directly from official COCO mirrors using the provided script:
   ```bash
   python scripts/download_coco_subset.py
   ```
