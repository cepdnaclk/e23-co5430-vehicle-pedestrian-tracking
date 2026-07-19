# Project Roadmap: Vehicle or Pedestrian Tracking in Video (CO543 / CO5430)

This roadmap outlines the milestones and timeline for the Computer Vision Project (2026), developed by Group G07.

## Team Members
- M.R.A. Rahman (E/23/282)
- A. Piraveen (E/23/273)
- M.M.M.A. Nular (E/23/249)
- V. Thanush (E/23/392)

## Roadmap & Timeline

### M0 – Group & Topic Registration (Jul 1 – Jul 7, 2026)
- **Focus:** Group formed (G07), topic P07 selected, dataset & GitHub link registered.
- **Owner(s):** All members

### M1 – Proposal & Project Plan (Jul 14, 2026)
- **Focus:** Complete and submit the project proposal document.
- **Owner(s):** All members

### M2 – Data Preparation (Jul 15 – Jul 21, 2026)
- **Focus:** Download MOT16 sequences, set up the evaluation scripts, and format frames.
- **Owner(s):** Rahman, Nular

### M3 – Dataset & Baseline Checkpoint (Jul 28, 2026)
- **Focus:** Get the basic YOLOv8 + SORT pipeline running.
- **Owner(s):** Piraveen, Thanush

### M4 – Method Development (Jul 29 – Aug 15, 2026)
- **Focus:** Integrate appearance re-ID (DeepSORT) / ByteTrack-style association; tune parameters.
- **Owner(s):** All members

### M5 – Prototype & Preliminary Results (Aug 18, 2026)
- **Focus:** Preliminary MOTA/IDF1 results, qualitative tracks, failure-case analysis.
- **Owner(s):** All members

### M6 – Experiment Freeze (Aug 25, 2026)
- **Focus:** Finish testing and lock in the final code.
- **Owner(s):** Rahman, Piraveen

### M7 – Draft Report (Sep 1, 2026)
- **Focus:** Report skeleton, result tables/figures, demo checklist, repo cleanup.
- **Owner(s):** Nular, Thanush

### M8 – Final Submission & Demo (Sep 7, 2026)
- **Focus:** Final report, repository, slides, demo video.
- **Owner(s):** All members

---
## Goals
- **Minimum:** Get YOLOv8 and SORT working together to output a tracked video on one sequence.
- **Expected:** Implement the upgraded tracker (DeepSORT/ByteTrack) and create a comparison table showing metric differences.
- **Stretch:** Break down tracking accuracy to compare vehicle vs. pedestrian tracking.
