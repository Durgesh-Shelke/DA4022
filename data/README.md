# Recorded experiment data

`DA4022_FINAL_TODAY/` preserves the completed evaluation's configurations, environment, source snapshot, measurements, figures, and generated summaries. The bundle's internal layout is retained so relative links in generated reports continue to work.

- [Main results](DA4022_FINAL_TODAY/report/main_summary.csv)
- [Per-scene/per-seed results](DA4022_FINAL_TODAY/report/main_scene_seed.csv)
- [Robustness results](DA4022_FINAL_TODAY/report/robustness_summary.csv)
- [Generated report](DA4022_FINAL_TODAY/report/FINAL_REPORT.md)
- [Environment](DA4022_FINAL_TODAY/environment.json)
- [Configuration](DA4022_FINAL_TODAY/config.json)

The final narrative report is in [../reports/DA4022_Final_Project_Report.md](../reports/DA4022_Final_Project_Report.md).

Large raw downloads, prepared tensor frames, pretrained weights, and adaptive tensor checkpoints are not committed. This archive is for inspecting the recorded study; resuming training requires a complete saved backup, including checkpoints and prepared frames. Fresh runs keep the existing local/Kaggle workspace output paths and do not write into this committed archive.
