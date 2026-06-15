import os
import shutil

src_dir = r"E:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate\charts_v2"
dst_dir = r"E:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate\charts_v2\tmp"

os.makedirs(dst_dir, exist_ok=True)

files_to_copy = [
    "Figure_46_Recall_Precision_NDCG_K_CLIP.png",
    "Figure_47_Recall_Precision_NDCG_K_MBNV2.png",
    "Figure_48_NDCG_K_BM3_ablation.png",
    "Figure_55_BM3_NDCG_10_by_sim_type_x_encoder.png",
    "Figure_56_NDCG_K_COMBIGCN_ablation.png",
    "Figure_64_NDCG_K_FREEDOM_ablation.png",
    "Figure_72_Tier_1_Best_Config_per_Model_Metrics_Overview.png",
    "Figure_74_Radar_Overview_CLIP_12_Configs.png",
    "Figure_75_Radar_Overview_MBNv2_12_Configs.png",
    "Figure_82_Best_Overall_Models_for_Each_Metric.png"
]

print(f"Copying files from {src_dir} to {dst_dir}...")
copied_count = 0
for filename in files_to_copy:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"✓ Copied: {filename}")
        copied_count += 1
    else:
        print(f"✗ Not found: {filename}")

print(f"\nDone! Copied {copied_count}/{len(files_to_copy)} files.")
