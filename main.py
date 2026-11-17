from src.data_loader import load_mpiigaze_dataset
from src.features import extract_features
from src.evaluate import run_within_subject_cv, run_loso_and_depth_analysis
from src.visualize import generate_plots

def main():

    df = load_mpiigaze_dataset(data_dir="./data")
    X_raw, X_geom = extract_features(df)

    print("\n[1/3] WITHIN-SUBJECT 5-FOLD CROSS-VALIDATION (Personalized Upper Bound)")
    within_df = run_within_subject_cv(df, X_raw, X_geom)
    print(within_df.to_string(index=False))

    print("\n[2/3] 15-SUBJECT LOSO-CV & LOW-SHOT AFFINE CALIBRATION (k = 1, 3, 5, 9)")
    loso_df, calib_curves, spatial_df, fitts_df = run_loso_and_depth_analysis(df, X_raw, X_geom)
    print(loso_df.to_string(index=False))

    print("\n[3/3] SPATIAL HETEROSCEDASTICITY & FITTS'S LAW UI BUTTON SIZING AUDIT")
    print(fitts_df.to_string(index=False))

    within_df.to_csv("./results/within_subject_results.csv", index=False)
    loso_df.to_csv("./results/loso_calibration_results.csv", index=False)
    fitts_df.to_csv("./results/fitts_law_ui_audit.csv", index=False)

    generate_plots(loso_df, calib_curves, spatial_df, out_dir="./results")
    print("\n[+] Done! CSV tables and PNG plots saved in ./results/")


if __name__ == "__main__":
    main()
