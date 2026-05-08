t(f"实验 {i + 1}: {r2:.4f}")

    mean_r2 = np.mean(test_r2_results)
    std_r2 = np.std(test_r2_results)
    print("-" * 45)
    print(f"🎯 真实平均表现 (Mean ± Std): {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"📈 最高分与最低分差距 (Max - Min): {max(test_r2_results) - min(test_r2_results):.4f}")
    print("=" * 45)
