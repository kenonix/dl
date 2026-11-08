import csv
import json
import os
import sys
import numpy as np

def main():
    src_file = '/home/kenonix/다운로드/EMGFilters/dl/emg_dataset_fingers_20260929_165231.csv'
    clean_3s_file = '/home/kenonix/다운로드/EMGFilters/dl/emg_dataset_fingers_3s_clean.csv'
    lda_4d_file = '/home/kenonix/다운로드/EMGFilters/dl/emg_dataset_fingers_3s_lda4d.csv'
    html_file = '/home/kenonix/다운로드/EMGFilters/dl/emg_3d_lda_fingers.html'

    print(f"[*] 원본 데이터셋 로드 중: {src_file}")
    with open(src_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    feature_names = header[:-2]
    data = np.array([[float(x) for x in r[:-2]] for r in rows])
    labels = np.array([int(r[-2]) for r in rows])
    actions = np.array([r[-1] for r in rows])

    # 1. 18개 블록에서 각각 중간 3.0초(150샘플 @ 50Hz) 추출
    transitions = []
    curr = None
    cnt = 0
    for i, a in enumerate(actions):
        if a != curr:
            if curr is not None:
                transitions.append((curr, cnt, i - cnt, i))
            curr = a
            cnt = 1
        else:
            cnt += 1
    transitions.append((curr, cnt, len(actions) - cnt, len(actions)))

    target_samples = 150 # 3.0초 @ 50Hz
    trimmed_indices = []
    for act_name, length, start_idx, end_idx in transitions:
        mid_start = start_idx + (length - target_samples) // 2
        mid_end = mid_start + target_samples
        trimmed_indices.extend(range(mid_start, mid_end))

    X_3s = data[trimmed_indices]
    y_3s = labels[trimmed_indices]
    act_3s = actions[trimmed_indices]

    # [1] 3초 정제 75차원 CSV 저장
    with open(clean_3s_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in trimmed_indices:
            writer.writerow(rows[idx])
    print(f"✔ [1] 3초 정제 데이터셋 저장 완료: {clean_3s_file} (총 {len(trimmed_indices)}샘플, 18블록 × 150샘플)")

    # [2] LDA (Linear Discriminant Analysis) 4차원 최적 투영 행렬 계산
    classes = np.unique(y_3s)
    n_classes = len(classes)
    d = X_3s.shape[1]
    n_samples = len(X_3s)

    mu_total = np.mean(X_3s, axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))

    for c in classes:
        Xc = X_3s[y_3s == c]
        mu_c = np.mean(Xc, axis=0)
        diff = Xc - mu_c
        Sw += diff.T @ diff
        diff_mu = (mu_c - mu_total).reshape(-1, 1)
        Sb += len(Xc) * (diff_mu @ diff_mu.T)

    # 수치적 안정성을 위한 정규화 (Regularized Scatter Matrix)
    reg = 1e-4 * np.trace(Sw) / d
    Sw_reg = Sw + reg * np.eye(d)
    invSw = np.linalg.pinv(Sw_reg)

    # 일반화 고유치 분해
    eigvals, eigvecs = np.linalg.eig(invSw @ Sb)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    sort_idx = np.argsort(eigvals)[::-1]
    top_eigvals = eigvals[sort_idx]
    top_eigvecs = eigvecs[:, sort_idx]

    # 상위 4개 고유벡터 (75 x 4)
    W_4d = top_eigvecs[:, :4]
    X_4d = X_3s @ W_4d

    eval_sum = np.sum(top_eigvals[top_eigvals > 0])
    explained_ratios = [float(ev / eval_sum * 100) for ev in top_eigvals[:4]]

    print("\n✔ [2] LDA 고유값 및 판별 분산 기여율 (4차원):")
    for i in range(4):
        print(f"   • LD{i+1}: 고유값 = {top_eigvals[i]:8.4f} (클래스 분별 기여율: {explained_ratios[i]:5.1f}%)")
    print(f"   ➔ 상위 4개 판별축 누적 기여율: {sum(explained_ratios):.1f}%")

    # [3] 4차원 LDA 데이터셋 CSV 저장
    lda_header = ['ld1', 'ld2', 'ld3', 'ld4', 'label', 'action_name']
    with open(lda_4d_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(lda_header)
        for i in range(len(X_4d)):
            row = [f"{X_4d[i, j]:.4f}" for j in range(4)] + [str(y_3s[i]), act_3s[i]]
            writer.writerow(row)
    print(f"✔ [3] 4차원 LDA 데이터셋 저장 완료: {lda_4d_file}")

    # [4] 4D LDA 공간 내 Gaussian Bayes LDA 분류기 5-Fold 교차 검증
    indices = np.arange(len(y_3s))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    accs_lda = []
    cm = np.zeros((6, 6), dtype=int)

    for k in range(5):
        val_idx = folds[k]
        tr_idx = np.setdiff1d(indices, val_idx)
        
        X_tr, y_tr = X_3s[tr_idx], y_3s[tr_idx]
        X_va, y_va = X_3s[val_idx], y_3s[val_idx]
        
        mu_tot = np.mean(X_tr, axis=0)
        Sw_k = np.zeros((d, d))
        Sb_k = np.zeros((d, d))
        means_k = {}
        priors_k = {}
        for c in classes:
            Xc = X_tr[y_tr == c]
            means_k[c] = np.mean(Xc, axis=0)
            priors_k[c] = len(Xc) / len(X_tr)
            diff = Xc - means_k[c]
            Sw_k += diff.T @ diff
            diff_mu = (means_k[c] - mu_tot).reshape(-1, 1)
            Sb_k += len(Xc) * (diff_mu @ diff_mu.T)
            
        Sw_k_reg = Sw_k + 1e-4 * np.trace(Sw_k) / d * np.eye(d)
        invSw_k = np.linalg.pinv(Sw_k_reg)
        ev_k, evec_k = np.linalg.eig(invSw_k @ Sb_k)
        s_idx = np.argsort(np.real(ev_k))[::-1]
        W_k = np.real(evec_k[:, s_idx[:4]])
        
        Z_tr = X_tr @ W_k
        Z_va = X_va @ W_k
        
        Z_means = {c: np.mean(Z_tr[y_tr == c], axis=0) for c in classes}
        Z_Sw = sum((Z_tr[y_tr == c] - Z_means[c]).T @ (Z_tr[y_tr == c] - Z_means[c]) for c in classes)
        Z_cov = Z_Sw / (len(X_tr) - len(classes))
        inv_Z_cov = np.linalg.pinv(Z_cov + 1e-5 * np.eye(4))
        
        preds = []
        for z in Z_va:
            scores = [z @ inv_Z_cov @ Z_means[c] - 0.5 * (Z_means[c] @ inv_Z_cov @ Z_means[c]) + np.log(priors_k[c]) for c in classes]
            preds.append(classes[np.argmax(scores)])
        preds = np.array(preds)
        acc = np.mean(preds == y_va)
        accs_lda.append(acc)
        for t, p in zip(y_va, preds):
            cm[t, p] += 1

    print(f"\n✔ [4] 4D LDA 공간 5-Fold 교차 검증 정확도: {np.mean(accs_lda)*100:.2f}%")
    class_names = ['휴식 (Relax)', '엄지 (Thumb)', '검지 (Index)', '중지 (Middle)', '약지 (Ring)', '새끼 (Pinky)']
    print("=== 🎯 4D LDA 오차 행렬 (Confusion Matrix) ===")
    for i, c in enumerate(class_names):
        tot = np.sum(cm[i])
        correct = cm[i, i]
        print(f"  {c:14s}: 정확도 {(correct/tot)*100:5.1f}% ({correct}/{tot}) | 예측: {list(cm[i])}")

    # [5] 3D 인터랙티브 Plotly HTML 생성 (LD1 x LD2 x LD3)
    colors = [
        'rgba(46, 204, 113, 0.85)',  # Relax (Green)
        'rgba(231, 76, 60, 0.85)',   # Thumb (Red)
        'rgba(52, 152, 219, 0.85)',  # Index (Blue)
        'rgba(243, 156, 18, 0.85)',  # Middle (Orange)
        'rgba(155, 89, 182, 0.85)',  # Ring (Purple)
        'rgba(26, 188, 156, 0.85)',  # Pinky (Teal)
    ]

    traces = []
    for c_idx, c in enumerate(classes):
        mask = (y_3s == c)
        pts = X_4d[mask]
        trace = {
            "type": "scatter3d",
            "mode": "markers",
            "name": class_names[c_idx],
            "x": pts[:, 0].round(3).tolist(),
            "y": pts[:, 1].round(3).tolist(),
            "z": pts[:, 2].round(3).tolist(),
            "text": [f"<b>{class_names[c_idx]}</b><br>LD1: {pts[j,0]:.2f}<br>LD2: {pts[j,1]:.2f}<br>LD3: {pts[j,2]:.2f}<br>LD4: {pts[j,3]:.2f}" for j in range(len(pts))],
            "marker": {
                "size": 4,
                "color": colors[c_idx],
                "opacity": 0.85
            }
        }
        traces.append(trace)

    layout = {
        "title": {
            "text": "🌟 sEMG 3초 정제 4차원 LDA 선형 판별 분석 3D 투영도 [emg_dataset_fingers_3s_lda4d.csv]",
            "font": {"color": "#38bdf8", "size": 20}
        },
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#0f172a",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 50},
        "scene": {
            "xaxis": {"title": f"LD 1 ({explained_ratios[0]:.1f}%)", "color": "#94a3b8", "gridcolor": "#334155"},
            "yaxis": {"title": f"LD 2 ({explained_ratios[1]:.1f}%)", "color": "#94a3b8", "gridcolor": "#334155"},
            "zaxis": {"title": f"LD 3 ({explained_ratios[2]:.1f}%)", "color": "#94a3b8", "gridcolor": "#334155"},
            "bgcolor": "#1e293b",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}}
        },
        "legend": {
            "font": {"color": "#f8fafc", "size": 13},
            "bgcolor": "rgba(30, 41, 59, 0.8)"
        }
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3초 정제 4차원 LDA 선형 판별 분석 3D 시각화</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; }}
        .header {{ max-width: 1200px; margin: 0 auto 20px auto; padding: 20px; background: rgba(30, 41, 59, 0.8); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }}
        h1 {{ margin: 0 0 10px 0; font-size: 22px; color: #38bdf8; }}
        p {{ margin: 5px 0; color: #94a3b8; font-size: 14px; line-height: 1.6; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; background: #0284c7; color: white; margin-right: 8px; }}
        #plot {{ width: 100%; max-width: 1200px; height: 750px; margin: 0 auto; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">LDA 4D Projection</span>
        <span class="badge" style="background:#10b981;">3.0s Trimmed</span>
        <span class="badge" style="background:#8b5cf6;">6 Classes</span>
        <h1>🖐️ 다섯 손가락 3초 정제 4차원 LDA (Linear Discriminant Analysis) 3D 투영 공간</h1>
        <p>• <b>데이터셋</b>: <code>emg_dataset_fingers_20260929_165231.csv</code>에서 과도기 제외 안정된 중간 3.0초(150샘플) 추출 ➔ 총 2,700샘플</p>
        <p>• <b>LDA 차원 축소</b>: 클래스 간 분산(Sb) 극대화 & 클래스 내 분산(Sw) 극소화 최적화를 통해 75차원 ➔ 상위 4대 판별 벡터(LD1~LD4) 투영 (누적 분별력: {sum(explained_ratios):.1f}%)</p>
        <p>• <b>3D 투영 축</b>: X축 = LD1 ({explained_ratios[0]:.1f}%), Y축 = LD2 ({explained_ratios[1]:.1f}%), Z축 = LD3 ({explained_ratios[2]:.1f}%) | 4번째 축 LD4 ({explained_ratios[3]:.1f}%)는 CSV에 포함</p>
    </div>
    <div id="plot"></div>
    <script>
        const traces = {json.dumps(traces)};
        const layout = {json.dumps(layout)};
        Plotly.newPlot('plot', traces, layout, {{ responsive: true }});
    </script>
</body>
</html>"""

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✔ [5] 3D LDA 인터랙티브 시각화 HTML 생성 완료: {html_file}")

if __name__ == '__main__':
    main()
