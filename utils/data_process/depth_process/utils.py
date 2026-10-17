import numpy as np

# ===================== V1: MAD Median Absolute Deviation (你现有) =====================
def compute_scale_offset_v1(colmap_samp, mono_samp):    
    t_colmap = np.median(colmap_samp)
    s_colmap = np.mean(np.abs(colmap_samp - t_colmap))
    t_mono = np.median(mono_samp)
    s_mono = np.mean(np.abs(mono_samp - t_mono)) + 1e-8

    scale = s_colmap / s_mono
    offset = t_colmap - t_mono * scale
    return scale, offset

# ===================== V2: Mean-Standard Deviation 均值标准差对齐 =====================
def compute_scale_offset_v2(colmap_samp, mono_samp):    
    mean_r = np.mean(colmap_samp)
    std_r = np.std(colmap_samp)
    mean_m = np.mean(mono_samp)
    std_m = np.std(mono_samp) + 1e-8
    
    scale = std_r / std_m
    offset = mean_r - mean_m * scale
    return scale, offset

# ===================== V3: Ordinary Least Squares 普通最小二乘 =====================
def compute_scale_offset_v3(colmap_samp, mono_samp):
    # 目标: minimize sum((s*zm + o - zr)^2)
    mean_m = np.mean(mono_samp)
    mean_r = np.mean(colmap_samp)
    cov_mr = np.mean((mono_samp - mean_m) * (colmap_samp - mean_r))
    var_m = np.mean((mono_samp - mean_m) ** 2) + 1e-8
    scale = cov_mr / var_m
    offset = mean_r - scale * mean_m
    return scale, offset

# ===================== V4: Huber Weighted Robust Least Squares 加权鲁棒最小二乘 =====================
def compute_scale_offset_v4(colmap_samp, mono_samp):    
    mean_m = np.mean(mono_samp)
    mean_r = np.mean(colmap_samp)
    cov_mr = np.mean((mono_samp - mean_m) * (colmap_samp - mean_r))
    var_m = np.mean((mono_samp - mean_m) ** 2) + 1e-8
    s0 = cov_mr / var_m
    o0 = mean_r - s0 * mean_m

    # 计算残差，给大残差降权重
    residual = np.abs(s0 * mono_samp + o0 - colmap_samp)
    huber_delta = np.median(residual) * 1.345
    weight = np.where(residual < huber_delta, 1.0, huber_delta / residual)

    # 加权重新拟合
    sum_w = np.sum(weight)
    sum_wm = np.sum(weight * mono_samp)
    sum_wr = np.sum(weight * colmap_samp)
    sum_wmm = np.sum(weight * mono_samp * mono_samp)
    sum_wmr = np.sum(weight * mono_samp * colmap_samp)

    denom = sum_w * sum_wmm - sum_wm ** 2 + 1e-8
    scale = (sum_w * sum_wmr - sum_wm * sum_wr) / denom
    offset = (sum_wr - scale * sum_wm) / sum_w
    return scale, offset

# ===================== V5: L1 Median Regression (Minimize sum |s*zm+o - zr|) =====================
def compute_scale_offset_v5(colmap_samp, mono_samp, iterations=50):
    # 迭代L1回归，最小化绝对误差，极度抗外点
    scale, offset = compute_scale_offset_v3(colmap_samp, mono_samp)
    for _ in range(iterations):
        res = np.abs(scale * mono_samp + offset - colmap_samp)
        w = 1.0 / (res + 1e-8)
        sum_w = np.sum(w)
        sum_wm = np.sum(w * mono_samp)
        sum_wr = np.sum(w * colmap_samp)
        sum_wmm = np.sum(w * mono_samp ** 2)
        sum_wmr = np.sum(w * mono_samp * colmap_samp)
        denom = sum_w * sum_wmm - sum_wm ** 2 + 1e-8
        scale = (sum_w * sum_wmr - sum_wm * sum_wr) / denom
        offset = (sum_wr - scale * sum_wm) / sum_w
    return scale, offset

# ===================== V6: RANSAC + Inner LS 随机抽样剔除外点 =====================
def compute_scale_offset_v6(colmap_samp, mono_samp, ransac_iter=200, inlier_thresh=0.1):
    z_m = mono_samp
    z_r = colmap_samp
    n = len(z_m)
    best_inlier_count = 0
    best_s = 1.0
    best_o = 0.0

    for _ in range(ransac_iter):
        idx = np.random.choice(n, size=2, replace=False)
        m_sub = z_m[idx]
        r_sub = z_r[idx]
        dx = m_sub[1] - m_sub[0]
        if abs(dx) < 1e-8:
            continue
        s_tmp = (r_sub[1] - r_sub[0]) / dx
        o_tmp = r_sub[0] - s_tmp * m_sub[0]
        pred = s_tmp * z_m + o_tmp
        inlier_mask = np.abs(pred - z_r) < inlier_thresh
        cnt = np.sum(inlier_mask)
        if cnt > best_inlier_count:
            best_inlier_count = cnt
            best_s = s_tmp
            best_o = o_tmp

    # 用最优内点集重新最小二乘提升精度
    pred = best_s * z_m + best_o
    clean_mask = np.abs(pred - z_r) < inlier_thresh
    m_clean = z_m[clean_mask]
    r_clean = z_r[clean_mask]
    mean_m = np.mean(m_clean)
    mean_r = np.mean(r_clean)
    cov = np.mean((m_clean - mean_m) * (r_clean - mean_r))
    var = np.mean((m_clean - mean_m) ** 2) + 1e-8
    final_scale = cov / var
    final_offset = mean_r - final_scale * mean_m
    return final_scale, final_offset

def vec_compare(a, b):
    pred = a
    gt = b
    corr = np.corrcoef(pred, gt)[0,1]
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    dot = np.dot(pred, gt)
    norm_a = np.linalg.norm(pred)
    norm_b = np.linalg.norm(gt)
    cos = dot / (norm_a * norm_b + 1e-8)
    return corr, rmse, cos 

# ===================== 批量测试对比工具 =====================
def eval_all_methods(colmap_samp, mono_samp):
    methods = [
        ("v1 MAD", compute_scale_offset_v1),
        ("v2 MeanStd", compute_scale_offset_v2),
        ("v3 LS", compute_scale_offset_v3),
        ("v4 Huber-LS", compute_scale_offset_v4),
        ("v5 L1-Reg", compute_scale_offset_v5),
        ("v6 RANSAC-LS", compute_scale_offset_v6),
    ]
    result = {}
    for name, func in methods:
        s, o = func(colmap_samp, mono_samp)
        pred = s * mono_samp + o        
        
        corr = np.corrcoef(pred, colmap_samp)[0,1]
        rmse = np.sqrt(np.mean((pred - colmap_samp)**2))
        
        corr, rmse, cos = vec_compare(pred, colmap_samp)
        
        result[name] = {
            "scale": round(s,4),
            "offset": round(o,4),
            "corr": round(corr,3),
            "rmse": round(rmse,3), 
            "cos" : round(cos, 3)
        }
    return result