# -*- coding: utf-8 -*-
"""
Assignment 4: Bayes Decision Theory (1-D, 2 คลาส: '+' และ '-')

หลักการ (ดู g(x) ด้านล่าง):
- sigma_plus == sigma_minus  -> เส้นแบ่ง (decision boundary) เป็นเส้นตรง = LDA  (ข้อ 1, 4)
- sigma_plus != sigma_minus  -> เส้นแบ่งเป็นสมการกำลังสอง = QDA        (ข้อ 2, 3)

main() รันทุกเคสในตาราง CASES ให้อัตโนมัติ: LDA/QDA x prior เท่ากัน/ไม่เท่ากัน
x โหมด fixed/sampled  เพิ่มหรือแก้เคสได้ที่ CASES ที่เดียว
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# ===================== ตัวแปรที่ปรับได้ (แก้ตรงนี้ที่เดียว) =====================
MU_PLUS      = 3.0    # mean ของคลาส +
MU_MINUS     = -1.0   # mean ของคลาส -
SIGMA_PLUS   = 1.0    # sd ของคลาส +   (= sigma_minus -> เคส LDA, != -> เคส QDA)
SIGMA_MINUS  = 1.5    # sd ของคลาส -   (ใช้ในเคส QDA)
N_SAMPLES    = 500    # จำนวนตัวอย่างที่สุ่ม (โหมด sampled: ประมาณค่าพารามิเตอร์)
SEED         = 42
X_MIN, X_MAX = -10.0, 10.0    # กรอบกราฟตั้งต้น (ขยายอัตโนมัติให้ครอบทุกจุดตัด)

# (ชื่อเคส, sigma ของคลาส -, prior ของคลาส +)
CASES = [
    ("equal_var_LDA",           SIGMA_PLUS,  0.5),   # ข้อ 1, 4: sigma เท่ากัน, prior เท่ากัน
    ("unequal_var_QDA",         SIGMA_MINUS, 0.5),   # ข้อ 2, 3: sigma ไม่เท่ากัน, prior เท่ากัน
    ("equal_var_LDA_prior07",   SIGMA_PLUS,  0.7),   # ข้อ 1, 4 + prior ไม่เท่ากัน
    ("unequal_var_QDA_prior07", SIGMA_MINUS, 0.7),   # ข้อ 2, 3 + prior ไม่เท่ากัน
]
# ================================================================================

ZERO_A = 1e-12    # เกณฑ์เดียวที่ใช้ตัดสิน LDA (A = 0) vs QDA (A != 0) ทั้งไฟล์


def gaussian_pdf(x, mu, sigma):
    """likelihood: N(x; mu, sigma^2)"""
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def discriminant(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m):
    """g(x) = ln[pi_p * p(x|+)] - ln[pi_m * p(x|-)]   (ตัดสินใจ + เมื่อ g(x) > 0)"""
    return (np.log(pi_p / pi_m) + np.log(sigma_m / sigma_p)
            - (x - mu_p) ** 2 / (2 * sigma_p ** 2)
            + (x - mu_m) ** 2 / (2 * sigma_m ** 2))


def posterior(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m):
    """
    Bayes rule: posterior = prior * likelihood / sum
    คำนวณผ่าน g(x) เพื่อไม่ให้ 0/0 ตอน x ไกลจาก mean มาก (likelihood underflow ทั้งคู่)
    posterior_+ = 1 / (1 + e^-g(x))
    """
    g = np.clip(discriminant(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m), -500, 500)
    post_p = 1 / (1 + np.exp(-g))
    return post_p, 1 - post_p


def quadratic_coeffs(mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m):
    """
    g(x) = ln(pi_p/pi_m) + ln(sigma_m/sigma_p)
           - (x-mu_p)^2/(2*sigma_p^2) + (x-mu_m)^2/(2*sigma_m^2)
    เขียนใหม่เป็น A*x^2 + B*x + C = 0  (A = 0 -> LDA, A != 0 -> QDA)
    """
    A = 1 / (2 * sigma_m ** 2) - 1 / (2 * sigma_p ** 2)
    B = mu_p / sigma_p ** 2 - mu_m / sigma_m ** 2
    C = (np.log(pi_p / pi_m) + np.log(sigma_m / sigma_p)
         - mu_p ** 2 / (2 * sigma_p ** 2) + mu_m ** 2 / (2 * sigma_m ** 2))
    return A, B, C


def decision_boundary(A, B, C):
    """แก้ A*x^2 + B*x + C = 0 คืนรากจริงที่เรียงแล้ว"""
    if abs(A) < ZERO_A:                      # เคส LDA: เส้นตรง B*x + C = 0
        if B == 0:                           # mu เท่ากันด้วย -> ไม่มีขอบตัดสินใจ
            return []
        return [-C / B]
    roots = np.roots([A, B, C])              # เคส QDA: อาจได้ 2 ราก
    return sorted(r.real for r in roots if abs(r.imag) < 1e-9)


def estimate_gaussian(samples):
    """ประมาณ mean, sd จากตัวอย่างที่สุ่มมา"""
    return samples.mean(), samples.std(ddof=1)


def plot_range(boundaries):
    """ขยายกรอบกราฟให้ครอบทุกจุดตัด (เผื่อขอบ 10% ของช่วง)"""
    lo, hi = X_MIN, X_MAX
    if boundaries:
        lo, hi = min(lo, min(boundaries)), max(hi, max(boundaries))
    pad = 0.1 * (hi - lo)
    return lo - pad, hi + pad


def plot_case(mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m, boundaries, title, fname):
    x_lo, x_hi = plot_range(boundaries)
    x = np.linspace(x_lo, x_hi, 2000)
    like_p = gaussian_pdf(x, mu_p, sigma_p)
    like_m = gaussian_pdf(x, mu_m, sigma_m)
    post_p, post_m = posterior(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title)

    ax1.plot(x, like_p, label="likelihood +")
    ax1.plot(x, like_m, label="likelihood -")
    ax1.set_title("Likelihood")
    ax1.legend()

    ax2.plot(x, post_p, label="posterior +")
    ax2.plot(x, post_m, label="posterior -")
    ax2.set_title("Posterior")
    ax2.legend()

    # ใช้แกน Y ร่วมกันทั้งสองกราฟ: posterior สูงสุด 1 ส่วน likelihood อาจเกิน 1 ถ้า sigma เล็ก
    y_hi = max(1.05, max(like_p.max(), like_m.max()) * 1.15)

    for ax in (ax1, ax2):
        ax.set_ylim(0, y_hi)
        for b in boundaries:
            ax.axvline(b, color="k", linestyle="--", linewidth=1)
            ax.annotate(f"x*={b:.3f}", xy=(b, y_hi), xytext=(3, -12),
                        textcoords="offset points", fontsize=8, color="k")
        ax.set_xlim(x_lo, x_hi)

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def run(mode, sigma_minus_true, pi_plus, case_name):
    pi_m = 1 - pi_plus

    if mode == "fixed":
        mu_p, sigma_p, mu_m, sigma_m = MU_PLUS, SIGMA_PLUS, MU_MINUS, sigma_minus_true
    else:  # mode == "sampled": สุ่มตัวอย่างแล้วประมาณค่าพารามิเตอร์กลับมา
        rng = np.random.default_rng(SEED)
        samples_p = rng.normal(MU_PLUS, SIGMA_PLUS, N_SAMPLES)
        samples_m = rng.normal(MU_MINUS, sigma_minus_true, N_SAMPLES)
        mu_p, sigma_p = estimate_gaussian(samples_p)
        mu_m, sigma_m = estimate_gaussian(samples_m)

    A, B, C = quadratic_coeffs(mu_p, sigma_p, pi_plus, mu_m, sigma_m, pi_m)
    kind = "LDA (linear)" if abs(A) < ZERO_A else "QDA (quadratic)"
    boundaries = decision_boundary(A, B, C)

    roots_txt = ", ".join(f"{b:.4f}" for b in boundaries)
    print(f"[{case_name}/{mode}] {kind}: "
          f"mu_plus={mu_p:.3f}, sigma_plus={sigma_p:.3f}, "
          f"mu_minus={mu_m:.3f}, sigma_minus={sigma_m:.3f}, "
          f"pi_plus={pi_plus:.2f} | A={A:.4f}, B={B:.4f}, C={C:.4f} "
          f"-> decision boundary x* = [{roots_txt}]")

    plot_case(mu_p, sigma_p, pi_plus, mu_m, sigma_m, pi_m, boundaries,
              f"{kind} - {case_name} - {mode} params (pi+={pi_plus})",
              f"{case_name}_{mode}.png")


def main():
    for case_name, sigma_minus_true, pi_plus in CASES:
        run("fixed", sigma_minus_true, pi_plus, case_name)
        run("sampled", sigma_minus_true, pi_plus, case_name)


if __name__ == "__main__":
    main()
