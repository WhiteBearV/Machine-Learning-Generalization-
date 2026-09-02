# -*- coding: utf-8 -*-
"""
Assignment 4: Bayes Decision Theory (1-D, 2 คลาส: '+' และ '-')

หลักการ (ดู g(x) ด้านล่าง):
- sigma_plus == sigma_minus  -> เส้นแบ่ง (decision boundary) เป็นเส้นตรง = LDA  (ข้อ 1, 4)
- sigma_plus != sigma_minus  -> เส้นแบ่งเป็นสมการกำลังสอง = QDA        (ข้อ 2, 3)
เปลี่ยนโหมด LDA/QDA ได้แค่แก้ SIGMA_PLUS / SIGMA_MINUS ด้านล่างให้เท่ากัน/ไม่เท่ากัน
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# ===================== ตัวแปรที่ปรับได้ (แก้ตรงนี้ที่เดียว) =====================
MU_PLUS      = 3.0    # mean ของคลาส +
MU_MINUS     = -1.0   # mean ของคลาส -
SIGMA_PLUS   = 1.0    # sd ของคลาส +   (= SIGMA_MINUS -> เคส LDA, != -> เคส QDA)
SIGMA_MINUS  = 1.5    # sd ของคลาส -
PI_PLUS      = 0.5    # prior คลาส +  (PI_MINUS = 1 - PI_PLUS)
N_SAMPLES    = 500     # จำนวนตัวอย่างที่สุ่ม (โหมด b: ประมาณค่าพารามิเตอร์)
SEED         = 42
X_MIN, X_MAX = -10.0, 10.0
# ================================================================================


def gaussian_pdf(x, mu, sigma):
    """likelihood: N(x; mu, sigma^2)"""
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def posterior(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m):
    """Bayes rule: posterior = prior * likelihood / sum"""
    like_p = gaussian_pdf(x, mu_p, sigma_p) * pi_p
    like_m = gaussian_pdf(x, mu_m, sigma_m) * pi_m
    total = like_p + like_m
    return like_p / total, like_m / total


def decision_boundary(mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m):
    """
    g(x) = ln(pi_p/pi_m) + ln(sigma_m/sigma_p)
           - (x-mu_p)^2/(2*sigma_p^2) + (x-mu_m)^2/(2*sigma_m^2)
    เขียนใหม่เป็น A*x^2 + B*x + C = 0 แล้วแก้สมการ (A=0 -> LDA, A!=0 -> QDA)
    """
    A = 1 / (2 * sigma_m ** 2) - 1 / (2 * sigma_p ** 2)
    B = mu_p / sigma_p ** 2 - mu_m / sigma_m ** 2
    C = (np.log(pi_p / pi_m) + np.log(sigma_m / sigma_p)
         - mu_p ** 2 / (2 * sigma_p ** 2) + mu_m ** 2 / (2 * sigma_m ** 2))

    if abs(A) < 1e-12:                       # เคส LDA: เส้นตรง B*x + C = 0
        return [-C / B]
    roots = np.roots([A, B, C])              # เคส QDA: อาจได้ 2 ราก
    return sorted(r.real for r in roots if abs(r.imag) < 1e-9)


def estimate_gaussian(samples):
    """ประมาณ mean, sd จากตัวอย่างที่สุ่มมา"""
    return samples.mean(), samples.std(ddof=1)


def plot_case(mu_p, sigma_p, pi_p, mu_m, sigma_m, pi_m, boundaries, title, fname):
    x = np.linspace(X_MIN, X_MAX, 1000)
    like_p = gaussian_pdf(x, mu_p, sigma_p)
    like_m = gaussian_pdf(x, mu_m, sigma_m)
    post_p, post_m = posterior(x, mu_p, sigma_p, pi_p, mu_m, sigma_m, 1 - pi_p)

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

    for ax in (ax1, ax2):
        for b in boundaries:
            ax.axvline(b, color="k", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def run(mode):
    pi_m = 1 - PI_PLUS

    if mode == "fixed":
        mu_p, sigma_p, mu_m, sigma_m = MU_PLUS, SIGMA_PLUS, MU_MINUS, SIGMA_MINUS
    else:  # mode == "sampled": สุ่มตัวอย่างแล้วประมาณค่าพารามิเตอร์กลับมา
        rng = np.random.default_rng(SEED)
        samples_p = rng.normal(MU_PLUS, SIGMA_PLUS, N_SAMPLES)
        samples_m = rng.normal(MU_MINUS, SIGMA_MINUS, N_SAMPLES)
        mu_p, sigma_p = estimate_gaussian(samples_p)
        mu_m, sigma_m = estimate_gaussian(samples_m)

    kind = "LDA (linear)" if abs(sigma_p - sigma_m) < 1e-9 else "QDA (quadratic)"
    boundaries = decision_boundary(mu_p, sigma_p, PI_PLUS, mu_m, sigma_m, pi_m)

    print(f"[{mode}] {kind}: mu_plus={mu_p:.3f}, sigma_plus={sigma_p:.3f}, "
          f"mu_minus={mu_m:.3f}, sigma_minus={sigma_m:.3f} "
          f"-> decision boundary x* = {boundaries}")

    plot_case(mu_p, sigma_p, PI_PLUS, mu_m, sigma_m, pi_m, boundaries,
              f"{kind} - {mode} params", f"{mode}.png")


def main():
    run("fixed")
    run("sampled")


if __name__ == "__main__":
    main()
