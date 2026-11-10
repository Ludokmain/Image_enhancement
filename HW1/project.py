import numpy as np
import cv2
from src.degradation import degrade
from src.stretch import stretch_rgb_channels, stretch_luminance
from src.metrics import psnr, ssim
from src.equalization import equalize_lab, equalize_hsv
from src.personalized import denoise_then_stretch
import os
for d in ["degraded", "linearcontraststretch", "equalization", "personalmethod"]:
    os.makedirs(f"results/{d}", exist_ok=True)


########################baseline######################

original_cats= cv2.imread('original/badcats.png')
degraded_cats = degrade(original_cats)#baseline
cv2.imwrite('results/degraded/badcats_degraded.png', degraded_cats)
#cv2.imshow('original_cats', original_cats) #finestre che non mi servono dato che poi le ho messe side by side
#cv2.imshow('degraded_cats', degraded_cats)


original_girl= cv2.imread('original/girlinthesnow.png')
degraded_girl = degrade(original_girl)#baseline
cv2.imwrite('results/degraded/girlinthesnow_degraded.png', degraded_girl)
#cv2.imshow('original_girl', original_girl)
#cv2.imshow('degraded_girl', degraded_girl)


original_goat= cv2.imread('original/goat.png')
degraded_goat = degrade(original_goat)#baseline
cv2.imwrite('results/degraded/goat_degraded.png', degraded_goat)
#cv2.imshow('original_goat', original_goat)
#cv2.imshow('degraded_goat', degraded_goat)


original_desktop= cv2.imread('original/baddesktop.png')
degraded_desktop = degrade(original_desktop)#baseline
cv2.imwrite('results/degraded/baddesktop_degraded.png', degraded_desktop)
#cv2.imshow('original_desktop', original_desktop)
#cv2.imshow('degraded_desktop', degraded_desktop)


original_guy= cv2.imread('original/guyinthefog.png')
degraded_guy = degrade(original_guy)#baseline
cv2.imwrite('results/degraded/guyinthefog_degraded.png', degraded_guy)
#cv2.imshow('original_guy', original_guy)
#cv2.imshow('degraded_guy', degraded_guy)


original_schoolkids= cv2.imread('original/schoolkids.png')
degraded_schoolkids = degrade(original_schoolkids)#baseline
cv2.imwrite('results/degraded/schoolkids_degraded.png', degraded_schoolkids)
#cv2.imshow('original_schoolkids', original_schoolkids)
#cv2.imshow('degraded_schoolkids', degraded_schoolkids)


#####################linear contrast stretch#####################



# --- A) per canale ---
contrasted_cats = stretch_rgb_channels(degraded_cats)
#cv2.imshow('cats_linear_contrast_stretched', contrasted_cats)
# --- B) solo luminanza (YCrCb) ---
luminance_cats = stretch_luminance(degraded_cats)
#cv2.imshow("cats_linear_contrast_luminance", luminance_cats)
#######side by side#######
combo_cats = np.hstack([original_cats,degraded_cats, contrasted_cats, luminance_cats]) 
cv2.imshow("original_cats vs degraded_cats vs cats_linear_contrast_stretched vs cats_linear_contrast_luminance", combo_cats)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_cats.png', combo_cats)


contrasted_girl = stretch_rgb_channels(degraded_girl)
#cv2.imshow('girl_linear_contrast_stretched', contrasted_girl)
luminance_girl = stretch_luminance(degraded_girl)
#cv2.imshow("girl_linear_contrast_luminance", luminance_girl)
#######side by side#######
combo_girl = np.hstack([original_girl,degraded_girl, contrasted_girl, luminance_girl]) 
cv2.imshow("original_girl vs degraded_girl vs girl_linear_contrast_stretched vs girl_linear_contrast_luminance", combo_girl)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_girl.png', combo_girl)

contrasted_goat = stretch_rgb_channels(degraded_goat)
#cv2.imshow('goat_linear_contrast_stretched', contrasted_goat)
luminance_goat = stretch_luminance(degraded_goat)
#cv2.imshow("goat_linear_contrast_luminance", luminance_goat)
#######side by side#######
combo_goat = np.hstack([original_goat,degraded_goat, contrasted_goat, luminance_goat]) 
cv2.imshow("original_goat vs degraded_goat vs goat_linear_contrast_stretched vs goat_linear_contrast_luminance", combo_goat)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_goat.png', combo_goat)

contrasted_desktop = stretch_rgb_channels(degraded_desktop)
#cv2.imshow('desktop_linear_contrast_stretched', contrasted_desktop)
luminance_desktop = stretch_luminance(degraded_desktop)
#cv2.imshow("desktop_linear_contrast_luminance", luminance_desktop)
#######side by side#######
combo_desktop = np.hstack([original_desktop,degraded_desktop, contrasted_desktop, luminance_desktop])  
cv2.imshow("original_desktop vs degraded_desktop vs desktop_linear_contrast_stretched vs desktop_linear_contrast_luminance", combo_desktop)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_desktop.png', combo_desktop)

contrasted_guy = stretch_rgb_channels(degraded_guy)
#cv2.imshow('guy_linear_contrast_stretched', contrasted_guy)
luminance_guy = stretch_luminance(degraded_guy)
#cv2.imshow("guy_linear_contrast_luminance", luminance_guy)
#######side by side#######
combo_guy = np.hstack([original_guy,degraded_guy, contrasted_guy, luminance_guy]) 
cv2.imshow("original_guy vs degraded_guy vs guy_linear_contrast_stretched vs guy_linear_contrast_luminance", combo_guy)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_guy.png', combo_guy)

contrasted_schoolkids = stretch_rgb_channels(degraded_schoolkids)
#cv2.imshow('schoolkids_linear_contrast_stretched', contrasted_schoolkids)
luminance_schoolkids = stretch_luminance(degraded_schoolkids)
#cv2.imshow("schoolkids_linear_contrast_luminance", luminance_schoolkids)
#######side by side#######
combo_schoolkids = np.hstack([original_schoolkids,degraded_schoolkids, contrasted_schoolkids, luminance_schoolkids]) 
cv2.imshow("original_schoolkids vs degraded_schoolkids vs schoolkids_linear_contrast_stretched vs schoolkids_linear_contrast_luminance", combo_schoolkids)
cv2.imwrite('results/linearcontraststretch/linear_contrast_stretched_combo_schoolkids.png', combo_schoolkids)

######################################################
equalized_cats_lab =  equalize_lab(degraded_cats)
equalized_cats_hsv = equalize_hsv(degraded_cats)
combo_equalized_cats = np.hstack([original_cats,degraded_cats, equalized_cats_lab, equalized_cats_hsv]) 
cv2.imshow("original_cats vs degraded_cats vs equalized_cats_lab vs equalized_cats_hsv", combo_equalized_cats)
cv2.imwrite('results/equalization/equalized_combo_cats.png', combo_equalized_cats)

equalized_girl_lab =  equalize_lab(degraded_girl)
equalized_girl_hsv = equalize_hsv(degraded_girl)
combo_equalized_girl = np.hstack([original_girl,degraded_girl, equalized_girl_lab, equalized_girl_hsv]) 
cv2.imshow("original_girl vs degraded_girl vs equalized_girl_lab vs equalized_girl_hsv", combo_equalized_girl)
cv2.imwrite('results/equalization/equalized_combo_girl.png', combo_equalized_girl)

equalized_goat_lab =  equalize_lab(degraded_goat)
equalized_goat_hsv = equalize_hsv(degraded_goat)
combo_equalized_goat = np.hstack([original_goat,degraded_goat, equalized_goat_lab, equalized_goat_hsv]) 
cv2.imshow("original_goat vs degraded_goat vs equalized_goat_lab vs equalized_goat_hsv", combo_equalized_goat)
cv2.imwrite('results/equalization/equalized_combo_goat.png', combo_equalized_goat)

equalized_desktop_lab =  equalize_lab(degraded_desktop)
equalized_desktop_hsv = equalize_hsv(degraded_desktop)
combo_equalized_desktop = np.hstack([original_desktop,degraded_desktop, equalized_desktop_lab, equalized_desktop_hsv]) 
cv2.imshow("original_desktop vs degraded_desktop vs equalized_desktop_lab vs equalized_desktop_hsv", combo_equalized_desktop)
cv2.imwrite('results/equalization/equalized_combo_desktop.png', combo_equalized_desktop)

equalized_guy_lab =  equalize_lab(degraded_guy)
equalized_guy_hsv = equalize_hsv(degraded_guy)
combo_equalized_guy = np.hstack([original_guy,degraded_guy, equalized_guy_lab, equalized_guy_hsv]) 
cv2.imshow("original_guy vs degraded_guy vs equalized_guy_lab vs equalized_guy_hsv", combo_equalized_guy)
cv2.imwrite('results/equalization/equalized_combo_guy.png', combo_equalized_guy)

equalized_schoolkids_lab =  equalize_lab(degraded_schoolkids)
equalized_schoolkids_hsv = equalize_hsv(degraded_schoolkids)
combo_equalized_schoolkids = np.hstack([original_schoolkids,degraded_schoolkids, equalized_schoolkids_lab, equalized_schoolkids_hsv]) 
cv2.imshow("original_schoolkids vs degraded_schoolkids vs equalized_schoolkids_lab vs equalized_schoolkids_hsv", combo_equalized_schoolkids)
cv2.imwrite('results/equalization/equalized_combo_schoolkids.png', combo_equalized_schoolkids)

############################personalizzato################################

denoise_then_stretch_cats = denoise_then_stretch(degraded_cats)
#cv2.imshow('denoised_then_stretched_cats', denoise_then_stretch_cats)
combo_denoised_then_stretch_cats = np.hstack([original_cats,degraded_cats, denoise_then_stretch_cats])
cv2.imshow("original_cats vs degraded_cats vs denoise_then_stretch_cats", combo_denoised_then_stretch_cats)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_cats.png', combo_denoised_then_stretch_cats)

denoise_then_stretch_girl = denoise_then_stretch(degraded_girl)
#cv2.imshow('denoised_then_stretch_girl', denoise_then_stretch_girl)
combo_denoised_then_stretch_girl = np.hstack([original_girl,degraded_girl, denoise_then_stretch_girl])
cv2.imshow("original_girl vs degraded_girl vs denoise_then_stretch_girl", combo_denoised_then_stretch_girl)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_girl.png', combo_denoised_then_stretch_girl)

denoise_then_stretch_goat = denoise_then_stretch(degraded_goat)
#cv2.imshow('denoised_then_stretch_goat', denoise_then_stretch_goat)
combo_denoised_then_stretch_goat = np.hstack([original_goat,degraded_goat, denoise_then_stretch_goat])
cv2.imshow("original_goat vs degraded_goat vs denoise_then_stretch_goat", combo_denoised_then_stretch_goat)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_goat.png', combo_denoised_then_stretch_goat)   

denoise_then_stretch_desktop = denoise_then_stretch(degraded_desktop)
#cv2.imshow('denoised_then_stretch_desktop', denoise_then_stretch_desktop)
combo_denoised_then_stretch_desktop = np.hstack([original_desktop,degraded_desktop, denoise_then_stretch_desktop])
cv2.imshow("original_desktop vs degraded_desktop vs denoise_then_stretch_desktop", combo_denoised_then_stretch_desktop)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_desktop.png', combo_denoised_then_stretch_desktop)

denoise_then_stretch_guy = denoise_then_stretch(degraded_guy)
#cv2.imshow('denoised_then_stretch_guy', denoise_then_stretch_guy)
combo_denoised_then_stretch_guy = np.hstack([original_guy,degraded_guy, denoise_then_stretch_guy])
cv2.imshow("original_guy vs degraded_guy vs denoise_then_stretch_guy", combo_denoised_then_stretch_guy)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_guy.png', combo_denoised_then_stretch_guy)   

denoise_then_stretch_schoolkids = denoise_then_stretch(degraded_schoolkids)
#cv2.imshow('denoised_then_stretch_schoolkids', denoise_then_stretch_schoolkids)
combo_denoised_then_stretch_schoolkids = np.hstack([original_schoolkids,degraded_schoolkids, denoise_then_stretch_schoolkids])
cv2.imshow("original_schoolkids vs degraded_schoolkids vs denoise_then_stretch_schoolkids", combo_denoised_then_stretch_schoolkids)
cv2.imwrite('results/personalmethod/denoised_then_stretch_combo_schoolkids.png', combo_denoised_then_stretch_schoolkids) 



##################### METRICS TABLE #####################
# What it does:
#  1. computes PSNR and SSIM for EVERY image x EVERY method
#  2. prints a readable table in the terminal
#  3. saves results/metrics.csv, results/metrics_table.tex and results/metrics_tables.pdf 
# Every metric is computed against the original image.

import os
import csv

# Column names shown in the table (same order as the tuples below)
method_names = [
    "Baseline",
    "Stretch RGB",
    "Stretch Y",
    "HE Lab",
    "HE HSV",
    "Denoise+Stretch",
]

# image name: original, then one result per method, same order as method_names
images = {
    "badcats":      (original_cats,       degraded_cats,       contrasted_cats,       luminance_cats,       equalized_cats_lab,       equalized_cats_hsv,       denoise_then_stretch_cats),
    "girlinthesnow":(original_girl,       degraded_girl,       contrasted_girl,       luminance_girl,       equalized_girl_lab,       equalized_girl_hsv,       denoise_then_stretch_girl),
    "goat":         (original_goat,       degraded_goat,       contrasted_goat,       luminance_goat,       equalized_goat_lab,       equalized_goat_hsv,       denoise_then_stretch_goat),
    "baddesktop":   (original_desktop,    degraded_desktop,    contrasted_desktop,    luminance_desktop,    equalized_desktop_lab,    equalized_desktop_hsv,    denoise_then_stretch_desktop),
    "guyinthefog":  (original_guy,        degraded_guy,        contrasted_guy,        luminance_guy,        equalized_guy_lab,        equalized_guy_hsv,        denoise_then_stretch_guy),
    "schoolkids":   (original_schoolkids, degraded_schoolkids, contrasted_schoolkids, luminance_schoolkids, equalized_schoolkids_lab, equalized_schoolkids_hsv, denoise_then_stretch_schoolkids),
}

# ---------- 1) compute all the metrics ----------
table = {}
rows = []  # flat list, used for the CSV
for img_name, (orig, *results) in images.items():
    table[img_name] = {}
    for method, result in zip(method_names, results):
        p = float(psnr(orig, result))
        s = float(ssim(orig, result))
        table[img_name][method] = {"PSNR": p, "SSIM": s}
        rows.append({"image": img_name, "method": method, "PSNR_dB": round(p, 2), "SSIM": round(s, 4)})

# mean of each method over all images (useful last row of the table)
mean = {m: {k: sum(table[i][m][k] for i in images) / len(images) for k in ("PSNR", "SSIM")}
        for m in method_names}

# ---------- 2) print in the terminal ----------
for metric, fmt in (("PSNR", ".2f"), ("SSIM", ".4f")):
    print(f"\n{metric}" + (" [dB]" if metric == "PSNR" else ""))
    print(f"{'image':<15}" + "".join(f"{m:>17}" for m in method_names))
    for img_name in images:
        print(f"{img_name:<15}" + "".join(f"{table[img_name][m][metric]:>17{fmt}}" for m in method_names))
    print(f"{'MEAN':<15}" + "".join(f"{mean[m][metric]:>17{fmt}}" for m in method_names))

# ---------- 3) save the CSV ----------
os.makedirs("results", exist_ok=True)
with open("results/metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "method", "PSNR_dB", "SSIM"])
    writer.writeheader()
    writer.writerows(rows)

# ---------- 4) save LaTeX tables ----------
def fmt_cell(img_name, method, best_p, best_s):
    p = table[img_name][method]["PSNR"] if img_name != "MEAN" else mean[method]["PSNR"]
    s = table[img_name][method]["SSIM"] if img_name != "MEAN" else mean[method]["SSIM"]
    p_txt = f"\\textbf{{{p:.2f}}}" if p == best_p else f"{p:.2f}"
    s_txt = f"\\textbf{{{s:.3f}}}" if s == best_s else f"{s:.3f}"
    return f"{p_txt} / {s_txt}"

lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{l" + "c" * len(method_names) + "}",
    r"\hline",
    "Image & " + " & ".join(method_names) + r" \\",
    r"\hline",
]
for img_name in list(images) + ["MEAN"]:
    src = mean if img_name == "MEAN" else table[img_name]
    best_p = max(src[m]["PSNR"] for m in method_names)
    best_s = max(src[m]["SSIM"] for m in method_names)
    if img_name == "MEAN":
        lines.append(r"\hline")
    lines.append(img_name + " & " + " & ".join(fmt_cell(img_name, m, best_p, best_s) for m in method_names) + r" \\")
lines += [
    r"\hline",
    r"\end{tabular}",
    r"\caption{PSNR [dB] / SSIM of each method with respect to the original image. Best value per row in bold.}",
    r"\label{tab:metrics}",
    r"\end{table}",
]
with open("results/metrics_table.tex", "w") as f:
    f.write("\n".join(lines))

print("\nSaved results/metrics.csv and results/metrics_table.tex")



# ---------- 5) save the tables in a PDF ----------
# matplotlib is used only to draw the tables.
# The best value of each row is in bold on a green background.
import matplotlib
matplotlib.use("Agg")                      # draws without opening windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def draw_table(pdf, metric, fmt, title):
    row_names = list(images) + ["MEAN"]
    cells = []
    for img_name in row_names:
        src = mean if img_name == "MEAN" else table[img_name]
        cells.append([format(src[m][metric], fmt) for m in method_names])

    fig, ax = plt.subplots(figsize=(11.69, 4))   # A4 landscape width
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    t = ax.table(cellText=cells, rowLabels=row_names, colLabels=method_names,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)    

    for (r, c), cell in t.get_celld().items():
        if r == 0 or c == -1:                    # header row / row labels
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e6e6e6")
    # highlight the best method of each row 
    for r, row in enumerate(cells, start=1):
        values = [float(v) for v in row]
        best_c = values.index(max(values))
        t[r, best_c].set_text_props(fontweight="bold")
        t[r, best_c].set_facecolor("#cdeccd")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

with PdfPages("results/metrics_tables.pdf") as pdf:
    draw_table(pdf, "PSNR", ".2f", "PSNR [dB] w.r.t. the original (higher is better)")
    draw_table(pdf, "SSIM", ".4f", "SSIM w.r.t. the original (higher is better, max 1)")

print("\nSaved results/metrics.csv, results/metrics_table.tex and results/metrics_tables.pdf")




cv2.waitKey(0)
cv2.destroyAllWindows()