"""
Ajusta el modelo analitico ESTANDAR de umbral de codigo de superficie
(Fowler et al. 2012):

    p_L(d) = A * (p_phys / p_th) ** ((d+1)/2)

pero en vez de usar los valores de A y p_th publicados en el paper (o los
que usa internamente Azure QRE, que no podemos auditar), se ajustan A y
p_th directamente a los 27 puntos que YA MEDIMOS con nuestra propia
simulacion Monte Carlo real (Stim + PyMatching, qec_surface_code_results.json).

Esto es lo que permite construir un estimador de recursos FTQC que no
depende de ningun numero de un vendor (ni Azure, ni NVIDIA): la formula es
publica (Fowler 2012), pero las CONSTANTES vienen de nuestros propios
datos, no de una tabla ajena.
"""
import json
import numpy as np
from scipy.optimize import curve_fit


def load_data(path="qec_surface_code_results.json"):
    rows = json.load(open(path))
    d = np.array([r["distance"] for r in rows], dtype=float)
    p = np.array([r["p_physical"] for r in rows], dtype=float)
    p_l = np.array([r["p_logical"] for r in rows], dtype=float)
    n = np.array([r["num_shots"] for r in rows], dtype=float)
    errs = np.array([r["num_logical_errors"] for r in rows], dtype=float)
    return d, p, p_l, n, errs


def model_log(X, log_A, p_th):
    d, p = X
    # log p_L = log A + (d+1)/2 * log(p / p_th)
    return log_A + (d + 1.0) / 2.0 * np.log(p / p_th)


def fit(verbose=True, p_max_for_fit=0.01):
    d, p, p_l, n, errs = load_data()

    # HALLAZGO HONESTO durante el ajuste: para p_physical >= ~0.02, p_logico
    # satura en ~0.5 para los 3 distancias (3, 5 y 7 dan practicamente el
    # mismo ~0.50) -- es el regimen de "moneda al aire" (el qubit logico ya
    # no retiene ninguna informacion), no una region donde la forma de ley
    # de potencia p_L = A*(p/p_th)^((d+1)/2) pueda aplicar. Un primer ajuste
    # ingenuo sobre los 27 puntos completos dio R^2=0.28 (muy malo) por
    # forzar esa saturacion dentro de un modelo que no la contempla. La
    # correccion, exactamente como se hace en la literatura (Fowler et al.
    # ajustan solo en el regimen sub-umbral donde el modelo es valido): usar
    # unicamente los puntos con p_physical <= p_max_for_fit para el ajuste.
    mask = (errs > 0) & (p <= p_max_for_fit)
    n_excluded_zero = int(((errs == 0)).sum())
    n_excluded_saturation = int((p > p_max_for_fit).sum())

    y = np.log(p_l[mask])
    X = (d[mask], p[mask])

    # pesos: propagacion de error (delta) de Var(log X) = Var(X)/X^2, con el
    # error estandar binomial que la propia simulacion ya reporto por punto
    all_rows = json.load(open("qec_surface_code_results.json"))
    stderr_all = np.array([r["p_logical_stderr"] for r in all_rows])
    sigma_log = stderr_all[mask] / p_l[mask]

    p0 = [np.log(0.1), 0.006]
    popt, pcov = curve_fit(model_log, X, y, p0=p0, sigma=sigma_log,
                            absolute_sigma=True, maxfev=20000)
    log_A_fit, p_th_fit = popt
    A_fit = np.exp(log_A_fit)
    perr = np.sqrt(np.diag(pcov))
    p_th_stderr = perr[1]

    # bondad de ajuste (en el subconjunto usado para ajustar)
    y_pred = model_log(X, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    result = {
        "A": A_fit, "p_th": p_th_fit, "p_th_stderr": p_th_stderr,
        "r2_log_fit": r2, "n_points_used": int(mask.sum()),
        "n_points_excluded_zero_errors": n_excluded_zero,
        "n_points_excluded_saturation_regime": n_excluded_saturation,
        "p_max_used_for_fit": p_max_for_fit,
    }

    if verbose:
        print("Ajuste del modelo de umbral p_L(d) = A * (p/p_th)^((d+1)/2)")
        print(f"  sobre nuestros propios {int(mask.sum())} puntos medidos (de 27 totales):")
        print(f"    - {n_excluded_zero} excluidos por 0 errores logicos observados")
        print(f"    - {n_excluded_saturation} excluidos por estar en el regimen de saturacion "
              f"(p_fisico > {p_max_for_fit*100:.1f}%, p_logico satura en ~0.50 sin importar d)")
        print(f"  A (ajustado)      = {A_fit:.4f}")
        print(f"  p_th (ajustado)   = {p_th_fit*100:.3f}%  +/- {p_th_stderr*100:.3f}pp")
        print(f"  R^2 (en escala log) = {r2:.4f}")
        print()
        print("  Comparacion: el cruce observado a ojo en el barrido fue ~0.5%-0.8%;")
        print(f"  el valor publicado (Fowler et al. 2012, ruido depolarizado) es ~1.0%.")
        print(f"  Nuestro ajuste ({p_th_fit*100:.2f}%) es propio, no tomado de ninguna de las dos fuentes.")

    json.dump(result, open("threshold_fit_result.json", "w"), indent=2)
    return result


if __name__ == "__main__":
    fit()
