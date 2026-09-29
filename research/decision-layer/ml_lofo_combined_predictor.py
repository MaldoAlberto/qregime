"""
ml_lofo_combined_predictor.py -- responde la segunda peticion de mejora:
"incorpora predictores/metricas adicionales, posiblemente via ML, para
mejorar la capacidad predictiva."

CONTEXTO IMPORTANTE (por que esto no se habia hecho ya, honestamente): una
version anterior de este proyecto SI entreno un arbol de decision (85.1%)
y una regresion logistica (88.8%) sobre un dataset mas chico (62
instancias, 4 familias), pero con VALIDACION CRUZADA ORDINARIA de 5
pliegues -- no "leave-one-family-out" (LOFO). Cuando la regla de UNA sola
feature (spectral_sparsifiability) se re-verifico despues con LOFO real
sobre el dataset completo de 98 instancias / 9 familias, la cifra honesta
cayo a 70.4% -- exactamente igual al baseline trivial. Los modelos de ML
COMBINADOS nunca se re-evaluaron con ese mismo protocolo honesto. Esto lo
hace aqui.

METODO: combina TODAS las features estructurales ya calculadas para las
98 instancias (incluyendo el costo de contraccion de red tensorial, que
en el paper solo se uso en 9 puntos para una correlacion cruda -- aqui se
usa como feature de un clasificador real, para las 98 instancias
completas), entrena regresion logistica y un arbol de decision, y evalua
con LeaveOneGroupOut POR FAMILIA -- ni una instancia de la familia de
prueba fue vista en el entrenamiento de ese pliegue, el mismo estandar
que el resto de este proyecto exige.

SEGUNDA PRUEBA, mas dura aun: los mismos modelos, entrenados con las 98
instancias COMPLETAS, se evaluan contra el conjunto de 18 circuitos
GRANDES y nunca vistos de large_scale_generalization_test.py -- una
prueba de generalizacion a escala, no solo a familia.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

FEATURES = ["num_qubits", "num_gates", "avg_degree", "max_degree",
            "tcount_frac", "spectral_sparsifiability", "contraction_width", "log2_flops"]


def load_98():
    with open("circuit_dataset_with_tensor_network.json") as f:
        rows = json.load(f)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([1 if r["zx_tcount_reduction_pct"] > 0 else 0 for r in rows])
    groups = np.array([r["name"].split("_")[0] for r in rows])
    return X, y, groups, rows


def lofo_eval(model_ctor, X, y, groups):
    logo = LeaveOneGroupOut()
    fold_acc = []
    all_correct = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        model = model_ctor()
        model.fit(Xtr, y[train_idx])
        pred = model.predict(Xte)
        acc = (pred == y[test_idx]).mean()
        fold_acc.append((groups[test_idx][0], acc, len(test_idx)))
        all_correct += (pred == y[test_idx]).sum()
    pooled_acc = all_correct / len(y)
    return fold_acc, pooled_acc


def main():
    X, y, groups, rows = load_98()
    print(f"Dataset: {len(y)} instancias, {len(set(groups))} familias, "
          f"clase mayoritaria (ayuda={y.mean()*100:.1f}% de las veces)\n")

    print("=== Regresion logistica, LOFO real (LeaveOneGroupOut por familia) ===")
    fold_acc, pooled = lofo_eval(lambda: LogisticRegression(max_iter=2000), X, y, groups)
    for fam, acc, n in fold_acc:
        print(f"  familia de prueba = {fam:14s} n={n:3d}  precision={acc*100:5.1f}%")
    print(f"  -> precision LOFO agrupada (pooled): {pooled*100:.1f}%\n")

    print("=== Arbol de decision (profundidad<=3), LOFO real ===")
    fold_acc_tree, pooled_tree = lofo_eval(
        lambda: DecisionTreeClassifier(max_depth=3, random_state=0), X, y, groups)
    for fam, acc, n in fold_acc_tree:
        print(f"  familia de prueba = {fam:14s} n={n:3d}  precision={acc*100:5.1f}%")
    print(f"  -> precision LOFO agrupada (pooled): {pooled_tree*100:.1f}%\n")

    majority_baseline = max(y.mean(), 1 - y.mean())
    print(f"Baseline trivial (clase mayoritaria, sin LOFO -- limite superior optimista): {majority_baseline*100:.1f}%")
    print(f"Regla de una sola feature ya desplegada (spectral_sparsifiability, LOFO real, sesion anterior): 70.4%")

    # --- segunda prueba: generalizacion a los 18 circuitos GRANDES, nunca vistos ---
    print("\n=== Generalizacion a escala: modelos entrenados en las 98 instancias, "
          "evaluados en los 18 circuitos grandes nunca vistos ===")
    try:
        with open("circuit_dataset_large_scale_heldout.json") as f:
            large_rows = json.load(f)
        with open("large_scale_deployed_rule_results.json") as f:
            large_results = json.load(f)
    except FileNotFoundError:
        print("  (faltan los archivos del barrido a gran escala -- correr primero "
              "large_scale_generalization_test.py y large_scale_deployed_rule_check.py)")
        return

    # solo se puede evaluar con las features que SI se calcularon tambien
    # para el conjunto grande (no se corrio el costo de contraccion de red
    # tensorial ahi -- declarado como limite, cotengra es costoso a esta
    # escala y no se justifico el gasto solo para esta prueba adicional)
    reduced_features = ["num_qubits", "num_gates", "avg_degree", "max_degree", "tcount_frac"]
    Xr = np.array([[r[f] for f in FEATURES if f in reduced_features] for r in rows], dtype=float)
    idx = [FEATURES.index(f) for f in reduced_features]
    Xr = X[:, idx]
    scaler = StandardScaler().fit(Xr)
    logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(Xr), y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(scaler.transform(Xr), y)

    large_by_name = {r["name"]: r for r in large_rows}
    correct_log, correct_tree, n_eval = 0, 0, 0
    for row in large_results["rows"]:
        name = row["name"]
        lr = large_by_name.get(name)
        if lr is None:
            continue
        feats = np.array([[lr[f] for f in reduced_features]], dtype=float)
        actual = 1 if row["actual_helps"] else 0
        pred_log = int(logreg.predict(scaler.transform(feats))[0])
        pred_tree = int(tree.predict(scaler.transform(feats))[0])
        correct_log += (pred_log == actual)
        correct_tree += (pred_tree == actual)
        n_eval += 1
        print(f"  {name:20s}  logreg={pred_log}  arbol={pred_tree}  real={actual}")
    print(f"\n  Regresion logistica en circuitos grandes nunca vistos: {correct_log}/{n_eval} = {correct_log/n_eval*100:.1f}%")
    print(f"  Arbol de decision en circuitos grandes nunca vistos:    {correct_tree}/{n_eval} = {correct_tree/n_eval*100:.1f}%")
    print(f"  (para comparar: la regla YA DESPLEGADA obtuvo {large_results['accuracy']*100:.1f}% "
          f"en el mismo conjunto, ver large_scale_deployed_rule_check.py)")

    with open("ml_lofo_results.json", "w") as f:
        json.dump({
            "logreg_lofo_pooled": pooled, "tree_lofo_pooled": pooled_tree,
            "majority_baseline": majority_baseline,
            "logreg_large_scale_acc": correct_log / n_eval,
            "tree_large_scale_acc": correct_tree / n_eval,
            "deployed_rule_large_scale_acc": large_results["accuracy"],
            "n_98": len(y), "n_large": n_eval,
        }, f, indent=2)


if __name__ == "__main__":
    main()
