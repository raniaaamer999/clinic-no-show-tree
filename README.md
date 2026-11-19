# Clinic no-show decision tree

This project fits one classification tree that predicts whether a clinic appointment will be a no-show, scores that tree on held-out appointments, and turns a predicted no-show into an overbook decision. `clinic_appointments.csv` is the input and is not modified.

## Setup

From this folder, use the existing environment:

```bash
source .venv/bin/activate
python analysis.py
```

To recreate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`. On Windows, use `.venv\Scripts\activate`.

The script uses pandas, scikit-learn, and matplotlib.

## What the script does

- Checks that the CSV has 400 rows, 8 columns, 91 no-shows, and no missing values. If a check fails, it prints the difference and continues.
- Treats `no-show` as the positive class and `checked-in` as the negative class. All seven appointment features are offered to the tree.
- Holds out 30% of the rows with a stratified split and random seed 42 (280 training appointments, 120 test appointments). The tree is fit on the training rows only.
- Fits one CART tree, called v1, with Gini impurity and `max_depth=3`. Depth 3 caps the tree at eight leaves so it stays readable and does not grow until every training leaf is pure. That depth is fixed before the test appointments are scored.
- On the test appointments, prints the baseline and v1 confusion matrices in TN, FP, FN, TP order, plus accuracy, precision for no-shows, and recall for no-shows. The baseline predicts `checked-in` for every appointment. v1 training accuracy is printed next to v1 test accuracy.
- Saves a labeled diagram as `decision_tree.png` and prints every root-to-leaf rule. Node ids in the printout match the diagram.
- Applies one decision function to upcoming appointments D and E: `overbook` when v1 predicts `no-show`, otherwise `hold`. For each appointment it prints the path, the leaf, the predicted class, and the decision.
- Values the test confusion matrices with the assignment costs: TN = $0, TP = $0, FP = −$60, FN = −$120. It prints total and per-appointment value for the baseline and for v1.
- Scales the per-appointment value to a year using `APPOINTMENTS_PER_YEAR = 10_000` at the top of `analysis.py`. That volume is an assumption, not a count from the CSV.

Running `python analysis.py` prints the checks, the tree, the test metrics, the decisions for D and E, and the value figures. Those printed numbers are the results to cite.
