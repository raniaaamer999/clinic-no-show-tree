"""Clinic no-show decision tree and overbooking rule.

Fits one CART classification tree on a stratified training split of
clinic_appointments.csv and scores it on the held-out appointments.
The test rows are used only for evaluation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "clinic_appointments.csv"
TREE_PLOT_PATH = ROOT / "decision_tree.png"

RANDOM_SEED = 42
TEST_SIZE = 0.30
# Shallow on purpose. See the printed depth note in main().
MAX_DEPTH = 3

POSITIVE = "no-show"
NEGATIVE = "checked-in"
CLASS_LABELS = [NEGATIVE, POSITIVE]

FEATURE_COLUMNS = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]
TARGET_COLUMN = "no_show"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Assignment cost table. Positive class is no-show.
# TN: hold a slot the patient attends.
# FP: overbook a slot the patient attends (overtime and a refund).
# FN: hold a slot the patient misses (empty slot).
# TP: overbook a slot the patient misses (the second patient fills it).
VALUE_TN = 0
VALUE_FP = -60
VALUE_FN = -120
VALUE_TP = 0

# Editable scaling assumption. This volume is not in the CSV.
APPOINTMENTS_PER_YEAR = 10_000

APPOINTMENT_D = {
    "lead_time_days": 21,
    "prior_no_shows": 1,
    "age": 27,
    "distance_miles": 9.5,
    "sms_reminder": 0,
    "new_patient": 1,
    "appointment_hour": 15,
}
APPOINTMENT_E = {
    "lead_time_days": 3,
    "prior_no_shows": 0,
    "age": 61,
    "distance_miles": 2.0,
    "sms_reminder": 1,
    "new_patient": 0,
    "appointment_hour": 9,
}

DISPLAY_NAMES = [
    "lead time (days)",
    "prior no-shows",
    "age",
    "distance (miles)",
    "SMS reminder",
    "new patient",
    "appointment hour",
]


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def check_appointments(frame: pd.DataFrame) -> None:
    """Compare the file with the assignment's data checks, then continue."""
    actual = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "no_shows": int((frame[TARGET_COLUMN] == POSITIVE).sum()) if TARGET_COLUMN in frame.columns else None,
        "missing_values": int(frame.isna().sum().sum()),
    }
    expected = {
        "rows": 400,
        "columns": 8,
        "no_shows": 91,
        "missing_values": 0,
    }
    print(f"File: {DATA_PATH.name}")
    print(f"Columns: {', '.join(map(str, frame.columns))}")
    for name in expected:
        print(f"{name}: expected {expected[name]}, found {actual[name]}")
    if actual["rows"]:
        no_show_rate = actual["no_shows"] / actual["rows"]
        print(f"No-show rate: {actual['no_shows']} / {actual['rows']} = {no_show_rate:.1%}")

    differences = []
    for name in expected:
        if actual[name] != expected[name]:
            differences.append(f"{name}: expected {expected[name]}, found {actual[name]}")
    if list(frame.columns) != EXPECTED_COLUMNS:
        differences.append(
            "column names or order: expected "
            + ", ".join(EXPECTED_COLUMNS)
            + "; found "
            + ", ".join(map(str, frame.columns))
        )
    if TARGET_COLUMN in frame.columns:
        labels = sorted(frame[TARGET_COLUMN].dropna().astype(str).unique())
        unexpected = [label for label in labels if label not in CLASS_LABELS]
        print(f"Outcome labels: {', '.join(labels) if labels else '(none)'}")
        if unexpected:
            differences.append(
                "unexpected outcome labels: " + ", ".join(unexpected)
            )
    if differences:
        print("Differences:")
        for item in differences:
            print(f"  - {item}")
        print("Continuing with the file as it is.")
    else:
        print("All checks passed.")

    missing_columns = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise SystemExit(
            "Cannot fit the tree. Missing columns: " + ", ".join(missing_columns)
        )


def counts_in_order(actual, predicted) -> tuple[int, int, int, int]:
    """Return TN, FP, FN, TP with checked-in negative and no-show positive."""
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return tn, fp, fn, tp


def report_predictions(name: str, actual, predicted) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = counts_in_order(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    print(name)
    print(f"  Confusion matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")
    print(f"  Accuracy: {accuracy:.4f}")
    if tp + fp == 0:
        print("  Precision for no-shows: undefined (no predicted no-shows)")
    else:
        precision = precision_score(actual, predicted, pos_label=POSITIVE, zero_division=0)
        print(f"  Precision for no-shows: {precision:.4f}  ({tp} / {tp + fp})")
    if tp + fn == 0:
        print("  Recall for no-shows: undefined (no actual no-shows)")
    else:
        recall = recall_score(actual, predicted, pos_label=POSITIVE, zero_division=0)
        print(f"  Recall for no-shows: {recall:.4f}  ({tp} / {tp + fn})")
    return tn, fp, fn, tp


def assignment_value(tn: int, fp: int, fn: int, tp: int) -> float:
    return tn * VALUE_TN + fp * VALUE_FP + fn * VALUE_FN + tp * VALUE_TP


def report_value(name: str, tn: int, fp: int, fn: int, tp: int, n_appointments: int) -> float:
    total = assignment_value(tn, fp, fn, tp)
    per_appointment = total / n_appointments
    annual = per_appointment * APPOINTMENTS_PER_YEAR
    print(name)
    print(
        f"  Value = ({VALUE_TN})*TN + ({VALUE_FP})*FP + ({VALUE_FN})*FN + ({VALUE_TP})*TP"
    )
    print(
        f"        = ({VALUE_TN})*{tn} + ({VALUE_FP})*{fp} + ({VALUE_FN})*{fn} + ({VALUE_TP})*{tp}"
    )
    print(f"  Total value on the test appointments: {money(total)}")
    print(f"  Value per test appointment: {money(per_appointment)}")
    print(
        f"  Annual value at {APPOINTMENTS_PER_YEAR:,} appointments/year: {money(annual)}"
    )
    return total


def class_count_text(model, node: int) -> str:
    """Return training class counts. tree_.value stores class proportions."""
    proportions = model.tree_.value[node][0]
    weight = float(model.tree_.weighted_n_node_samples[node])
    classes = [str(label) for label in model.classes_]
    counts = [float(proportion) * weight for proportion in proportions]
    return ", ".join(f"{classes[i]} {counts[i]:.0f}" for i in range(len(classes)))


def root_to_leaf_rules(model) -> list[str]:
    tree = model.tree_
    rules = []

    def walk(node: int, conditions: list[str]) -> None:
        if tree.children_left[node] == -1:
            condition_text = " AND ".join(conditions) if conditions else "all training appointments"
            predicted = str(model.classes_[int(tree.value[node][0].argmax())])
            rules.append(
                f"Leaf node {node}: IF {condition_text} THEN predict {predicted} "
                f"(training counts: {class_count_text(model, node)}; "
                f"samples {int(tree.n_node_samples[node])})"
            )
            return
        name = DISPLAY_NAMES[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        walk(tree.children_left[node], conditions + [f"{name} <= {threshold:.2f}"])
        walk(tree.children_right[node], conditions + [f"{name} > {threshold:.2f}"])

    walk(0, [])
    return rules


def features_used(model) -> list[str]:
    tree = model.tree_
    used = []
    for node in range(tree.node_count):
        feature = int(tree.feature[node])
        if feature >= 0:
            name = DISPLAY_NAMES[feature]
            if name not in used:
                used.append(name)
    return used


def make_slot_decision(model):
    """Return overbook when v1 predicts no-show, otherwise hold."""

    def slot_decision(appointment: dict) -> str:
        missing = [column for column in FEATURE_COLUMNS if column not in appointment]
        if missing:
            raise KeyError("Appointment is missing features: " + ", ".join(missing))
        row = pd.DataFrame([appointment], columns=FEATURE_COLUMNS)
        predicted = str(model.predict(row)[0])
        if predicted == POSITIVE:
            return "overbook"
        if predicted == NEGATIVE:
            return "hold"
        raise ValueError(f"Unexpected predicted class: {predicted}")

    return slot_decision


def describe_path(model, appointment: dict) -> tuple[int, str]:
    """Print the root-to-leaf path and return the leaf id and predicted class."""
    tree = model.tree_
    node = 0
    row = pd.DataFrame([appointment], columns=FEATURE_COLUMNS)
    while tree.children_left[node] != -1:
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        value = float(row.iloc[0, feature])
        if value <= threshold:
            child = int(tree.children_left[node])
            relation = "<="
        else:
            child = int(tree.children_right[node])
            relation = ">"
        print(
            f"  node {node}: {DISPLAY_NAMES[feature]} = {value:g} {relation} {threshold:.2f} "
            f"-> node {child}"
        )
        node = child
    predicted = str(model.predict(row)[0])
    print(f"  leaf node {node}: training counts {class_count_text(model, node)}")
    print(f"  predicted class: {predicted}")
    return node, predicted


def save_tree_diagram(model, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=DISPLAY_NAMES,
        class_names=[str(label) for label in model.classes_],
        filled=True,
        rounded=True,
        impurity=True,
        node_ids=True,
        proportion=False,
        precision=2,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        "v1 CART classification tree for clinic no-shows\n"
        "Gini impurity, max depth 3, fit on the training appointments only\n"
        "Left branch is <= the threshold. value is [checked-in count, no-show count].",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    appointments = pd.read_csv(DATA_PATH)

    section("1. Data check")
    check_appointments(appointments)
    print(f"Positive class: {POSITIVE}")
    print(f"Negative class: {NEGATIVE}")
    print("Features offered to the tree: " + ", ".join(FEATURE_COLUMNS))

    section("2. Stratified train/test split")
    features = appointments[FEATURE_COLUMNS]
    outcome = appointments[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        outcome,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=outcome,
    )
    if len(x_train) + len(x_test) != len(appointments):
        raise SystemExit("Train and test splits do not cover every appointment.")
    if set(x_train.index) & set(x_test.index):
        raise SystemExit("Train and test splits overlap.")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"test_size: {TEST_SIZE:.0%}, stratified on {TARGET_COLUMN}")
    print(f"Training appointments: {len(x_train)}")
    print(f"Test appointments: {len(x_test)}")
    print(
        "Training no-shows: "
        f"{int((y_train == POSITIVE).sum())} ({(y_train == POSITIVE).mean():.1%})"
    )
    print(
        "Test no-shows: "
        f"{int((y_test == POSITIVE).sum())} ({(y_test == POSITIVE).mean():.1%})"
    )
    print("The tree is fit on the training appointments only.")

    section("3. v1 classification tree")
    print(
        f"Chosen depth: {MAX_DEPTH}. "
        "The training split has 280 appointments. "
        "With no depth limit, CART can keep splitting until leaves are pure "
        "and memorize training rows. Depth 3 allows at most eight leaves, "
        "so the rule stays readable on one diagram. "
        "This depth was set before any test-set metric was calculated. "
        "The test appointments are not used to choose features or depth."
    )
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    if int(model.tree_.n_node_samples[0]) != len(y_train):
        raise SystemExit("The tree was not fit on the training rows alone.")
    if model.get_depth() > MAX_DEPTH:
        raise SystemExit("The fitted tree is deeper than the chosen limit.")

    used = features_used(model)
    unused = [name for name in DISPLAY_NAMES if name not in used]
    root_feature = DISPLAY_NAMES[int(model.tree_.feature[0])]
    root_threshold = float(model.tree_.threshold[0])
    leaf_nodes = [
        node
        for node in range(model.tree_.node_count)
        if model.tree_.children_left[node] == -1
    ]
    smallest = min(leaf_nodes, key=lambda node: int(model.tree_.n_node_samples[node]))
    smallest_n = int(model.tree_.n_node_samples[smallest])
    print(f"Criterion: gini. max_depth: {MAX_DEPTH}. Fitted depth: {model.get_depth()}.")
    print(f"Leaves: {model.get_n_leaves()}.")
    print(f"Root split: {root_feature} <= {root_threshold:.2f} goes left.")
    print("Features the tree splits on, from the root outward: " + ", ".join(used))
    print("Features offered but unused at this depth: " + ", ".join(unused))
    print(f"Smallest leaf is node {smallest}, with {smallest_n} training appointment(s).")
    if smallest_n == 1:
        print(
            "That leaf rests on a single training appointment, so it is a weak "
            "basis for a standing clinic policy."
        )
    print(
        f"For the clinic, start at the root split on {root_feature}. "
        f"The fitted tree uses {', '.join(used)}. "
        f"It does not use {', '.join(unused)}, so those features do not change "
        f"a depth-{MAX_DEPTH} prediction."
    )

    section("4. Tree rules from root to leaves")
    print(
        "Integer features are split at a .50 threshold. "
        "prior no-shows <= 2.50 means 0, 1, or 2 prior misses. "
        "SMS reminder <= 0.50 means no reminder was sent."
    )
    print()
    print(export_text(model, feature_names=DISPLAY_NAMES, decimals=2, show_weights=True))
    for rule in root_to_leaf_rules(model):
        print(rule)

    section("5. Tree diagram")
    save_tree_diagram(model, TREE_PLOT_PATH)
    print(f"Saved {TREE_PLOT_PATH.name}")
    print("Node ids on the diagram match the node ids in the rules and the paths below.")

    section("6. Test-set evaluation")
    print("Baseline predicts checked-in for every test appointment.")
    baseline_pred = pd.Series(NEGATIVE, index=y_test.index)
    v1_test_pred = model.predict(x_test)
    v1_train_pred = model.predict(x_train)
    baseline_counts = report_predictions("Baseline", y_test, baseline_pred)
    print()
    v1_counts = report_predictions("v1", y_test, v1_test_pred)
    train_accuracy = accuracy_score(y_train, v1_train_pred)
    test_accuracy = accuracy_score(y_test, v1_test_pred)
    print(
        f"v1 training accuracy: {train_accuracy:.4f} | "
        f"v1 test accuracy: {test_accuracy:.4f}"
    )
    gap = train_accuracy - test_accuracy
    print(f"Training minus test accuracy: {gap:+.4f}")
    print(f"The two accuracies differ by {abs(gap) * 100:.1f} percentage points.")
    if gap > 0.05:
        print(
            "Training accuracy is higher by more than 5 percentage points, "
            "the pattern that shows a tree memorizing training rows."
        )
    print("These scores use the held-out test appointments. The tree was not refit on them.")

    section("7. Decisions for appointments D and E")
    slot_decision = make_slot_decision(model)
    for label, appointment in ("D", APPOINTMENT_D), ("E", APPOINTMENT_E):
        print(f"Appointment {label}")
        feature_text = ", ".join(f"{column}={appointment[column]:g}" for column in FEATURE_COLUMNS)
        print(f"  features: {feature_text}")
        _leaf, predicted = describe_path(model, appointment)
        decision = slot_decision(appointment)
        print(f"  decision: {decision}")
        expected = "overbook" if predicted == POSITIVE else "hold"
        if decision != expected:
            raise SystemExit(f"Appointment {label}: decision does not match the predicted class.")
        print()

    section("8. Value on the test appointments")
    print("Cost table from the assignment, per appointment:")
    print(f"  TN hold and checked-in: {money(VALUE_TN)}")
    print(f"  FP overbook and checked-in: {money(VALUE_FP)}")
    print(f"  FN hold and no-show: {money(VALUE_FN)}")
    print(f"  TP overbook and no-show: {money(VALUE_TP)}")
    print(
        f"Annual scaling uses {APPOINTMENTS_PER_YEAR:,} appointments per year. "
        "This volume is an assumption, not a fact from clinic_appointments.csv. "
        "Annual value = (test total / test appointments) x appointments per year, "
        "which assumes a year looks like this test set."
    )
    print()
    n_test = len(y_test)
    baseline_total = report_value("Baseline", *baseline_counts, n_test)
    print()
    v1_total = report_value("v1", *v1_counts, n_test)
    print()
    difference = v1_total - baseline_total
    per_difference = difference / n_test
    print(f"v1 minus baseline, test total: {money(difference)}")
    print(f"v1 minus baseline, per test appointment: {money(per_difference)}")
    print(
        "v1 minus baseline, per year at "
        f"{APPOINTMENTS_PER_YEAR:,} appointments: {money(per_difference * APPOINTMENTS_PER_YEAR)}"
    )
    print(
        "A positive difference means the depth-3 overbooking rule has a higher "
        "test-set value than holding every slot."
    )


if __name__ == "__main__":
    main()
