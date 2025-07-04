import pandas as pd
from typing import List, Tuple, Union


def calculate_k_anonymity(df: pd.DataFrame, qid_cols: List[str]) -> Tuple[Union[int, float], int]:
    """
    Calcola k-anonymity (k minimo) e il numero di record singoli (dove k=1).
    Restituisce (k_min, records_singoli).
    """
    if not qid_cols or df.empty or not all(col in df.columns for col in qid_cols):
        # print("DEBUG k-anonymity: QID vuoti, df vuoto, o QID non nel df.")
        return float('inf'), 0

    df_qid_dropped = df[qid_cols].dropna()
    if df_qid_dropped.empty:
        # print("DEBUG k-anonymity: df_qid_dropped vuoto.")
        return float('inf'), 0

    try:
        for col in qid_cols:
            if df_qid_dropped[col].dtype != 'object':
                df_qid_dropped[col] = df_qid_dropped[col].astype(str)

        eq_sizes = df_qid_dropped.groupby(qid_cols, observed=True,
                                          dropna=False).size()  # observed=True è default ma esplicito
    except KeyError:
        # print(f"DEBUG k-anonymity: KeyError durante il groupby: {e}")
        return float('inf'), 0  # Uno dei qid_cols non è valido dopo il dropna? Improbabile.

    if eq_sizes.empty:
        # print("DEBUG k-anonymity: eq_sizes vuoto.")
        return float('inf'), 0

    class_size_counts = eq_sizes.value_counts().sort_index()

    records_singoli = class_size_counts.get(1, 0)
    k_min = eq_sizes.min()

    return k_min, records_singoli


def calculate_l_diversity(df: pd.DataFrame, qid_cols: List[str], sensitive_col: str) -> int:
    """
    Calcola l-diversity (l minimo) per un attributo sensibile dato un set di QID.
    Restituisce l_min.
    """
    if not qid_cols or sensitive_col not in df.columns or df.empty or not all(col in df.columns for col in qid_cols):
        # print(f"DEBUG l-diversity per {sensitive_col}: Input non validi.")
        return 0

    cols_to_process = qid_cols + [sensitive_col]
    df_subset = df[cols_to_process].dropna()
    if df_subset.empty:
        # print(f"DEBUG l-diversity per {sensitive_col}: df_subset vuoto dopo dropna.")
        return 0

    try:
        # Assicura tipi stringa per groupby e nunique consistente
        for col in qid_cols:
            if df_subset[col].dtype != 'object':
                df_subset[col] = df_subset[col].astype(str)
        if df_subset[sensitive_col].dtype != 'object':
            df_subset[sensitive_col] = df_subset[sensitive_col].astype(str)

        # Calcola il numero di valori unici dell'attributo sensibile per ogni gruppo di QID
        diversities = df_subset.groupby(qid_cols, observed=True, dropna=False)[sensitive_col].nunique()
    except KeyError:
        # print(f"DEBUG l-diversity per {sensitive_col}: KeyError durante il groupby: {e}")
        return 0

    if diversities.empty:
        # print(f"DEBUG l-diversity per {sensitive_col}: series 'diversities' vuota.")
        return 0

    return diversities.min()
