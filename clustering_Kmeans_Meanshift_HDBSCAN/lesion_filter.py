import pandas as pd

def filter_by_voxel_count(df: pd.DataFrame, min_voxels: int, max_voxels: int = None) -> pd.DataFrame:
    """
    Filtra las filas de df cuyo valor en 'num_voxels' esté >= min_voxels
    y (si max_voxels no es None) <= max_voxels.

    Args:
        df (pd.DataFrame): DataFrame con al menos la columna 'num_voxels'.
        min_voxels (int): Límite inferior inclusivo.
        max_voxels (int, optional): Límite superior inclusivo. Si es None, solo aplica el límite inferior.

    Returns:
        pd.DataFrame: Nuevo DataFrame con las filas que cumplen la condición.
    """
    # Creamos la máscara booleana para el filtro
    mask = df['num_voxels'] >= min_voxels
    if max_voxels is not None:
        mask &= df['num_voxels'] <= max_voxels

    # Devolvemos solo las filas válidas
    return df.loc[mask].reset_index(drop=True)