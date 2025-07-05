from pathlib import Path
from typing import Union, List, Tuple
import pandas as pd


def extract_column_name(filepath: Union[str, Path]) -> dict:
    temp_file = filepath
    df = pd.read_excel(temp_file)
    column_names = df.columns.to_list()
    possible_names = []
    for col in df.columns:
        if df[col].dtype == object:
            possible_names.extend(df[col].dropna().unique().tolist())

    filtered_names = [name for name in possible_names if
                      isinstance(name, str) and name.istitle() and len(name.split()) <= 3]

    metadata = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": column_names,
        "has_nulls": df.isnull().values.any(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "filename": filepath.name
    }
    result = {
        "column_names": column_names,
        "filtered_names": filtered_names,
        "metadata": metadata
    }
    return result


def viz_recommendation_extraction(filenames: List) -> Tuple[dict, dict]:
    base_dir = Path(__file__).parent.parent.parent
    viz_columns = {}
    viz_rows = {"sample_data": {"sample_rows": {}}}  # Initialize properly

    for filename in filenames:
        filedir = base_dir / "media" / filename
        df = pd.read_excel(filedir)

        column_names = df.columns.to_list()
        viz_columns[filename] = column_names

        sample_rows = df.head(3).to_dict(orient="records")

        viz_rows["sample_data"]["sample_rows"][filename] = sample_rows

    return viz_columns, viz_rows


if __name__ == "__main__":
    viz_columns, viz_rows = viz_recommendation_extraction(["S&M Data.xlsx"])
    print(viz_rows)
    print(viz_columns)
