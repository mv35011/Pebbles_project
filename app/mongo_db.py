
from pathlib import Path
import pandas as pd
import pymongo
import os
from dotenv import load_dotenv
load_dotenv()

mongo_string = os.getenv("MONGO_STRING")


def store_excel_in_db(mongo_string: str, file_name: str):
    try:
        base_dir = Path(__file__).resolve().parent.parent
        sample_excel_path = base_dir / "media" / f"{file_name}"

        df = pd.read_excel(sample_excel_path)

        client = pymongo.MongoClient(mongo_string)
        db = client["Dashboard_project"]
        collection = db["tests"]

        result = collection.insert_one({
            "_id": f"{file_name}",
            "filename": sample_excel_path.name,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview": df.head(5).to_dict("records")
        })

        print("Document inserted with ID:", result.inserted_id)

    except Exception as e:
        print("Error with the mongo db operation:", e)
