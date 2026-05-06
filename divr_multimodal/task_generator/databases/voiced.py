import pandas as pd
from pathlib import Path
from typing import List, Set, Tuple
from divr_diagnosis import DiagnosisMap

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedSession,
    ProcessedText,
)


class Voiced(Base):
    DB_NAME = "voiced"

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        _, df = self.__read_data(source_path=source_path)
        return set(df["Diagnosis"].tolist())

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        data_path, all_data = self.__read_data(source_path)

        for _, row in all_data.iterrows():
            speaker_id = row["ID"]
            raw_diagnosis = row["Diagnosis"]
            diagnosis = (
                diagnosis_map[raw_diagnosis]
                if raw_diagnosis in diagnosis_map
                else diagnosis_map.unclassified
            )
            age = int(row["Age"])
            gender = Gender.format(row["Gender"])

            if (
                allow_incomplete_classification
                or not diagnosis.incompletely_classified
            ):
                text_key = f"{data_path}/{speaker_id}.wav"
                text_payload = (
                    f"dataset=voiced; speaker_id={speaker_id}; "
                    f"age={age}; gender={gender}; "
                    f"original_label={raw_diagnosis}; "
                    f"smoking={row['smoking']}; "
                    f"drinking={row['drinking']}"
                )
                num_texts = 1
                if min_tasks is None or num_texts >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"voiced_{speaker_id}",
                            speaker_id=speaker_id,
                            age=age,
                            gender=gender,
                            diagnosis=[diagnosis],
                            texts=[
                                ProcessedText(
                                    text_key=text_key,
                                    text=text_payload,
                                )
                            ],
                            num_texts=num_texts,
                        )
                    ]
        return sessions

    def __read_data(self, source_path: Path) -> Tuple[Path, pd.DataFrame]:
        data_path = source_path / "voice-icar-federico-ii-database-1.0.0"

        info_files = list(data_path.rglob("*-info.txt"))
        rows = []
        for ifile in info_files:
            df = pd.read_csv(ifile, delimiter="\t", header=None)
            df.dropna(how="all", inplace=True)
            row = pd.Series(
                list(df[1]), index=df[0].apply(lambda x: x.replace(":", ""))
            )
            row = self.__fix_errors(ifile, row)
            rows += [row]
        all_data = pd.DataFrame(rows)
        all_data = all_data[["ID", "Diagnosis", "Gender", "Age", "Smoker", "Alcohol consumption"]]
        all_data["Diagnosis"] = all_data["Diagnosis"].str.lower().str.strip()
        all_data['smoking'] = all_data['Smoker'].str.lower().str.strip().apply(self.__clean_smoking)
        all_data['drinking'] = all_data['Alcohol consumption'].str.lower().str.strip().apply(self.__clean_drinking)
        return data_path, all_data

    def __fix_errors(self, ifile: Path, row: pd.Series) -> pd.Series:
        """
        Used for fixing errors in the DB
        """
        filekey = ifile.stem.removesuffix("-info")
        if row["ID"] != filekey:
            print(
                "Info: Fixing DB error where "
                f"original ID={row['ID']}, ifile={ifile}"
            )
            row["ID"] = filekey
        return row
    
    def __clean_smoking(self, smoking_value: str) -> str:
        """
        Clean and standardize smoking data
        """
        if pd.isna(smoking_value):
            return "unknown"
        smoking_value = smoking_value.lower().strip()
        if smoking_value in {"yes", "casual smoker"}:
            return "active"
        elif smoking_value in {"no"}:
            return "never"
        else:
            return "unknown"
        
    def __clean_drinking(self, drinking_value: str) -> str:
        """
        Clean and standardize drinking data
        """
        if pd.isna(drinking_value):
            return "unknown"
        drinking_value = drinking_value.lower().strip()
        if drinking_value in {"habitual drinker", "casual drinker"}:
            return "active"
        elif drinking_value in {"nondrinker"}:
            return "never"
        else:
            return "unknown"