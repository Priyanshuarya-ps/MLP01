import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __int__(self):
        self.data_transformation_config=DataTransformationConfig
    
    def get_data_trasnformer_object(self):
        try:
            numerical_columns= ['reading_score', 'writing_score']
            categorical_coulmns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_enconder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical columns standard scaling completed and columns:{numerical_columns}")
            logging.info(f"Categorical columns encoding is completed and columns:{categorical_coulmns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_coulmns)
                ]
            )

        except:
            pass
