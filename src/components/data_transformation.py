import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import yaml
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


with open("src/config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __int__(self):
        self.data_transformation_config=DataTransformationConfig
    
    def get_data_trasnformer_object(self):
        try:
            numerical_columns= config['columns']['numerical_columns']
            categorical_coulmns= config['columns']['categorical_columns']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy=config['imputer_strategy']['numerical_imputer_strategy'])),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Numerical columns imputation and scaling completed and columns:{numerical_columns}")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy=config['imputer_strategy']['categorical_imputer_strategy'])),
                    ("one_hot_enconder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Categorical columns encoding is completed and columns:{categorical_coulmns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_coulmns)
                ]
            )

            logging.info(f"Column transformer preprocessor object is created")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)            
        
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_trasnformer_object()

            target_column_name=config['columns']['target_column']
            numerical_columns= config['columns']['numerical_columns']
            logging.info(f"target column and numerical created and columns are:{target_column_name} and {numerical_columns}")   

            input_feature_train_df=train_df.drop(columns=target_column_name)
            target_feature_train_df=train_df[target_column_name]

            logging.info(f"inpiut df and target df created for training data")

            input_feature_test_df=test_df.drop(columns=target_column_name)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Fit transform is applied on training data and transform is applied on testing data")

            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            logging.info(f"Combined training array is created")

            test_arr= np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Combined testing array is created")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

