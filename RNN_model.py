'''
Create a training model for RNN
'''

"""
Update:
Go back to embedding application

Try to do:


The place need to updated when adding features to the model 


"""

# Importing the libraries
import os
import time
import json
import random
import joblib
import pickle
import torch
import airportsdata
import argparse

import pandas as pd
import numpy as np
import dask.dataframe as dd

from torch import nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# import the ray package
# import ray
# from ray import tune
# from ray.air import session
# from ray.air.checkpoint import Checkpoint
# from ray.tune.schedulers import ASHAScheduler


class DatasetFormation():
    def __init__(self, data_root, seats_file, perf_file, x_feat, if_add_time_info=True,
                 sequence_length=12, pred_num_quarters=2, 
                 start_year=2004, end_year=2022):
        """
        Create a class for forming the dataset for RNN model
        """
        self.root = data_root
        self.df = None
        self.train_df = None
        self.test_df = None
        self.scaled_df = None
        self.full_df = None
        self.unique_airports = None
        self.unique_airlines = None
        self.le_airports = LabelEncoder()
        self.le_airlines = LabelEncoder()
        self.airports = airportsdata.load('IATA')
        self.not_found_codes = set()  # set of airport codes not found in airportsdata
        self.not_found_hubs = set() # set of hubs not found in the dataset
        self.missing_airports = {} # dictionary of missing airports
        self.major_hubs = {} # dictionary of major hubs
        self.al_embeddings = None
        self.ap_embeddings = None
        self.al_embedding_dict = {}
        self.ap_embedding_dict = {}
        self.scaler = StandardScaler()
        self.seat_scaler = StandardScaler()
        self.date_scaler = MinMaxScaler()
        # self.date_scaler = StandardScaler()
        self.seq_len = sequence_length
        self.n_future = pred_num_quarters
        self.start_year = start_year
        self.end_year = end_year
        self.year_split = None
        self.al_class = None

        # Judge if root is on windows or linux
        if '\\' in self.root:
            pop_file = '\Pop.csv'
        else:
            pop_file = 'Pop.csv'

        # Read the data
        self.df_seat = pd.read_csv(self.root + seats_file)
        self.df_perf = pd.read_csv(self.root + perf_file)
        self.df_pop = pd.read_csv(self.root + pop_file)

        # Define fixed parameters
        self.days_in_quarter = {'Q1': 90, 'Q2': 91, 'Q3': 92, 'Q4': 92}
        self.quarter_mapping = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
        self.x_features = x_feat
        self.time_add = if_add_time_info
        self.cat_features = ['al_code', 'orig_code', 'dest_code', 'al_type', 'state_o', 'state_d']
        self.num_features = [f for f in self.x_features if f not in self.cat_features]
        self.cat_mapping = None
        self.embed_dim_mapping = None
        # Remove 'Seats' from x_features before scaling
        self.x_features_without_seats = [feat for feat in self.num_features if feat != "Seats"]

        self.init_missing_airports()
        self.init_al_info()

    def init_missing_airports(self):
        self.missing_airports = {
            'OXR': {'name': 'Oxnard Airport', 'lat': 34.2008, 'lon': -119.207},
            'ISO': {'name': 'Kinston Regional Jetport', 'lat': 35.3314, 'lon': -77.6088},
            'OWD': {'name': 'Norwood Memorial Airport', 'lat': 42.1906, 'lon': -71.1731},
            'FMN': {'name': 'Four Corners Regional Airport', 'lat': 36.7411, 'lon': -108.229},
            'VGT': {'name': 'North Las Vegas Airport', 'lat': 36.2107, 'lon': -115.194},
            'PDT': {'name': 'Eastern Oregon Regional Airport', 'lat': 45.6951, 'lon': -118.841},
            'UTM': {'name': 'Tunica Municipal Airport', 'lat': 34.6800, 'lon': -90.3467},
            'MWH': {'name': 'Grant County International Airport', 'lat': 47.2077, 'lon': -119.320},
            'OFK': {'name': 'Norfolk Regional Airport', 'lat': 41.9855, 'lon': -97.4351},
            'TKF': {'name': 'Truckee-Tahoe Airport', 'lat': 39.3200, 'lon': -120.139},
            'ILE': {'name': 'Killeen Municipal Airport', 'lat': 31.0858, 'lon': -97.6794},
            'WYS': {'name': 'Yellowstone Airport', 'lat': 44.6884, 'lon': -111.1176},
            'VIS': {'name': 'Visalia Municipal Airport', 'lat': 36.3187, 'lon': -119.392},
            'TVF': {'name': 'Thief River Falls Regional Airport', 'lat': 48.0656, 'lon': -96.1850},
            'TSS': {'name': 'East 34th Street Heliport', 'lat': 40.7425, 'lon': -73.9722},
            'TSM': {'name': 'Taos Regional Airport', 'lat': 36.4582, 'lon': -105.672},
            'HII': {'name': 'Lake Havasu City Airport', 'lat': 34.5711, 'lon': -114.358},
            'GYY': {'name': 'Gary/Chicago International Airport', 'lat': 41.6163, 'lon': -87.4128},
            'FNL': {'name': 'Northern Colorado Regional Airport', 'lat': 40.4518, 'lon': -104.988},
            'ISN': {'name': 'Sloulin Field International Airport', 'lat': 48.1779, 'lon': -103.642},
            'FKL': {'name': 'Venango Regional Airport', 'lat': 41.3781, 'lon': -79.8601},
            'MYF': {'name': 'Montgomery Field', 'lat': 32.8158, 'lon': -117.139},
            'OLF': {'name': 'L. M. Clayton Airport', 'lat': 48.0945, 'lon': -105.575},
            'PVC': {'name': 'Provincetown Municipal Airport', 'lat': 42.0722, 'lon': -70.2214},
            'SCK': {'name': 'Stockton Metropolitan Airport', 'lat': 37.8942, 'lon': -121.238},
            'CLD': {'name': 'McClellan-Palomar Airport', 'lat': 33.1283, 'lon': -117.28},
            'MWA': {'name': 'Williamson County Regional Airport', 'lat': 37.755, 'lon': -89.0111},
            'JRB': {'name': 'Downtown Manhattan Heliport', 'lat': 40.7012, 'lon': -74.009},
            'MBL': {'name': 'Manistee County-Blacker Airport', 'lat': 44.2725, 'lon': -86.2469},
            'LAM': {'name': 'Los Alamos Airport', 'lat': 35.8798, 'lon': -106.269},
            'SOP': {'name': 'Moore County Airport', 'lat': 35.2375, 'lon': -79.3887},
            'CIC': {'name': 'Chico Municipal Airport', 'lat': 39.7954, 'lon': -121.858},
            'IWD': {'name': 'Gogebic-Iron County Airport', 'lat': 46.5275, 'lon': -90.1314},
            'ESC': {'name': 'Delta County Airport', 'lat': 45.7227, 'lon': -87.0937},
            'NPT': {'name': 'Newport State Airport', 'lat': 41.5329, 'lon': -71.2825},
            'WDG': {'name': 'Enid Woodring Regional Airport', 'lat': 36.3792, 'lon': -97.7911},
            'ELY': {'name': 'Ely Airport', 'lat': 39.2997, 'lon': -114.841},
            'LCK': {'name': 'Rickenbacker International Airport', 'lat': 39.8138, 'lon': -82.9278},
            'TEB': {'name': 'Teterboro Airport', 'lat': 40.8499, 'lon': -74.0608},
            'SLE': {'name': 'Salem Municipal Airport', 'lat': 44.9095, 'lon': -123.003},
            'CNY': {'name': 'Canyonlands Field', 'lat': 38.755, 'lon': -109.754},
            'RIW': {'name': 'Riverton Regional Airport', 'lat': 43.0642, 'lon': -108.459},
            'MTH': {'name': 'Florida Keys Marathon Airport', 'lat': 24.7261, 'lon': -81.0514},
            'MCK': {'name': 'McCook Ben Nelson Regional Airport', 'lat': 40.2064, 'lon': -100.592},
            'HOB': {'name': 'Lea County Regional Airport', 'lat': 32.6875, 'lon': -103.217},
            'PLB': {'name': 'Clinton County Airport', 'lat': 44.6884, 'lon': -73.5248},
            'PGD': {'name': 'Punta Gorda Airport', 'lat': 26.9196, 'lon': -81.9905},
            'PNC': {'name': 'Ponca City Regional Airport', 'lat': 36.7311, 'lon': -97.0997},
            'HHR': {'name': 'Hawthorne Municipal Airport', 'lat': 33.9228, 'lon': -118.335},
            'PFN': {'name': 'Panama City–Bay County International Airport', 'lat': 30.2121, 'lon': -85.6828},
            'JHW': {'name': 'Chautauqua County/Jamestown Airport', 'lat': 42.1533, 'lon': -79.258},
            'SBD': {'name': 'San Bernardino International Airport', 'lat': 34.0954, 'lon': -117.235},
            'VEL': {'name': 'Vernal Regional Airport', 'lat': 40.4409, 'lon': -109.51},
            'EFD': {'name': 'Ellington Airport', 'lat': 29.6040, 'lon': -95.1763},
        }
        print("Missing airports database loaded.")

    def init_al_info(self):
        self.major_hubs = {
            'UA': ['ORD', 'LAX', 'SFO', 'DEN', 'IAH'],  # United Airlines
            'AA': ['DFW', 'CLT', 'PHL', 'MIA', 'PHX'],  # American Airlines
            'DL': ['ATL', 'JFK', 'MSP', 'DTW', 'SLC'],  # Delta Air Lines
            'AS': ['SEA', 'PDX', 'ANC'],  # Alaska Airlines
            'B6': ['JFK', 'BOS', 'FLL'],  # JetBlue
            'CO': ['EWR', 'IAH', 'CLE'],  # Continental Airlines (merged with UA in 2012)
            'F9': ['DEN', 'MCO', 'PHL'],  # Frontier Airlines
            'WN': ['ATL', 'DAL', 'DEN'],  # Southwest Airlines
            'NW': ['DTW', 'MSP', 'MEM'],  # Northwest Airlines (merged with DL in 2010)
            'VX': ['SFO', 'LAX', 'DCA'],  # Virgin America (acquired by AS in 2018)
            'NK': ['FLL', 'MCO', 'DTW'],  # Spirit Airlines
            'FL': ['ATL', 'MCO', 'TPA'],  # AirTran Airways (acquired by WN in 2011)
            'US': ['CLT', 'PHL', 'PHX'],  # US Airways (merged with AA in 2013)
            'YX': ['MKE', 'MCI', 'OMA'],  # Midwest Airlines (merged with FL in 2010)
            'G4': ['LAS', 'SFB', 'PGD'],  # Allegiant Air
        }

        # self.al_class = {
        #     'UA': 1.0,
        #     'AA': 1.0,
        #     'DL': 1.0,
        #     'AS': 0.6,
        #     'WN': 0.6,
        #     'F9': 0.5,
        #     'NK': 0.5,
        #     'B6': 0.4,
        #     'G4': 0.4,
        #     'CO': 0.1,
        #     'NW': 0.1,
        #     'VX': 0.1,
        #     'FL': 0.1,
        #     'US': 0.1,
        #     'YX': 0.1
        # }

        self.al_class = {
            'UA': 1,
            'AA': 1,
            'DL': 1,
            'AS': 2,
            'WN': 2,
            'F9': 3,
            'NK': 3,
            'B6': 4,
            'G4': 4,
            'CO': 5,
            'NW': 5,
            'VX': 5,
            'FL': 5,
            'US': 5,
            'YX': 5
        }

        print("Major hubs database loaded.")

    def one_step_process(self, boundary_quarter="Q2 2022", test_boundary_quarter="Q1 2021"):
        """
        Run all the steps in one step.
        """

        # self.init_missing_airports()
        self.clean_data()
        print("The data is cleaned.")

        self.assgin_geo_features()
        print("The geo features are assigned.")

        self.transform_od_al()
        print("The origin, destination and airline are encoded.")

        self.attach_coordinates()
        print("The airports coordinates are calculated.")

        # self.create_embedding_features()
        # print("The embeddings are created.")

        self.assign_airline_features()
        print("The airline features are assigned.")

        self.assign_sort_date()
        self.assign_pandamic_year()
        print("The pandamic features are assigned.")

        self.calculate_competitors()
        print("The competitors are calculated.")

        self.calculate_market_share()
        print("The market share is calculated.")

        self.calculate_market_size()
        print("The market size is calculated.")

        self.split_data(boundary_quarter=boundary_quarter, test_boundary_quarter = test_boundary_quarter)
        print("The data is split into train and test.")

        self.save_trainning_data()
        self.save_testing_data()
        print("The train and test data are saved.")

        self.scaler_data()
        print("The train data is scaled to scaled_df.")

        self.create_date_features()
        print("The date is transformed to relevant values.")

        self.save_scaled_data()
        print("The scaled data is saved.")

        self.final_preparation()
        print("The full trainning dataset is prepared.")

    def clean_data(self):
        """
        1. Merge the two dataframes
        2. Calculate `Seats` and `Flights` for NaN values
        3. Transfer the column `Miles` to numeric values
        4. Calculate `ASMs` for NaN values
        5. Drop unnecessary columns
        6. Add columns of year, quarter and Alpha (Origin + Dest)
        """

        # Merge the two dataframes
        merged_df = pd.merge(self.df_perf, self.df_seat, on=['Date', 'Mkt Al', 'Orig', 'Dest'], how='left')

        # Dictionary to map quarter to number of days in that quarter
        merged_df['Quarter'] = merged_df['Date'].str[0:2]
        merged_df['Days'] = merged_df['Quarter'].map(self.days_in_quarter)

        # Calcuate `Seats` and `Flights` for NaN values
        merged_df.loc[merged_df['Seats'].isna(), 'Seats'] = merged_df['Seats/Day'] * merged_df['Days']
        merged_df.loc[merged_df['Flights'].isna(), 'Flights'] = merged_df['Deps/Day'] * merged_df['Days']

        # # Transfer the column `Miles` to numeric values
        if merged_df['Miles'].dtype != 'int64':
            merged_df['Miles'] = merged_df['Miles'].str.replace(',', '')
            merged_df['Miles'] = pd.to_numeric(merged_df['Miles'])        

        # Calculate `ASMs` for NaN values
        merged_df.loc[merged_df['ASMs'].isna(), 'ASMs'] = merged_df['Seats'] * merged_df['Miles']

        # Drop unnecessary columns
        merged_df = merged_df.drop(['Days', 'Quarter'], axis=1)

        # Add columns of year, quarter and Alpha (Origin + Dest)
        merged_df['year'] = merged_df['Date'].str[-4:]
        merged_df['quarter'] = merged_df['Date'].str[1:2]
        merged_df['Alpha'] = merged_df['Orig'] + merged_df['Dest']

        self.df = merged_df
    
    def assgin_geo_features(self):

        # Merge df_pop to data_format.df accoring to `Orig` in data_format.df,
        self.df = self.df.merge(self.df_pop, how='left', left_on='Orig', right_on='ap')
        self.df.rename(columns={'g1': 'g1_o', 'g2': 'g2_o', 'log': 'log_o', 'StateCode': 'state_o'}, inplace=True)
        self.df.drop(columns=['ap'], inplace=True)

        # Merge df_pop to data_format.df accoring to `Dest` in data_format.df
        self.df = self.df.merge(self.df_pop, how='left', left_on='Dest', right_on='ap')
        self.df.rename(columns={'g1': 'g1_d', 'g2': 'g2_d', 'log': 'log_d', 'StateCode': 'state_d'}, inplace=True)
        self.df.drop(columns=['ap'], inplace=True)

    def transform_od_al(self):
        """
        Transform the origin, destination and airline into encoded values.
        """
        # Create a list of unique airports
        com_airports = pd.concat([self.df['Orig'], self.df['Dest']])
        self.unique_airports = com_airports.unique()

        # Create a list of unique airlines
        self.unique_airlines = self.df['Mkt Al'].unique()

        # Encode the airports and airlines
        self.le_airports.fit(self.unique_airports)
        self.le_airlines.fit(self.unique_airlines)

        joblib.dump(self.le_airports, 'le_airports.pkl')
        joblib.dump(self.le_airlines, 'le_airlines.pkl')

        # Create a column of encoded airports and airlines
        self.df['orig_code'] = self.le_airports.transform(self.df['Orig'])
        self.df['dest_code'] = self.le_airports.transform(self.df['Dest'])
        self.df['al_code'] = self.le_airlines.transform(self.df['Mkt Al'])

    def get_coordinates(self, airport_code):
        """
        Get the coordinates of an airport given its code.
        """
        try:
            return self.airports[airport_code]['lat'], self.airports[airport_code]['lon']
        except KeyError:
            if airport_code in self.missing_airports:
                missing_info = self.missing_airports[airport_code]
                if airport_code not in self.not_found_codes:
                    # print(f"Note: Airport code {airport_code} ({missing_info['name']}) was not found. Using self-defined data source.")
                    self.not_found_codes.add(airport_code)
                return missing_info['lat'], missing_info['lon']
            else:
                if airport_code not in self.not_found_codes:
                    print(f"Warning: Airport code {airport_code} not found in any data source.")
                    self.not_found_codes.add(airport_code)
                    print("Stop and check not found codes: ", self.not_found_codes)
                return 0, 0

    def attach_coordinates(self):
        """
        Transfer the origin and destination airport codes to coordinates.
        """
        self.df['orig_coord'] = self.df['Orig'].apply(self.get_coordinates)
        self.df['dest_coord'] = self.df['Dest'].apply(self.get_coordinates)
        self.df['orig_lat'] = self.df['orig_coord'].apply(lambda x: x[0])
        self.df['orig_lon'] = self.df['orig_coord'].apply(lambda x: x[1])
        self.df['dest_lat'] = self.df['dest_coord'].apply(lambda x: x[0])
        self.df['dest_lon'] = self.df['dest_coord'].apply(lambda x: x[1])
        self.df = self.df.drop(['orig_coord', 'dest_coord'], axis=1)

    def create_embedding_features(self, al_feature_size=2, ap_feature_size=3):
        """
        STOP USING THIS FUNCTION.
        Create embedding features for airlines and airports.
        """
        self.al_embeddings = nn.Embedding(len(self.unique_airlines), al_feature_size)
        self.ap_embeddings = nn.Embedding(len(self.unique_airports), ap_feature_size)

        al_data = torch.tensor(self.df['al_code'].values)
        orig_data = torch.tensor(self.df['orig_code'].values)
        dest_data = torch.tensor(self.df['dest_code'].values)

        al_embed = self.al_embeddings(al_data)
        orig_embed = self.ap_embeddings(orig_data)
        dest_embed = self.ap_embeddings(dest_data)

        # Concatenate the embedding vectors and add columns to the dataframe
        embed_vector = torch.cat((al_embed, orig_embed, dest_embed), 1)
        embed_vector_array = embed_vector.detach().numpy()
        embed_vector_df = pd.DataFrame(embed_vector_array)
        embed_vector_df.columns = ['al_embed_1', 'al_embed_2', 'orig_embed_1', 'orig_embed_2', 'orig_embed_3', 'dest_embed_1', 'dest_embed_2', 'dest_embed_3']
        self.df = pd.concat([self.df, embed_vector_df], axis=1)

        # Create mapping dictionaries
        self.al_embedding_dict = {al: emb for al, emb in zip(self.unique_airlines, self.al_embeddings.weight.detach().numpy())}
        self.ap_embedding_dict = {ap: emb for ap, emb in zip(self.unique_airports, self.ap_embeddings.weight.detach().numpy())}

        # Write data to CSV files
        pd.DataFrame([(k, *v.tolist()) for k, v in self.al_embedding_dict.items()]).to_csv("al_embedding.csv", index=False)
        pd.DataFrame([(k, *v.tolist()) for k, v in self.ap_embedding_dict.items()]).to_csv("ap_embedding.csv", index=False)

        # print("Embedding features created.")

    def assign_airline_features(self):
        """
        Assign airline features to the dataframe.
        Includes: 'If legacy carrier'; If Hub;
        """
        # self.df['if_legacy'] = self.df['Mkt Al'].apply(lambda x: 1 if x in self.legacy_carriers else 0)
        # If hub:
        self.df['if_hub'] = self.df.apply(self.is_hub, axis=1)

        # Airline type:
        self.df['al_type'] = self.df['Mkt Al'].apply(self.get_al_type)

    def is_hub(self, row):
        """
        This function checks if the flight is a hub flight.
        """
        airline = row['Mkt Al']
        if airline in self.major_hubs:
            if row['Orig'] in self.major_hubs[airline] or row['Dest'] in self.major_hubs[airline]:
                return 1
            else:
                return 0
        else:
            # self.not_found_hubs.add(airline)
            return 0
        
    def get_al_type(self, airline):
        if airline in self.al_class:
            return self.al_class[airline]
        else:
            return 0.05
        
    def assign_sort_date(self):
        """
        Assign a sort date to each row.
        """
        self.df['SortDate'] = self.df['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))

    def assign_pandamic_year(self):
        """
        Assign a pandamic year to each row.
        """
        self.df['pand_year'] = self.df['SortDate'].apply(self.pandamic_year)

    def pandamic_year(self, year):
        """
        This function returns the pandamic year of a given year.
        """
        b_quarter = "Q4 2019"
        e_quarter = "Q3 2021"
        bq_num = int(b_quarter.split(' ')[1]) * 4 + int(b_quarter.split(' ')[0][1])
        eq_num = int(e_quarter.split(' ')[1]) * 4 + int(e_quarter.split(' ')[0][1])
        if year < bq_num:
            return 1
        elif year < eq_num:
            return 2
        else:
            return 3

    def calculate_competitors(self):
        # Count the number of competitors per route
        self.df['num_comp'] = self.df.groupby(['year', 'quarter', 'Orig', 'Dest'])['Mkt Al'].transform('nunique')

    def calculate_market_share(self):
        # Calculate the market share of each airline per route
        # self.df['mkt_share'] = self.df.groupby(['year', 'quarter', 'Orig', 'Dest'])['Pax/Day'].transform(lambda x: x / x.sum())

        sum_pax = self.df.groupby(['year', 'quarter', 'Orig', 'Dest'])['Pax/Day'].transform('sum')
        self.df['mkt_share'] = self.df['Pax/Day'] / sum_pax

    def calculate_market_size(self):
        # Calculate the market size of each route
        self.df['mkt_size'] = self.df.groupby(['year', 'quarter', 'Orig', 'Dest'])['Seats'].transform('sum')

    def split_data(self, boundary_quarter, test_boundary_quarter):
        """
        This function splits the data into training and validation sets.
        """

        boundary_num = int(boundary_quarter.split(' ')[1]) * 4 + int(boundary_quarter.split(' ')[0][1])

        # self.year_split = int(train_size * (self.end_year - self.start_year) + self.start_year)
        # self.year_split = 2021
        # self.train_df = self.df[self.df['year'] <= str(self.year_split)]
        # self.train_df = self.df[self.df['year'] >= str(self.start_year)]
        self.train_df = self.df[self.df['SortDate'] <= boundary_num].copy()

        # limit the begin year of the training data
        self.train_df = self.train_df[self.train_df['year'] >= str(self.start_year)]
        # reset the index
        self.train_df.reset_index(drop=True, inplace=True)

        # Needed to be updated
        # This part is not right since to predict 2022 data.
        # For 2022 Q1, Q2 data, if trained 10 quarters to predict, 
        # then the data needed is 2019(2), 2020(4), 2021(4)
        # For 2022 Q3, Q4 data, if trained 10 quarters to predict, 
        # then the data needed is 2020(4), 2021(4), 2022(2)

        test_boundary_num = int(test_boundary_quarter.split(' ')[1]) * 4 + int(test_boundary_quarter.split(' ')[0][1])

        # self.test_df = self.df[self.df['year'] >= str(self.end_year - self.seq_len//4)]
        self.test_df = self.df[self.df['SortDate'] >= test_boundary_num].copy()
        
        # Drop `SortDate` column
        if 'SortDate' in self.train_df.columns:
            self.train_df.drop(columns=['SortDate'], inplace=True)
        if 'SortDate' in self.test_df.columns:
            self.test_df.drop(columns=['SortDate'], inplace=True)

    def scaler_data(self):
        """
        Scale the dataset using the standard scaler.
        """
        self.scaled_df = self.train_df.copy()

        scaled_features = self.scaler.fit_transform(self.scaled_df[self.x_features_without_seats])

        # scale the 'seats' feature separately and store the transformation
        self.scaled_df['Seats'] = self.seat_scaler.fit_transform(self.scaled_df['Seats'].values.reshape(-1,1))

        self.scaled_df[self.x_features_without_seats] = scaled_features

    def create_date_features(self, relative_date="2003-01-01"):
        """
        Convert the date to a numeric value
        Note: Relative date will be set to 2003-01-01 by default
        """
        if self.scaled_df is not None:
            # Split the quarter and year, map the quarter to a month and create a correct datetime string
            date_split = self.scaled_df['Date'].str.split(' ', expand=True)
            self.scaled_df['Date_delta'] = pd.to_datetime(date_split[0].map(self.quarter_mapping) + '/01/' + date_split[1])

            # Convert the date to a numeric value
            self.scaled_df['Date_delta'] = (self.scaled_df['Date_delta'] - pd.Timestamp(relative_date)) // pd.Timedelta('1s')

            # Shrinking the date value to a smaller range
            self.scaled_df['Date_delta'] = self.scaled_df['Date_delta'] / 100000

            # Scaling the date feature
            self.scaled_df['time_scaled'] = self.date_scaler.fit_transform(self.scaled_df['Date_delta'].values.reshape(-1,1))
        else:
            print("The dataset has not been scaled yet.")
        #     if self.scaled_df is not None:
        #         # Split the quarter and year, map the quarter to a month and create a correct datetime string
        #         self.scaled_df['Date_delta'] = self.scaled_df['Date'].apply(lambda x: pd.to_datetime(self.quarter_mapping[x.split(' ')[0]]+'/01/'+x.split(' ')[1])) # Note: Check if this is correct
        #         # Convert the date to a numeric value
        #         self.scaled_df['Date_delta'] = (self.scaled_df['Date_delta'] - pd.Timestamp(relative_date)) // pd.Timedelta('1s')
        #         # Shrinking the date value to a smaller range
        #         self.scaled_df['Date_delta'] = self.scaled_df['Date_delta'] / 100000

        #         # Sclaing the date feature
        #         self.scaled_df['time_scaled'] = self.date_scaler.fit_transform(self.scaled_df['Date_delta'].values.reshape(-1,1))
        #     else:
        #         print("The dataset has not been scaled yet.")

    def load_scaled_data(self, train_filename='training_data.csv', scaled_filename='scaled_data.csv'):
        self.train_df = pd.read_csv(train_filename)
        self.scaled_df = pd.read_csv(scaled_filename)
        print("Scaled data loaded.")

        # rebuild the main scaler
        self.scaler.fit(self.train_df[self.x_features_without_seats])
        print("Main scaler rebuilt.")
        
        # rebuild the seat scaler
        self.seat_scaler.fit(self.train_df['Seats'].values.reshape(-1,1))
        print("Seat scaler rebuilt.")

        # rebuild the date scaler
        self.date_scaler.fit(self.scaled_df['Date_delta'].values.reshape(-1,1))

        # Load the LabelEncoders
        self.le_airports = joblib.load('le_airports.pkl')
        self.le_airlines = joblib.load('le_airlines.pkl')

    def create_dim_mapping(self):
        # cat_features = ['Mkt Al', 'Orig', 'Dest', 'al_type', 'state_o', 'state_d']
        # numerical features
        # self.num_features = [f for f in self.x_features if f not in self.cat_features]

        # Mapping for each categorical feature to its unique values
        self.cat_mapping = {f: list(set(self.scaled_df[f])) for f in self.cat_features}
        # make sure all 'al_code' and 'orig_code' and 'dest_code' are in the mapping
        self.cat_mapping['al_code'] = [i for i in range(len(self.le_airlines.classes_))]
        
        # Update the mapping for 'orig_code' and 'dest_code' to reflect the label encoder's classes
        
        self.cat_mapping['orig_code'] = [i for i in range(len(self.le_airports.classes_))]
        self.cat_mapping['dest_code'] = [i for i in range(len(self.le_airports.classes_))]
        # self.cat_mapping['al_code'] = list(self.le_airlines.classes_)

        # Mapping for each categorical feature to its number of unique values
        self.embed_dim_mapping = {f: min(6, len(unique_values)//2*2) for f, unique_values in self.cat_mapping.items()}
        

        # Save the mapping to a file
        with open('cat_mapping.pkl', 'wb') as f:
            pickle.dump(self.cat_mapping, f)
        with open('embed_dim_mapping.pkl', 'wb') as f:
            pickle.dump(self.embed_dim_mapping, f)

        # # Load the mappings
        # with open("cat_mapping.pkl", "rb") as f:
        #     cat_mapping = pickle.load(f)

    def final_preparation(self):
        if self.year_split is None:
            # train_size = 0.9 # temporary value
            self.year_split = 2021

        if self.scaled_df is not None:
            # Create a mapping for each categorical feature to its unique values
            self.create_dim_mapping()
            # Create binary features for the year and quarter
            self.scaled_df = pd.get_dummies(self.scaled_df, columns=['quarter'])
            # group the dataframe for each airline and each route
            datasets = []
            for _, route_df in self.scaled_df.groupby(["Mkt Al", "Orig", "Dest"]):
                if len(route_df) < self.seq_len + self.n_future:
                    continue
                route_df = route_df.sort_values("Date_delta")
                datasets.append(FlightDataset(route_df, self.seq_len, self.num_features, 
                                              self.cat_features, self.embed_dim_mapping,
                                              time_add=self.time_add, n_future=self.n_future))
            self.full_df = torch.utils.data.ConcatDataset(datasets)
        else:
            print("The dataset has not been scaled yet.")
  
    def print_data(self, scaled=False, num_rows=5):

        if not scaled:
            # print the columns of the dataset
            print("The columns of the dataset are:")
            print(self.df.columns)

            # print the first num_rows rows of the dataset
            print("The first {} rows of the dataset are:".format(num_rows))
            print(self.df.head(num_rows))
            print("-" * 50)
        elif self.scaled_df is not None:
            # print the columns of the dataset
            print("The columns of the dataset are:")
            print(self.scaled_df.columns)

            # print the first num_rows rows of the dataset
            print("The first {} rows of the dataset are:".format(num_rows))
            print(self.scaled_df.head(num_rows))
            print("-" * 50)
        else:
            print("The dataset has not been scaled yet.")

    def save_data(self, file_name="dataset_for_analysis.csv"):
        self.df.to_csv(file_name, index=False)
        print("The dataset is saved as {}.".format(file_name))

    def save_scaled_data(self, filename="scaled_data.csv"):
        self.scaled_df.to_csv(filename, index=False)
    
    def save_trainning_data(self, filename="training_data.csv"):
        self.train_df.to_csv(filename, index=False)

    def save_testing_data(self, filename="testing_data.csv"):
        self.test_df.to_csv(filename, index=False)


class FlightDataset(Dataset):
    """
    Create a dataset that can be used for dataloading.
    """
    def __init__(self, df, sequence_length, num_feat, cat_feat, embed_dim_mapping, time_add=True, n_future=2):
        self.df = df
        self.sequence_length = sequence_length
        self.num_features = num_feat
        self.cat_features = cat_feat
        self.n_future = n_future
        self.dummy_quarter = [*[f"quarter_{i}" for i in range(1, 5)]]
        self.time_add = time_add

        if self.time_add:
            self.num_features = self.num_features + self.dummy_quarter + ['time_scaled']
        else:
            self.num_features = self.num_features + self.dummy_quarter

    def __len__(self):
        return len(self.df) - self.sequence_length - self.n_future + 1

    def __getitem__(self, idx):
        # Get the relevant slice of the dataframe
        df_slice = self.df.iloc[idx : idx + self.sequence_length]

        # Get the first row and last row of the time_scaled column
        first_row = df_slice['time_scaled'].iloc[0]
        last_row = df_slice['time_scaled'].iloc[-1]
        time_range = (first_row, last_row)
        loc_key = (df_slice['Mkt Al'].iloc[0], df_slice['Orig'].iloc[0], df_slice['Dest'].iloc[0], df_slice['Date'].iloc[-1]) # airline, origin, destination
        
        # Construct the sequence data
        sequence_data = df_slice[self.num_features].astype(float).values
        cat_sequence_data = torch.LongTensor(df_slice[self.cat_features].values)

        # Get the relevant slice of the dataframe for the target seats
        df_target_slice = self.df.iloc[idx + self.sequence_length : idx + self.sequence_length + self.n_future]

        # "Seats" is the 19th column (0-indexed)
        target_data = df_target_slice["Seats"].values 

        # Return the sequence data and the target value. "Seats" is the 19th column (0-indexed)
        # return torch.from_numpy(sequence_data[:-1]), torch.tensor(sequence_data[-1, 18])  
        return torch.from_numpy(sequence_data), cat_sequence_data, torch.tensor(target_data), time_range, loc_key


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, encoder_outputs):
        # MultiHeadAttention requires input as [sequence len, batch size, hidden_dim]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_output, attn_output_weights = self.attention(encoder_outputs, encoder_outputs, encoder_outputs)
        outputs = attn_output.transpose(0, 1)  # Transpose again to get back to [batch size, sequence len, hidden_dim]
        weights = attn_output_weights.transpose(0, 1)  # Transpose again to get back to [batch size, sequence len, sequence len]
        # outputs = outputs.mean(dim=1)  # Average over the sequence length
        return outputs, weights


class RNNNet(nn.Module):
    """
    Define an RNN network.
    """
    def __init__(self, cat_feat, cat_mapping, embed_dim_mapping, input_dim=51, hidden_dim=100, 
                 output_dim=1, n_layers=2, drop_prob=0.2, rnn_type="GRU", bidirectional=True, 
                 num_heads=8, if_skip=True, if_feed_drop=True, if_feed_norm=True, MSE=True):
        super(RNNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cat_features = cat_feat
        self.if_skip = if_skip
        self.dropout = nn.Dropout(drop_prob)
        self.if_feed_drop = if_feed_drop
        self.if_feed_norm = if_feed_norm
        self.bidirect = bidirectional
        self.num_heads = num_heads
        self.MSE = MSE

        # Define the embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(len(unique_values), embed_dim_mapping[f])
            for f, unique_values in cat_mapping.items()
            })
        
        # Update the input dimension of the RNN based on the dimensions of the embeddings
        input_dim += sum(embed_dim_mapping.values()) - len(cat_mapping)

        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, 
                              dropout=drop_prob, bidirectional=self.bidirect)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, 
                               dropout=drop_prob, bidirectional=self.bidirect)
        else:
            raise ValueError("Invalid RNN type specified: %s. Choose either 'GRU' or 'LSTM'." % rnn_type)

        self.rnn_to_attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = SelfAttention(hidden_dim, num_heads=num_heads)

        # # Adding Feed-Forward Layers
        if not self.if_feed_norm:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Add weight decay (L2 Regularization) for fully connected layers
            self.fc1 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim), dim=None)
            self.fc2 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim), dim=None)

        # Adding Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # self.fc = nn.Linear(hidden_dim, output_dim) # output_dim should now be n_future
        self.fc_mean = nn.Linear(hidden_dim, output_dim) 
        self.fc_std = nn.Linear(hidden_dim, output_dim) 
        self.relu = nn.ReLU()  

    def forward(self, x, cat_x, h):
        x_embed = [self.embeddings[f](cat_x[:, :, i]) for i, f in enumerate(self.cat_features)]
        x_embed = torch.cat(x_embed, 2)
        x = torch.cat((x, x_embed), 2)

        rnn_out, h = self.rnn(x, h)
        
        if self.bidirect:
            rnn_out = self.rnn_to_attention(rnn_out)
        skip_rnn_out = rnn_out
        out, _ = self.attention(rnn_out)

        # skip connection
        if self.if_skip:
            out = out + skip_rnn_out  # Adding skip connection

        # out = torch.mean(out, dim=1)
        
        # feed-forward layers
        if not self.if_feed_drop:
            out = self.fc1(out)
            out = self.fc2(out)
        else:
            out = self.dropout(self.fc1(out))
            out = self.dropout(self.fc2(out))

        out = torch.mean(out, dim=1)
        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.relu(out)

        if not self.MSE:
            mean = self.fc_mean(out)
            std = self.fc_std(out)
            std = F.softplus(std)
            return mean, std, h
        else:
            out = self.fc_mean(out)
            return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_directions = 2 if self.rnn.bidirectional else 1
        if isinstance(self.rnn, nn.LSTM):
            hidden = (weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_())
        else:
            hidden = weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_()
        return hidden 


def collate_fn(batch):
    # Separate the sequence data and the targets
    sequences, cat_seq, targets, time_range, loc_key = zip(*batch)

    # Pad the sequences
    sequences = pad_sequence(sequences, batch_first=True)

    # Pad the categorical sequences
    cat_seq = pad_sequence([torch.Tensor(t) for t in cat_seq], batch_first=True, padding_value=-1) # Check it

    # Pad the targets with -1
    targets = pad_sequence([torch.Tensor(t) for t in targets], batch_first=True, padding_value=-1) # Check it 

    return sequences, cat_seq, targets, time_range, loc_key


def compute_error_table(seats_true, seats_pred, seats_pred_std):
    '''
    Compute the error table for the given true and predicted values.
    '''
    all_metrics = []

    for i in range(seats_true.shape[1]):
        # Get the seats for the current quarter
        quarter_seats_true = torch.tensor(seats_true[:, i])
        quarter_seats_pred = torch.tensor(seats_pred[:, i])
        quarter_seats_pred_std = torch.tensor(seats_pred_std[:, i])

        abs_error = torch.abs(quarter_seats_pred - quarter_seats_true)
        abs_percent_error = 100 * abs_error / torch.clamp(quarter_seats_true, min=1e-9)
        
        # Calculate the metrics
        metrics = {
            'MAE': abs_error.mean().item(),
            'RMSE': torch.sqrt((abs_error**2).mean()).item(),
            'MAPE': abs_percent_error.mean().item(),
            # 'sMAPE': 200 * abs_error / (quarter_seats_true + quarter_seats_pred).mean().item(),
            'std': quarter_seats_pred_std.mean().item(),
            '<1%': (abs_percent_error < 1).float().mean().item(),
            '<5%': (abs_percent_error < 5).float().mean().item(),
            '<10%': (abs_percent_error < 10).float().mean().item(),
            '<20%': (abs_percent_error < 20).float().mean().item()
        }      

        # Convert metrics to a pandas DataFrame and append to list
        all_metrics.append(pd.DataFrame(metrics, index=[i]))

    # Concatenate all the DataFrames in the list
    error_table = pd.concat(all_metrics)  
    
    return error_table


def calculate_quarters(pred_num_quarters, seq_num, start_quarter='Q4 2022'):
    # Extract quarter and year from the start quarter
    qtr, year = start_quarter.split(' ')
    qtr = int(qtr[1])  # Convert 'Qx' to an integer
    year = int(year)

    # Calculate the boundary quarter
    qtr -= pred_num_quarters
    while qtr < 1:
        qtr += 4
        year -= 1
    boundary_quarter = f'Q{qtr} {year}'

    # Calculate the test boundary quarter
    qtr -= (seq_num - 1)
    while qtr < 1:
        qtr += 4
        year -= 1
    test_boundary_quarter = f'Q{qtr} {year}'

    # Calculate the test data
    qtr += pred_num_quarters  # Add one quarter to get the start of test data
    while qtr > 4:
        qtr -= 4
        year += 1
    test_data = f'Q{qtr} {year}'

    return boundary_quarter, test_boundary_quarter, test_data


class MSELossWithPenalty(nn.Module):
    def __init__(self, penalty_weight=0.00001):
        super(MSELossWithPenalty, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, outputs, targets):
        # Calculate the MSE loss
        loss = self.mse_loss(outputs, targets)

        # Add a penalty for negative predictions
        negative_penalty = self.penalty_weight * torch.mean(F.relu(-outputs))

        # Return the total loss
        return loss + negative_penalty


def train(train_loader, net, seat_scaler, lr=0.001, device="cpu", batch_size=10, 
          epochs=10, momentum=0.95, save_model=False, resume_training=False,
          MSE=True):
    
    net = net.to(device)

    if not MSE:
        criterion = nn.GaussianNLLLoss()
    else:
        criterion = nn.MSELoss()
        
    start_epoch = -1

    # Define the Adam optimizer
    optimizer =torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-6, betas=(momentum, 0.999))
    if resume_training:
        checkpoint = torch.load('./checkpoint/checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] - 1
        print("Resuming training from epoch {}.".format(start_epoch))
        epochs = epochs + start_epoch + 1

    # Record the training loss every 50 iterations
    train_loss = []
    iter_record = []

    # Training loop
    for epoch in range(start_epoch + 1, epochs):
        print("Epoch: {}".format(epoch+1))
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Get the input variable and seats(y)
            # inputs, seats = data
            inputs, cat_inputs, seats, time_range, loc_key = data
            inputs = inputs.float().to(device)
            # cat_inputs = cat_inputs.long().to(device)
            cat_inputs = cat_inputs.to(device)
            seats = seats.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Dynamically get the batch size based on input size
            current_batch_size = inputs.size(0)

            # initialize the hidden state
            h = net.init_hidden(current_batch_size)

            if isinstance(h, tuple):
                h = tuple([each.to(device) for each in h])
            else:
                h = h.to(device)

            h = tuple([each.data for each in h]) if isinstance(h, tuple) else h.data
            
            if not MSE:
                # Forward pass, backward pass, and optimize
                mean, std, h = net(inputs, cat_inputs, h)
                loss = criterion(mean, seats, std)
            else:
                mean, h = net(inputs, cat_inputs, h)
                loss = criterion(mean, seats)

            loss.backward()  # Backward pass            
            optimizer.step() # Update the parameters

            # print(loss.item()) # Print the loss only for debugging purposes

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch+1, i+1, running_loss/100))
                train_loss.append(running_loss/100)
                iter_record.append(i + epoch*len(train_loader))
                running_loss = 0.0
        
        # Save the model
        if save_model:
            torch.save(net.state_dict(), 'model.pth')

        # save the checkpoint every 5 epochs
        if epoch % 5 == 4:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if not os.path.isdir('./checkpoint'):
                os.mkdir('./checkpoint')
            torch.save(checkpoint, './checkpoint/checkpoint_{}.pth'.format(epoch+1))

    print('Finished Training')

    final_error_table = evaluate_model(train_loader, net, seat_scaler, device=device, MSE=MSE)
    print(final_error_table)

    return train_loss, iter_record


def evaluate_model(loader, net, seat_scaler, device="cpu", MSE=True, n_times=1):
    # Set the network to evaluation mode

    if not MSE:
        net.eval()
    else:
        net.train()

    all_outputs = []
    all_seats = []
    all_std = []

    num_batches = len(loader)

    # Disable gradient computation
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Print progress
            if i % 500 == 499:
                print(f'Evaluating: batch {i+1} of {num_batches}')

            # Get the inputs and targets
            # inputs, seats, time_range = data
            inputs, cat_inputs, seats, time_range, loc_key = data
            inputs = inputs.float().to(device)
            cat_inputs = cat_inputs.to(device)
            seats = seats.float().to(device)

            # Dynamically get the batch size based on input size
            current_batch_size = inputs.size(0)

            # Initialize the hidden state
            h = net.init_hidden(current_batch_size)
            if isinstance(h, tuple):
                h = tuple([each.to(device) for each in h])
            else:
                h = h.to(device)

            h = tuple([each.data for each in h]) if isinstance(h, tuple) else h.data

            if not MSE:
                # Forward pass
                mean, std, h = net(inputs, cat_inputs, h)

                # Collect outputs and seats
                all_outputs.append(mean.cpu().numpy())
                all_seats.append(seats.cpu().numpy())
                all_std.append(std.cpu().numpy())
            else:
                # Forward pass for Monte Carlo Dropout
                preds = [net(inputs, cat_inputs, h) for _ in range(n_times)]
                preds_tensor = torch.stack([pred[0] for pred in preds])
                # preds_tensor = torch.stack([p for p, _ in preds])

                # Calculate the mean and std for Monte Carlo Dropout
                mc_mean = preds_tensor.mean(dim=0).detach().cpu().numpy()
                mc_std = preds_tensor.std(dim=0).detach().cpu().numpy()

                # Collect outputs and seats
                all_outputs.append(mc_mean)
                all_seats.append(seats.cpu().numpy())
                all_std.append(mc_std)

    # Convert list of arrays to single array
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_seats = np.concatenate(all_seats, axis=0)
    all_std = np.concatenate(all_std, axis=0)

    # Initialzie arrays to hold the unscaled outputs and seats
    all_outputs_unscaled = np.zeros_like(all_outputs)
    all_seats_unscaled = np.zeros_like(all_seats)
    all_std_unscaled = np.zeros_like(all_std)

    # Inverse scale each quarter separately
    for i in range(all_outputs.shape[1]):
        all_outputs_unscaled[:, i] = seat_scaler.inverse_transform(all_outputs[:, i].reshape(-1, 1)).flatten()
        all_seats_unscaled[:, i] = seat_scaler.inverse_transform(all_seats[:, i].reshape(-1, 1)).flatten()
        all_std_unscaled[:, i] = all_std[:, i] * seat_scaler.scale_

    # Calculate the deviation and print the table
    final_error_table = compute_error_table(all_seats_unscaled, all_outputs_unscaled, all_std_unscaled)

    return final_error_table


def main_program(args, folder_path, seats_file_name, perf_file_name):
    x_features = [
            "Miles", "Deps/Day", "Seats/Day", "Seats/Dep", "Pax/Day", "Pax/Dep",  # 6/6
            "Load Factor", "Lcl %", "Local Pax/Day", "Lcl Fare", "Seg Fare", "Sys Fare",  # 6/12
            "Yield", "SLA Yield", "PRASM", "SLA PRASM", "% Free", "% Lcl Free", "Flights", # 7/19
            "Seats", "ASMs", "orig_lat", "orig_lon", "dest_lat", "dest_lon",  # 6/25
            "al_code", "orig_code", "dest_code", # 3/28
            "if_hub", "al_type",    # 2/30
            "num_comp", "mkt_share", "mkt_size", # 3/33
            "g1_o", "g2_o", "log_o", "state_o", # 4/37
            "g1_d", "g2_d", "log_d", "state_d", # 4/41
        ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args is None:
        print("Using default parameters since no arguments are provided.")
        resume_training=False
        MSE_or_GaussianNLLLoss="MSE"

        pred_num_quarters=4
        seq_num=10
        if_add_time_info=False
        
        learning_rate=0.00001
        momentum=0.95
        batch_size=32
        epochs=25
        num_workers=4
        shuffle=True
        fixed_seed=True
        rnn_type="LSTM"
        
        n_layers=4
        drop_prob=0.3
        num_heads=6

        start_year = 2016
    else:
        print("Using the provided arguments.")
        # Control if resume training
        resume_training = args.resume_training
        MSE_or_GaussianNLLLoss = args.MSE_or_GaussianNLLLoss

        pred_num_quarters = args.pred_num_quarters # number of quarters to predict
        seq_num = args.seq_num # number of quarters in a sequence
        if_add_time_info = args.if_add_time_info # if add time information

        # Set NN parameters
        learning_rate = args.learning_rate
        momentum = args.momentum
        batch_size = args.batch_size
        epochs = args.epochs
        num_workers = args.num_workers
        shuffle = args.shuffle
        fixed_seed = args.fixed_seed
        rnn_type = args.rnn_type

        n_layers = args.n_layers
        drop_prob = args.drop_prob
        num_heads = args.num_heads

        start_year = args.start_year

    ############################# start training #############################

    # record the start time
    start_time = time.time()
    print("-------- Start ----------")

    # Check if "training_data.csv", "testing_data.csv" and "scaled_data.csv" exist
    if (not os.path.exists("training_data.csv") or 
        not os.path.exists("testing_data.csv") or 
        not os.path.exists("scaled_data.csv")):
        debug = False
    else:
        debug = True

    # Set random seed
    if fixed_seed:
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if device == "cuda":
            torch.cuda.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    print(f"Using {device} for training.")

    # Load the dataset
    data_format = DatasetFormation(folder_path, seats_file_name, perf_file_name, x_features, 
                                   if_add_time_info=if_add_time_info, sequence_length=seq_num, 
                                   pred_num_quarters=pred_num_quarters, start_year=start_year) # set 2015 temporarily to limit the data size
    if not debug:
        boundary_quarter, test_boundary_quarter, apply_data_boundary = calculate_quarters(pred_num_quarters, seq_num)
        data_format.one_step_process(boundary_quarter=boundary_quarter, test_boundary_quarter=test_boundary_quarter)
        # boundary_quarter = Q4 2022 - pred_num_quarters
        # test_boundary_quarter = Q4 2022 - seq_len - pred_num_quarters
        # data_format.load_scaled_data_val(load_apply_data=load_apply_data, test_date="Q3 2020")
        # In applying format, we enter test_data = Q4 2022 - seq_len
    else:
        data_format.load_scaled_data()
        data_format.final_preparation()
    full_dataset = data_format.full_df

    # create the dataloader with the training data
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, collate_fn=collate_fn)

    # Create the model
    input_dim = len(x_features) + int(if_add_time_info) + 4
    print(f"Input dimension: {input_dim}")
    
    net = RNNNet(cat_feat=data_format.cat_features, cat_mapping=data_format.cat_mapping,
             embed_dim_mapping=data_format.embed_dim_mapping,
             input_dim=input_dim, hidden_dim=300, output_dim=pred_num_quarters,
             n_layers=n_layers, drop_prob=drop_prob, rnn_type=rnn_type,
             bidirectional=True, num_heads=num_heads, 
             if_skip=True, if_feed_drop=True, if_feed_norm=True,
             MSE=(MSE_or_GaussianNLLLoss == "MSE"))
    
    # Train the model for the first time
    train_loss, iter_record = train(dataloader, net, seat_scaler=data_format.seat_scaler ,lr=learning_rate, 
                                    device=device, batch_size=batch_size, epochs=epochs, momentum=momentum, 
                                    save_model=True, resume_training=resume_training,
                                    MSE=(MSE_or_GaussianNLLLoss == "MSE"))

    # Plot the training loss
    plt.plot(iter_record, train_loss, label="Training loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    # Define the fig name based on the parameters
    fig_name = f"lr_{learning_rate}_momentum_{momentum}_batch_size_{batch_size}_epochs_{epochs}.png"
    plt.savefig(fig_name)
    # plt.savefig("training_loss.png")
    plt.show()

    # calculate the time used in minutes
    end_time = time.time()
    time_used = (end_time - start_time) / 60
    print(f"Time used: {time_used} minutes")


if __name__ == "__main__":
    # Set basic parameters
    folder_path = r'C:\Users\qilei.zhang\OneDrive - Frontier Airlines\Documents\Data\USconti'
    seats_file_name = r'\Schedule_Monthly_Summary_Report_Conti.csv'
    perf_file_name = r'\Airline_Performance_Report_Conti.csv'

    # Check if parameters.json file exists, if not create one with default values.
    if not os.path.exists('parameters.json'):
        parameters = {
            "resume_training": False,
            "MSE_or_GaussianNLLLoss": "MSE",
            "pred_num_quarters": 3,
            "seq_num": 10,
            "if_add_time_info": False,
            "learning_rate": 0.00001,
            "momentum": 0.95,
            "batch_size": 32,
            "epochs": 1,
            "num_workers": 1,
            "shuffle": True,
            "fixed_seed": True,
            "rnn_type": "LSTM",
            "n_layers": 4,
            "drop_prob": 0.3,
            "num_heads": 6,
            "start_year": 2016,
        }
        with open('parameters.json', 'w') as f:
            json.dump(parameters, f)
    
    # Load parameters from the JSON file.
    with open('parameters.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))
    
    main_program(args, folder_path, seats_file_name, perf_file_name)

  