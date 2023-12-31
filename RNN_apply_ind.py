"""
This file is used to apply the trained RNN model to the test data (2023Q1 and 2023Q2)

"""

# Importing the libraries
import os
import time
import random
import joblib
import pandas as pd
import numpy as np
import torch
import pickle
import seaborn as sns
import argparse
import json

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error

from RNN_model import DatasetFormation, RNNNet, collate_fn, compute_error_table, calculate_quarters
# from RNN_ana import DataAna


# Inherit the DatasetFormation class
class Validation(DatasetFormation):
    def __init__(self, folder_path, seats_file_name, perf_file_name, x_features, apply_file_name, *args, **kwargs):
        super().__init__(folder_path, seats_file_name, perf_file_name, x_features, *args, **kwargs)

        # Load the label encoder
        le_airports_root = self.data_root + 'le_airports.pkl'
        le_airlines_root = self.data_root + 'le_airlines.pkl'
        self.le_airports = joblib.load(le_airports_root)
        self.le_airlines = joblib.load(le_airlines_root)
        
        # Load the mappings
        cat_mapping_root = self.data_root + 'cat_mapping.pkl'
        with open(cat_mapping_root, "rb") as f:
            self.cat_mapping = pickle.load(f)
        embed_dim_mapping_root = self.data_root + 'embed_dim_mapping.pkl'
        with open(embed_dim_mapping_root, "rb") as f:
            self.embed_dim_mapping = pickle.load(f)
        print("Dimension mapping loaded.")

        # apply_file_name = 'Schedule_Monthly_Summary_2023Q12.csv'
        self.apply_sch_df = pd.read_csv(self.root + apply_file_name)
        self.apply_sch_df['SortDate'] = self.apply_sch_df['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))

    def create_real_test_data(self, test_date='Q1 2021', start_quarter='Q1 2023'):
        """
        test_date: the quarter and year that starts the test data, 
        i.e., the last quarter data available minus the sequence length
        """
        self.clean_data()
        print("The data is cleaned.")

        self.assgin_geo_features()
        print("The geo features are assigned.")

        # self.transform_od_al()
        # Create a column of encoded airports and airlines
        self.df['orig_code'] = self.le_airports.transform(self.df['Orig'])
        self.df['dest_code'] = self.le_airports.transform(self.df['Dest'])
        self.df['al_code'] = self.le_airlines.transform(self.df['Mkt Al'])
        print("The origin, destination and airline are encoded.")

        self.attach_coordinates()
        print("The airports coordinates are calculated.")

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

        # get the test data after 2022 Q3
        boundary_num = int(start_quarter.split(' ')[1]) * 4 + int(start_quarter.split(' ')[0][1])
        test_date_num = int(test_date.split(' ')[1]) * 4 + int(test_date.split(' ')[0][1])
        self.df['SortDate'] = self.df['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))
        self.test_df = self.df[self.df['SortDate'] >= test_date_num]
        self.test_df = self.test_df[self.test_df['SortDate'] < boundary_num].copy()
        
        # reset the index
        self.test_df.reset_index(drop=True, inplace=True)

        # save the test data
        apply_file_save_root = self.data_root + 'applying_data.csv'
        self.test_df.to_csv(apply_file_save_root, index=False)

        print("The test data is created.")

    def load_scaled_data_val(self, train_filename='./data/training_data.csv', 
                             test_filename='./data/testing_data.csv', 
                             scaled_filename='./data/scaled_data.csv', 
                             apply_filename='./data/applying_data.csv',
                             load_apply_data=True,
                             test_date='Q1 2021',
                             on_apply_data=True,
                             start_quarter='Q1 2023'):
        # load the original training data first to get the scaler
        self.train_df = pd.read_csv(train_filename)

        if on_apply_data:
            # load the testing data (or so-called validation data)
            if not load_apply_data:
                # self.test_df = pd.read_csv(test_filename) # Not needed
                self.create_real_test_data(test_date, start_quarter)
            else:
                self.test_df = pd.read_csv(apply_filename)
        else:
            self.test_df = pd.read_csv(test_filename) # now data

        self.scaled_df = pd.read_csv(scaled_filename)
        print("Original data loaded.")

        # rebuild the main scaler
        self.scaler.fit(self.train_df[self.x_features_without_seats])
        print("Main scaler rebuilt.")
        
        # rebuild the seat scaler
        self.seat_scaler.fit(self.train_df['Seats'].values.reshape(-1,1))
        print("Seat scaler rebuilt.")

        # rebuild the date scaler
        self.date_scaler.fit(self.scaled_df['Date_delta'].values.reshape(-1,1))

        # load the testing data, not scaled yet
        self.scaler_data_val()
        print("Validation/Test data scaled.")
        self.create_date_features_val()
        print("Date features created.")
        # self.save_scaled_data_val()  # temporary comment
        
        # prepare the validation data
        self.final_preparation_val(boundary_quarter=start_quarter, on_apply_data=on_apply_data)
        # self.final_preparation_val_new(boundary_quarter=start_quarter, on_apply_data=on_apply_data)
        print("Validation data prepared.")

    def scaler_data_val(self):
        """
        Scale the validation data using the scaler from training data
        """
        self.scaled_df = self.test_df.copy()

        scaled_features = self.scaler.transform(self.scaled_df[self.x_features_without_seats])

        # scale the 'seats' feature separately and store the transformation
        self.scaled_df['Seats'] = self.seat_scaler.transform(self.scaled_df['Seats'].values.reshape(-1,1))

        self.scaled_df[self.x_features_without_seats] = scaled_features

        # print("Apply data scaled.")

    def create_date_features_val(self, relative_date="2003-01-01"):
        """
        Convert the date to a numeric value
        Note: Relative date will be set to 2001-01-01 by default
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
            self.scaled_df['time_scaled'] = self.date_scaler.transform(self.scaled_df['Date_delta'].values.reshape(-1,1))
        else:
            print("The dataset has not been scaled yet.")

    def final_preparation_val(self, boundary_quarter='Q1 2023', on_apply_data=True):
        # Create binary features for the quarter
        # self.scaled_df = pd.get_dummies(self.scaled_df, columns=['year', 'quarter'])
        # group the dataframe for each airline and each route
        self.scaled_df = pd.get_dummies(self.scaled_df, columns=['quarter'])
        datasets = []
        for key, route_df in self.scaled_df.groupby(["Mkt Al", "Orig", "Dest"]):
            # Get the value of the seats from applying data
            seats_df = self.apply_sch_df[(self.apply_sch_df['Mkt Al'] == key[0]) & (self.apply_sch_df['Orig'] == key[1]) & (self.apply_sch_df['Dest'] == key[2])]

            if on_apply_data:
                if len(route_df) < (self.seq_len + self.skip_quarters):
                    continue

                # Get the end quarter of seats (start quarter + prediction quarters)
                qtr, year = boundary_quarter.split(' ')
                qtr = int(qtr[1])  # Convert 'Qx' to an integer
                year = int(year)

                if self.skip_quarters==0:
                    qtr += 1
                    while qtr > 4:
                        qtr -= 4
                        year += 1
                    start_quarter = f'Q{qtr} {year}'
                else:
                    start_quarter = boundary_quarter

                qtr += (self.n_future - 1)
                while qtr > 4:
                    qtr -= 4
                    year += 1
                end_quarter = f'Q{qtr} {year}'

                seats_start_date = int(start_quarter.split(' ')[1]) * 4 + int(start_quarter.split(' ')[0][1])
                seats_end_date = int(end_quarter.split(' ')[1]) * 4 + int(end_quarter.split(' ')[0][1])
                # seats_df[seats_df['SortDate'] >= seats_start_date]
                seats_df = seats_df[(seats_df['SortDate'] >= seats_start_date) & (seats_df['SortDate'] <= seats_end_date)]
                
                if len(seats_df) > 0:
                    # Get the seats values to list
                    seats_list = seats_df['Seats'].values

                    # fill out 0 for the seats_list if the seats_list is not long enough
                    if len(seats_df) < self.n_future:
                        seats_list = list(seats_list) + [0] * (self.n_future - len(seats_df))
                else:
                    seats_list = [0] * self.n_future # in case there is no seats data for the route
            else:
                if len(route_df) < self.seq_len:
                    continue

                # Get the end quarter of seats (start quarter + prediction quarters)
                qtr, year = boundary_quarter.split(' ')
                qtr = int(qtr[1])  # Convert 'Qx' to an integer
                year = int(year)

                # if self.skip_quarters==0:
                qtr += 1
                while qtr > 4:
                    qtr -= 4
                    year += 1
                start_quarter = f'Q{qtr} {year}'
                
                qtr += (self.skip_quarters + self.n_future - 1)
                while qtr > 4:
                    qtr -= 4
                    year += 1
                end_quarter = f'Q{qtr} {year}'

                seats_start_date = int(start_quarter.split(' ')[1]) * 4 + int(start_quarter.split(' ')[0][1])
                seats_end_date = int(end_quarter.split(' ')[1]) * 4 + int(end_quarter.split(' ')[0][1])
                # seats_df[seats_df['SortDate'] >= seats_start_date]
                seats_df = seats_df[(seats_df['SortDate'] >= seats_start_date) & (seats_df['SortDate'] <= seats_end_date)]

                if len(seats_df) < self.skip_quarters:
                    continue

                if len(seats_df) < (self.skip_quarters + self.n_future):
                    # seats_list = list(seats_list) + [0] * (self.n_future - len(seats_df))
                    seats_list = list(seats_df['Seats'].values) + [0] * (self.skip_quarters + self.n_future - len(seats_df))    

            route_df = route_df.sort_values("Date_delta")
            datasets.append(FlightDataset(route_df, self.seq_len, self.num_features,
                                          self.cat_features, self.skip_quarters,
                                          time_add=self.time_add, seats_values=seats_list,
                                          n_future=self.n_future, on_apply_data=on_apply_data,
                                          seat_scaler=self.seat_scaler))
        self.full_df = torch.utils.data.ConcatDataset(datasets)

    def save_scaled_data_val(self, filename='scaled_data_apply.csv'):
        filename = self.data_root + filename
        self.scaled_df.to_csv(filename, index=False)
        print("Scaled applying data saved.")

    def get_next_quarter(self, qtr, year, increment=1):
        qtr += increment
        while qtr > 4:
            qtr -= 4
            year += 1
        return f'Q{qtr} {year}'

    def quarter_to_date(self, quarter):
        qtr, year = quarter.split(' ')
        qtr = int(qtr[1])
        year = int(year)
        return year * 4 + qtr
    
    def get_seats_df(self, seats_df, start_quarter, end_quarter):
        seats_start_date = self.quarter_to_date(start_quarter)
        seats_end_date = self.quarter_to_date(end_quarter)
        seats_df = seats_df[(seats_df['SortDate'] >= seats_start_date) & (seats_df['SortDate'] <= seats_end_date)]
        return seats_df

    def final_preparation_val_new(self, boundary_quarter='Q1 2023', on_apply_data=True):
        self.scaled_df = pd.get_dummies(self.scaled_df, columns=['quarter'])
        datasets = []

        qtr, year = boundary_quarter.split(' ')
        qtr = int(qtr[1])
        year = int(year)

        if on_apply_data:
            start_quarter = self.get_next_quarter(qtr, year) if self.skip_quarters == 0 else boundary_quarter
            end_quarter = self.get_next_quarter(qtr, year, self.n_future - 1)
        else:
            start_quarter = self.get_next_quarter(qtr, year)
            end_quarter = self.get_next_quarter(qtr, year, self.skip_quarters + self.n_future - 1)

        for key, route_df in self.scaled_df.groupby(["Mkt Al", "Orig", "Dest"]):
            # Get the value of the seats from applying data
            seats_df = self.apply_sch_df[
                (self.apply_sch_df['Mkt Al'] == key[0]) & 
                (self.apply_sch_df['Orig'] == key[1]) & 
                (self.apply_sch_df['Dest'] == key[2])
            ]

            seats_df = self.get_seats_df(seats_df, start_quarter, end_quarter)

            if on_apply_data:
                if len(route_df) < (self.seq_len + self.skip_quarters):
                    continue
                seats_list = seats_df['Seats'].values if len(seats_df) > 0 else [0] * self.n_future
                # fill out 0 for the seats_list if the seats_list is not long enough
                if len(seats_df) < self.n_future:
                    seats_list = list(seats_list) + [0] * (self.n_future - len(seats_df))
            else:
                if len(route_df) < self.seq_len:
                    continue
                if len(seats_df) < self.skip_quarters:
                    continue

                seats_list = list(seats_df['Seats'].values) + [0] * (self.skip_quarters + self.n_future - len(seats_df))

            route_df = route_df.sort_values("Date_delta")
            datasets.append(
                FlightDataset(route_df, self.seq_len, self.num_features,
                            self.cat_features, self.skip_quarters,
                            time_add=self.time_add, seats_values=seats_list,
                            n_future=self.n_future, on_apply_data=on_apply_data,
                            seat_scaler=self.seat_scaler)
            )

        self.full_df = torch.utils.data.ConcatDataset(datasets)


class QuarterFilling:
    def __init__(self, seq_len, n_future, skip_quarters, on_apply_data=True):
        self.seq_len = seq_len
        self.n_future = n_future
        self.skip_quarters = skip_quarters
        self.on_apply_data = on_apply_data

    def fill_missing_quarters(self, route_df):
        # Convert 'Date' to a string format for consistency
        route_df['Date'] = route_df['Date'].astype(str)
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        # Define a function to generate the next quarter
        def next_quarter(q, y):
            if q == "Q4":
                return "Q1", y + 1
            else:
                next_q = quarters[quarters.index(q) + 1]
                return next_q, y

        # Generate a list of all quarters present in route_df
        all_dates = list(route_df['Date'])
        missing_quarters = []

        # Find missing quarters but not for the last few records
        if self.on_apply_data:
            range_end = len(all_dates) - self.skip_quarters - 1
        else:
            range_end = len(all_dates) - 1

        for i in range(range_end):
            q, y = all_dates[i].split()
            y = int(y)
            next_q, next_y = next_quarter(q, y)

            if f"{next_q} {next_y}" != all_dates[i + 1]:
                missing_quarters.append(f"{next_q} {next_y}")

        # Columns that need to have the same values for all rows
        consistent_columns = ['Mkt Al', 'Orig', 'Dest', 'Miles', 'Alpha', 'g1_o', 'g2_o', 'log_o', 'state_o',
                            'g1_d', 'g2_d', 'log_d', 'state_d', 'orig_code', 'dest_code', 'al_code',
                            'orig_lat', 'orig_lon', 'dest_lat', 'dest_lon', 'if_hub', 'al_type']

        # Get the values for these columns from the first row of the DataFrame
        consistent_values = route_df.loc[0, consistent_columns].to_dict()

        # Fill in the missing quarters
        for missing in missing_quarters:
            q, y = missing.split()
            zero_row = pd.Series({col: -1e10 for col in route_df.columns}, name=missing)
            zero_row['Date'] = missing
            # Set all quarter columns to False
            for i in [1, 2, 3, 4]:
                zero_row[f'quarter_{i}'] = False
            zero_row[f'quarter_{quarters.index(q) + 1}'] = True  # Set the correct quarter to true

            # Set consistent values for the specified columns
            for col, val in consistent_values.items():
                zero_row[col] = val

            # route_df = route_df.append(zero_row, ignore_index=True)
            route_df = pd.concat([route_df, pd.DataFrame([zero_row])]).reset_index(drop=True)

        # Sort by Date after adding missing quarters
        route_df = route_df.sort_values(by="Date").reset_index(drop=True)

        # If length is still less, prepend rows with zeros for previous quarters
        if self.on_apply_data:
            range_fill = self.seq_len + self.skip_quarters
        else:
            range_fill = self.seq_len

        while len(route_df) < (range_fill):
            first_q, first_y = route_df.iloc[0]['Date'].split()
            first_y = int(first_y)

            if first_q == "Q1":
                prev_q, prev_y = "Q4", first_y - 1
            else:
                prev_q = quarters[quarters.index(first_q) - 1]
                prev_y = first_y

            zero_row = pd.Series({col: -1e10 for col in route_df.columns}, name=f"{prev_q} {prev_y}")
            zero_row['Date'] = f"{prev_q} {prev_y}"
            for i in [1, 2, 3, 4]:
                zero_row[f'quarter_{i}'] = False
            zero_row[f'quarter_{quarters.index(prev_q) + 1}'] = True  # Set the correct quarter to 1

            for col, val in consistent_values.items():
                zero_row[col] = val
            route_df = pd.concat([pd.DataFrame([zero_row]), route_df]).reset_index(drop=True)

        return route_df

        
class FlightDataset(Dataset):
    """
    Create a dataset that can be used for dataloading.
    """
    def __init__(self, df, sequence_length, num_feat, cat_feat, skip_qrts,
                 time_add=True, seats_values=None, n_future=2, on_apply_data=True,
                 seat_scaler=None):
        self.df = df
        self.sequence_length = sequence_length
        self.num_features = num_feat
        self.cat_features = cat_feat
        self.skip_qrts = skip_qrts

        self.dummy_quarter = [*[f"quarter_{i}" for i in range(1, 5)]]

        if time_add:
            self.num_features = self.num_features + self.dummy_quarter + ['time_scaled']
        else:
            self.num_features = self.num_features + self.dummy_quarter

        self.seats_values = seats_values
        self.on_apply_data = on_apply_data
        self.n_future = n_future
        self.seat_scaler = seat_scaler

    def __len__(self):
        if self.on_apply_data:
            return len(self.df) - self.sequence_length - self.skip_qrts + 1
        else:
            return len(self.df) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Get the relevant slice of the dataframe
        df_slice = self.df.iloc[idx : idx + self.sequence_length]
        
        # Construct the sequence data
        sequence_data = df_slice[self.num_features].astype(float).values
        cat_sequence_data = torch.LongTensor(df_slice[self.cat_features].values)

        if self.skip_qrts > 0:
            if self.on_apply_data:
                # Get the relevant slice of the dataframe for the seats of skip quarters
                df_skip_slice = self.df.iloc[(idx + self.sequence_length) : 
                                            (idx + self.sequence_length + self.skip_qrts)]

                # attaching the skip quarter seats as a feature to the sequence data
                skip_values = df_skip_slice['Seats'].values.repeat(self.sequence_length).reshape(-1, self.sequence_length).T
                sequence_data = np.concatenate((sequence_data, skip_values), axis=1)
            else:
                # Get the relevant slice of the dataframe for the seats of skip quarters 
                # from the seats_values with the number of skip quarters
                df_skip_slice = self.seats_values[: self.skip_qrts]
                df_skip_slice = np.array(df_skip_slice).reshape(-1, 1)

                # use scaler to transform the skip quarter seats
                df_skip_slice = self.seat_scaler.transform(df_skip_slice)
                df_skip_slice = df_skip_slice.flatten()

                # attaching the skip quarter seats as a feature to the sequence data
                skip_values = df_skip_slice.repeat(self.sequence_length).reshape(-1, self.sequence_length).T
                sequence_data = np.concatenate((sequence_data, skip_values), axis=1)

        # Get the relevant slice of the dataframe for the target seats
        # df_target_slice = self.df.iloc[idx + self.sequence_length : idx + self.sequence_length]

        # Get the first row and last row of the time_scaled column
        first_row = df_slice['time_scaled'].iloc[0]
        last_row = df_slice['time_scaled'].iloc[-1]
        time_range = (first_row, last_row)
        loc_key = (df_slice['Mkt Al'].iloc[0], df_slice['Orig'].iloc[0], 
                   df_slice['Dest'].iloc[0], df_slice['Date'].iloc[-1]) # airline, origin, destination

        # "Seats" is the 19th column (0-indexed)
        # target_data = df_target_slice["Seats"].values 

        if self.on_apply_data:    
            target_data = self.seats_values
        else:
            target_data = self.seats_values[self.skip_qrts:]

        # Return the sequence data and the target value. "Seats" is the 19th column (0-indexed)
        # return torch.from_numpy(sequence_data[:-1]), torch.tensor(sequence_data[-1, 18])  
        return torch.from_numpy(sequence_data), cat_sequence_data, torch.tensor(target_data), time_range, loc_key
    

def validation(loader, net, seat_scaler, device='cpu', n_times=1, MSE=True, tune_folder=None, on_apply_data=True):
    # Set the network to evaluation mode
    net = net.to(device) 

    if MSE:
        # net.train()
        net.eval()
    else:
        net.eval()

    all_outputs = []
    all_seats = []
    all_std = []

    all_time_range = []
    all_loc_key = []

    # mc_means = []
    # mc_stds = []

    num_batches = len(loader)

    with torch.no_grad():
        i = 0
        for i, data in enumerate(loader):
            # Print progress
            i += 1
            if i % 100 == 99:
                print(f'Evaluating: batch {i+1} of {num_batches}')

            # Get the inputs and targets
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

            if MSE:
                # Forward pass for Monte Carlo Dropout
                preds = [net(inputs, cat_inputs, h) for _ in range(n_times)]

                # Convert list of tensor outputs to tensor
                # preds_tensor = torch.stack([p for p, _ in preds])
                preds_tensor = torch.stack([pred[0] for pred in preds])

                # Calculate the mean and std for Monte Carlo Dropout
                mc_mean = preds_tensor.mean(dim=0).detach().cpu().numpy()
                mc_std = preds_tensor.std(dim=0).detach().cpu().numpy()

                all_outputs.append(mc_mean)
                all_std.append(mc_std)
                all_seats.append(seats.cpu().numpy())
                
            else:
                # Forward pass
                # mean, std, h = net(inputs, h)
                mean, std, h = net(inputs, cat_inputs, h)

                # Collect outputs and seats
                all_outputs.append(mean.cpu().numpy())
                all_seats.append(seats.cpu().numpy())
                all_std.append(std.cpu().numpy())

                # # Collect inputs
                # all_inputs.append(inputs.cpu().numpy())

            # Collect time information
            all_time_range.append(time_range) # format [(begin, end), ...]

            # Collect loc_key
            all_loc_key.append(loc_key)

    # Concatenate outputs and seats
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_seats = np.concatenate(all_seats, axis=0)
    all_std = np.concatenate(all_std, axis=0)
    all_time_range = np.concatenate(all_time_range, axis=0)
    all_loc_key = np.concatenate(all_loc_key, axis=0)

    # # Concatenate mc_means and mc_stds
    # mc_means = np.concatenate(mc_means, axis=0)
    # mc_stds = np.concatenate(mc_stds, axis=0)

    # Initialzie arrays to hold the unscaled inputs, outputs and seats
    all_outputs_unscaled = np.zeros_like(all_outputs)
    all_seats_unscaled = np.zeros_like(all_seats)
    all_std_unscaled = np.zeros_like(all_std)
    # mc_means_unscaled = np.zeros_like(mc_means)
    # mc_stds_unscaled = np.zeros_like(mc_stds)

    # Inverse scale each quarter separately
    for i in range(all_outputs.shape[1]):
        all_outputs_unscaled[:, i] = seat_scaler.inverse_transform(all_outputs[:, i].reshape(-1, 1)).flatten()
        # all_seats_unscaled[:, i] = seat_scaler.inverse_transform(all_seats[:, i].reshape(-1, 1)).flatten()
        all_seats_unscaled[:, i] = all_seats[:, i]
        # rescale the std
        all_std_unscaled[:, i] = all_std[:, i] * seat_scaler.scale_

        # # rescale the mc_means and mc_stds
        # mc_means_unscaled[:, i] = seat_scaler.inverse_transform(mc_means[:, i].reshape(-1, 1)).flatten()
        # mc_stds_unscaled[:, i] = mc_stds[:, i] * seat_scaler.scale_

    # Remove 0 seats rows and corresponding outputs and std rows
    # Create a mask where no element in the rows of all_seats_unscaled is zero
    mask = np.all(all_seats_unscaled != 0, axis=1)

    # Filter the three arrays using the mask
    all_outputs_unscaled_t = all_outputs_unscaled[mask]
    all_std_unscaled_t = all_std_unscaled[mask]
    all_seats_unscaled_t = all_seats_unscaled[mask]

    # Calculate the deviation and print the table
    final_error_table = compute_error_table(all_seats_unscaled_t, 
                                            all_outputs_unscaled_t, 
                                            all_std_unscaled_t)
    # compute_error_table Needs update

    # Save the error table
    # error_table_filename = './results/error_table_apply.csv'
    if tune_folder is None:
        error_table_filename = './results/error_table_apply.csv'
    else:
        error_table_filename = f'./{tune_folder}/results/error_table_apply.csv'
    final_error_table.to_csv(error_table_filename, index=False)

    if on_apply_data:
        print(final_error_table)

    return all_outputs_unscaled, all_seats_unscaled, all_std_unscaled, all_loc_key


def create_route_dict(all_outputs_unscaled, all_seats_unscaled, all_std_unscaled, all_loc_key):
    result_dict = {}
    # Check if all four inputs have the same length
    assert len(all_outputs_unscaled) == len(all_seats_unscaled) == len(all_std_unscaled) == len(all_loc_key)
   
    for i in range(len(all_loc_key)):
        # if i in [2316, 2317]:
        #     print('found number')

        # Get the original categories
        al = all_loc_key[i][0]
        orig = all_loc_key[i][1]
        dest = all_loc_key[i][2]
        date = all_loc_key[i][3]

        # Create a key for the dictionary
        key = (al, orig, dest)
        
        # If the key is new, create a new entry in the dictionary
        if key not in result_dict:
            result_dict[key] = {
                'date': [],
                'outputs': [],
                'seats': [],
                'std': []
            }
        
        # Add the sequence to the dictionary
        result_dict[key]['date'].append(date)
        result_dict[key]['outputs'].append(all_outputs_unscaled[i])
        result_dict[key]['seats'].append(all_seats_unscaled[i])
        result_dict[key]['std'].append(all_std_unscaled[i])

    return result_dict


def route_dict_to_df(route_dict, skip_quarters=2):
    """
    transform the route dictionary to a dataframe
    but since it need a orignal dataframe to pair the data
    so it only apply to the training/validation data not to do prediction
    """
    i = 0
    print('There is {} routes in the route dictionary'.format(len(route_dict)))
    datasets = []
    for route, data in route_dict.items():
        i += 1
        if i % 100 == 0:
            print('Processing {}th route'.format(i))

        date = data['date']
        outputs = data['outputs']
        seats = data['seats']
        std = data['std']

        pred = {}
        truth = {}
        conf = {} # std
        
        for seq_index in range(len(outputs)):
            year = int(date[seq_index].split(' ')[1])
            quarter = int(date[seq_index].split(' ')[0][1])
            n_future = len(seats[seq_index])

            quarter += (skip_quarters)
            while quarter > 4:
                quarter -= 4
                year += 1
            # test_data = f'Q{qtr} {year}'

            for i in range(n_future):
                quarter += 1
                if quarter > 4:
                    quarter = 1
                    year += 1
                pred[(year, quarter)] = outputs[seq_index][i]
                truth[(year, quarter)] = seats[seq_index][i]
                conf[(year, quarter)] = std[seq_index][i]
        for year, quarter in pred.keys():
            datasets.append([route[0], route[1], route[2], year, quarter, truth[(year, quarter)], pred[(year, quarter)], conf[(year, quarter)]])
    df = pd.DataFrame(datasets, columns=['Mkt Al', 'Orig', 'Dest', 'year', 'quarter', 'Seats', 'pred', 'std'])
    
    return df 


def find_best_routes(year=2023, include_quarters=4, tune_folder=None):
    # df = pd.read_csv('./results/data_to_ana_apply.csv')
    if tune_folder is None:
        df = pd.read_csv('./results/data_to_ana_apply.csv')
    else:
        df = pd.read_csv(f'./{tune_folder}/results/data_to_ana_apply.csv')

    # Assuming df is your DataFrame
    df['percentage_error'] = abs(df['Seats'] - df['pred']) / df['Seats'] * 100

    df = df[(df['year']==year) & (df['quarter']<=include_quarters)]

    # Calculate the average percentage error for each route
    average_error = df.groupby(['Mkt Al', 'Orig', 'Dest'])['percentage_error'].mean().reset_index()
    average_error.columns = ['Mkt Al', 'Orig', 'Dest', 'average_percentage_error']

    # Merge average_error back into the original dataframe
    df = pd.merge(df, average_error, how='left', on=['Mkt Al', 'Orig', 'Dest'])

    # Group by 'Mkt Al', 'Orig' and 'Dest' and filter groups where all percentage errors are less than 5%
    def all_less_than_5_percent(x):
        return (x < 5).all()

    filtered_df = df.groupby(['Mkt Al', 'Orig', 'Dest']).filter(lambda x: all_less_than_5_percent(x['percentage_error']))

    # Get unique routes
    best_route = filtered_df[['Mkt Al', 'Orig', 'Dest']].drop_duplicates().reset_index(drop=True)

    # Save the best routes
    if tune_folder is None:
        best_route.to_csv('./results/best_route.csv', index=False)
    else:
        best_route.to_csv(f'./{tune_folder}/results/best_route.csv', index=False)


class DataAna():
    def __init__(self, ana_df_name, tune_folder=None):
        self.df = pd.read_csv(ana_df_name, index_col=0)
        self.competitors_quartiles = None
        self.seats_quartiles = None
        self.df_plot = self.df.copy()

        self.boundary_quarter = "Q1 2023"
        self.boundary_num = int(self.boundary_quarter.split(' ')[1]) * 4 + int(self.boundary_quarter.split(' ')[0][1])

        self.tune_folder = tune_folder

    def group_by_direction(self, group_bi_direction=True, seat_sum=False):
        if group_bi_direction:
            # Create a new column named 'route' containing sorted origin and destination
            self.df['route'] = self.df.apply(lambda row: ''.join(sorted([row['Orig'], row['Dest']])), axis=1)
            # Drop the 'Orig' and 'Dest' columns
            self.df = self.df.drop(columns=['Orig', 'Dest'])
            # Group by 'Mkt Al' (airline) and 'route', then aggregate by taking mean values for numerical columns
            grouped_df = self.df.groupby(['Mkt Al', 'route', 'Date']).mean().reset_index()

            if seat_sum:
                # We should sum up the 'Flights' and 'Seats' values after taking average as per your requirement
                flights_seats_df = self.df.groupby(['Mkt Al', 'route', 'Date'])[['Flights', 'Seats', 'pred']].sum().reset_index()
                # Merge the summed 'Flights' and 'Seats' back into the grouped_df
                grouped_df = pd.merge(grouped_df, flights_seats_df, on=['Mkt Al', 'route', 'Date'], suffixes=('', '_sum'))
                grouped_df['Flights'] = grouped_df['Flights_sum']
                grouped_df['Seats'] = grouped_df['Seats_sum']
                grouped_df.drop(['Flights_sum', 'Seats_sum'], axis=1, inplace=True)
        else:
            self.df['route'] = self.df['Orig'] + self.df['Dest']
            grouped_df = self.df.copy()
            
        self.df = grouped_df.copy()

    def calculate_competitors(self):
        # Count the unique 'Mkt Al' (airlines) per 'route'
        competitors_df = self.df.groupby(['route', 'Date'])['Mkt Al'].nunique().reset_index()
        competitors_df.columns = ['route', 'Date', 'Competitors']

        # Merge the competitors_df back into the original df
        self.df = pd.merge(self.df, competitors_df, on=['route','Date'], how='left')

    def classify_market_size(self):
        self.seats_quartiles = self.df['Seats'].quantile([.25, .5, .75]).values
        def market_size(seats):
            if seats <= self.seats_quartiles[0]:
                return 'tiny'
            elif seats <= self.seats_quartiles[1]:
                return 'small'
            elif seats <= self.seats_quartiles[2]:
                return 'large'
            else:
                return 'super'
        self.df['market_size'] = self.df['Seats'].apply(market_size)

    def classify_competitors(self, col_name='Competitors'):
        # We are assuming that df['Competitors'] is the number of competitors
        self.competitors_quartiles = self.df[col_name].quantile([.33, .66]).values
        def competition_level(competitors):
            if competitors <= self.competitors_quartiles[0]:
                return 'Few'
            elif competitors <= self.competitors_quartiles[1]:
                return 'Gen'
            else:
                return 'High'
        self.df['competition_level'] = self.df[col_name].apply(competition_level)

    def merge_previous_data(self, orig_df, if_group_bi_direction=True):
        """
        Merge the previous data with the current data
        """
        self.df = self.df.dropna(subset=['pred'])
        if 'TRUE' in self.df.columns:
            self.df = self.df.rename(columns={'TRUE': 'Seats'})
        if 'true' in self.df.columns:
            self.df = self.df.rename(columns={'true': 'Seats'})

        orig_df = orig_df[['Mkt Al', 'Orig', 'Dest', 'year', 'quarter', 'Seats']]
        self.df = pd.concat([self.df, orig_df], ignore_index=True)
        # self.df.sort_values(by=['Mkt Al', 'route', 'Date'], inplace=True)

        # Reset the index without the old index
        self.df = self.df.reset_index(drop=True)

        # self.df['route'] = self.df.apply(lambda row: ''.join(sorted([row['Orig'], row['Dest']])), axis=1)
        self.df['Date'] = self.df.apply(lambda row: 'Q' + str(row['quarter']) + ' ' + str(int(row['year'])), axis=1)

        # drop the year and quarter column
        self.df.drop(columns=['year', 'quarter'], inplace=True)

        self.group_by_direction(group_bi_direction=if_group_bi_direction, seat_sum=False)

        # Sort the dataframe by route and date
        self.df.sort_values(by=['Mkt Al', 'route', 'Date'], inplace=True)

        self.df_plot = self.df.copy()

        # Calculate the metrics 
        self.calculate_competitors()
        self.classify_competitors()
        self.classify_market_size()
        self.calculate_metrics_new()

        # Save the data
        if self.tune_folder is None:
            self.df.to_csv("./results/data_after_ana_app.csv", index=False)
        else:
            self.df.to_csv(f"./{self.tune_folder}/results/data_after_ana_app.csv", index=False)

    def calculate_metrics_new(self, pivot=False):
        """
        Calculate the date after 2023-01-01
        Change the self.boundary_quarter to change the default boundary quarter
        """
        output_ana_df = self.df.copy()
        output_ana_df = output_ana_df.dropna(subset=['pred'])
        output_ana_df['SortDate'] = output_ana_df['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))

        # output_ana_df = output_ana_df[output_ana_df['SortDate'] >= self.boundary_num]

        groups = output_ana_df.groupby(['Date', 'competition_level', 'market_size'])
        result = []
        for name, group in groups:
            y_true = group['Seats']
            y_pred = group['pred']
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # Calculate the average confidence interval
            std = group['std']
            avg_std = np.mean(std)

            # Calculate the percentage of error rates
            error_rate = np.abs((y_true - y_pred) / y_true)
            below_1 = np.sum(error_rate < 0.01) / len(error_rate) * 100
            below_5 = np.sum(error_rate < 0.05) / len(error_rate) * 100
            below_10 = np.sum(error_rate < 0.1) / len(error_rate) * 100
            below_20 = np.sum(error_rate < 0.2) / len(error_rate) * 100

            result.append([name[0], name[1], name[2], mae, rmse, mape, below_1, below_5, below_10, below_20, avg_std])

        result_df = pd.DataFrame(result, columns=['Date','competition_level', 'market_size', 'MAE', 'RMSE', 'MAPE', '<1%', '<5%', '<10%', '<20%', 'avg_std'])
        
        if pivot:
            # Pivot the DataFrame to have metrics as rows and combinations of competition_level and market_size as columns
            result_df = result_df.melt(id_vars=['competition_level', 'market_size'], var_name='Metric', value_name='Value')
            result_df = result_df.pivot_table(index='Metric', columns=['competition_level', 'market_size'], values='Value')

        # save the result
        if self.tune_folder is None:
            result_df.to_csv("./results/result_apply.csv", index=False)
        else:
            result_df.to_csv(f"./{self.tune_folder}/results/result_apply.csv", index=False)

        return result_df
    
    def plot_prediction(self, al, alpha, on_apply_data=True):
        """
        Use it after run calculate_metrics_new
        """
        # Captialize the input
        al = al.upper()
        alpha = alpha.upper()

        # Make sure aplha is a valid route
        alpha_list = [alpha[0:3], alpha[3:6]]
        alpha = ''.join(sorted(alpha_list))

        # Prepare the data
        df = self.df_plot.copy()
        # df = df[['Mkt Al', 'Orig', 'Dest', 'Date', 'Flights', 'Seats', 'pred', 'std']]
        # df['route'] =df.apply(lambda row: ''.join(sorted([row['Orig'], row['Dest']])), axis=1)
        # df = df.drop(columns=['Orig', 'Dest'])
        # df = df.groupby(['Mkt Al', 'route', 'Date']).mean().reset_index()  # Here df is the 10 quarter data

        # Filter dataframe based on Mkt Al, Orig, and Dest
        df_filtered = df.loc[(df['Mkt Al'] == al) & (df['route'] == alpha)].copy()

        if len(df_filtered) == 0:
            print(f'Route Not Existed for {al} {alpha}...')
            return
        elif len(df_filtered) < 10:
            print(f'Data not Enough {al} {alpha}...')
            return
        
        # output_ana_df_filtered = output_ana_df.loc[(output_ana_df['Mkt Al'] == al) & (output_ana_df['route'] == alpha)].copy()
        # # Create a separate column for sorting
        df_filtered['SortDate'] = df_filtered['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))

        # drop df_filtered `Flights` column
        # df_filtered = df_filtered.drop(columns=['Flights'])

        # # add output_ana_df_filtered to df_filtered
        # col_name = df_filtered.columns.tolist()
        # # df_filtered = df_filtered.append(output_ana_df_filtered[col_name], ignore_index=True)
        # df_filtered = pd.concat([df_filtered, output_ana_df_filtered[col_name]], ignore_index=True)

        # Ensure the dataframe is sorted by date 
        df_filtered = df_filtered.sort_values('SortDate')
        # output_ana_df_filtered = output_ana_df_filtered.sort_values('SortDate')

        ##### Plot the new figure #####
        # Set the style to a modern-looking style
        sns.set_style("whitegrid")

        # Define the colors and linewidth
        true_color = "#3498db" # blue
        pred_color = "#e74c3c" # red
        true_color_l = "#7FB3D5" # lighter blue
        pred_color_l = "#F1948A" # lighter red
        line_width = 2.0
        fill_alpha = 0.1

        # Create the plot
        plt.figure(figsize=(14, 6))

        # Plot the ground truthing values (Seats)
        line1, = plt.plot(df_filtered['Date'], df_filtered['Seats'], label='Airlines Performance (True)', 
                linestyle='-', marker='o', color=true_color, linewidth=line_width)

        # Plot the prediction values and confidence intervals
        line2, = plt.plot(df_filtered['Date'], df_filtered['pred'], label='Prediction', 
                linestyle='--', marker='o', color=pred_color, linewidth=line_width)
        lower_bound = df_filtered['pred'] - df_filtered['std'] * 1.96
        upper_bound = df_filtered['pred'] + df_filtered['std'] * 1.96
        plt.fill_between(df_filtered['Date'], lower_bound, upper_bound, color=pred_color, alpha=fill_alpha)

        # Create a Patch for the confidence interval
        interval_patch = mpatches.Patch(color=pred_color, alpha=fill_alpha, label='95% Confidence Interval')

        # Collect all the plots for legend
        handles = [line1, line2, interval_patch]  

        # Add legend, title, and labels. Also adjust their font sizes.
        plt.legend(loc='upper left',  handles=handles, fontsize=12)
        plt.title(f'Seats Trend for {al} {alpha}', fontsize=16)
        plt.xlabel('Date (Quarter)', fontsize=14)
        plt.ylabel('Seats', fontsize=14)

        if not on_apply_data:
            plt.ylim(bottom=0)

        # Increase tick label size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot
        plt.tight_layout() # This ensures that all labels are visible in the saved file
        if self.tune_folder is None:
            fig_name = './results/' + f'{al}_{alpha}_ind.png'
        else:
            fig_name = f'./{self.tune_folder}/results/' + f'{al}_{alpha}_ind.png'
        # fig_name = './results/' + f'{al}_{alpha}_ind.png'
        plt.savefig(fig_name, dpi=300) # Set dpi for higher resolution

        # Display the plot
        plt.show()

    def plot_prediction_n(self, al, alpha):
        """
        Only needed one plot one quarter
        """
        # Capitalize the input
        al = al.upper()
        alpha = alpha.upper()

        # Make sure alpha is a valid route
        alpha_list = [alpha[0:3], alpha[3:6]]
        alpha = ''.join(sorted(alpha_list))

        # Prepare the data
        df = self.df_plot.copy()

        # Filter dataframe based on Mkt Al, Orig, and Dest
        df_filtered = df.loc[(df['Mkt Al'] == al) & (df['route'] == alpha)].copy()

        if len(df_filtered) == 0:
            print(f'Route Not Existed for {al} {alpha}...')
            return
        elif len(df_filtered) < 10:
            print(f'Data not Enough {al} {alpha}...')
            return
        
        # # Create a separate column for sorting
        df_filtered['SortDate'] = df_filtered['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))

 
        # Ensure the dataframe is sorted by date 
        df_filtered = df_filtered.sort_values('SortDate')

        ##### Plot the new figure #####
        # Set the style to a modern-looking style
        sns.set_style("whitegrid")

        # Define the colors and linewidth
        true_color = "#3498db" # blue
        pred_color = "#e74c3c" # red
        line_width = 2.0

        # Create the plot
        plt.figure(figsize=(14, 6))

        # Plot the ground truthing values (Seats)
        plt.plot(df_filtered['Date'], df_filtered['Seats'], label='Airlines Performance (True)', 
                linestyle='-', marker='o', color=true_color, linewidth=line_width)

        # Plot the prediction values and confidence intervals
        plt.errorbar(df_filtered['Date'], df_filtered['pred'], yerr=df_filtered['std']*1.96,
                     label='Prediction 2023', linestyle='--', marker='o', color=pred_color, linewidth=line_width, elinewidth=1.0, capsize=5)

        # Add legend, title, and labels. Also adjust their font sizes.
        plt.legend(loc='upper left', fontsize=12)
        plt.title(f'Seats Trend for {al} {alpha}', fontsize=16)
        plt.xlabel('Date (Quarter)', fontsize=14)
        plt.ylabel('Seats', fontsize=14)

        # Increase tick label size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot
        plt.tight_layout() # This ensures that all labels are visible in the saved file
        plt.savefig(f'{al}_{alpha}_ind.png', dpi=300) # Set dpi for higher resolution

        # Display the plot
        plt.show()


def main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name, tune_folder=None):
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
    params = {'batch_size': 8,
              'shuffle': False,  # set False to keep the order of the dataset
              'num_workers': 1, # number of workers for the dataloader
              'collate_fn': collate_fn}
    

    # create a result folder if not exist
    results_path = './results' if tune_folder is None else f'./{tune_folder}/results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args is None:
        print("Using default parameters since no arguments are provided.")
        # resume_training=False
        MSE_or_GaussianNLLLoss="MSE"
        rnn_type="LSTM"
        pred_num_quarters=4
        seq_num=10
        if_add_time_info=False
        n_layers=4
        drop_prob=0.00001
        num_heads=6
        bidirectional = True
        if_skip = True
        if_feed_drop = True
        if_feed_norm = True
        hidden_dim = 300

        start_quarter = "Q1 2023" # or "Q4 2022"
        skip_quarters = 2

        validation_type = 'Val'
        tune = False
        use_bn = True
        activation_num = 0
    else:
        print("Using the provided arguments.")
        # Control if resume training
        MSE_or_GaussianNLLLoss = args.MSE_or_GaussianNLLLoss
        pred_num_quarters = args.pred_num_quarters # number of quarters to predict
        seq_num = args.seq_num # number of quarters in a sequence
        if_add_time_info = args.if_add_time_info # if add time information
        # # Set NN parameters
        rnn_type = args.rnn_type
        n_layers = args.n_layers
        # drop_prob = args.drop_prob
        drop_prob = 0.00001
        num_heads = args.num_heads

        bidirectional = args.bidirectional
        if_skip = args.if_skip
        if_feed_drop = args.if_feed_drop
        if_feed_norm = args.if_feed_norm
        hidden_dim = args.hidden_dim
        
        start_quarter = getattr(args, 'start_quarter', "Q1 2023")
        skip_quarters = getattr(args, 'skip_quarters', 2)
        # start_year = args.start_year

        validation_type = getattr(args, 'validation_type', 'Val')

        validation_type = getattr(args, 'validation_type', 'Val')
        tune = getattr(args, 'tune', False)

        use_bn = getattr(args, 'use_bn', True)
        activation_num = getattr(args, 'activation_num', 0)

    print("-------- Start ----------")
    # record the start time
    start_time = time.time()

    # Set the random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Will use {device} as device")

    # load the data
    # data_format = Validation(folder_path, seats_file_name, perf_file_name, x_features, if_add_time_info=if_add_time_info, sequence_length=10, pred_num_quarters=pred_num_quarters, ds_type=validation_type)
    # apply_file_name = '\Schedule_Monthly_Summary_2023Q1234.csv'
    data_format = Validation(folder_path, seats_file_name, perf_file_name, 
                             x_features, apply_file_name=apply_file_name, 
                             if_add_time_info=if_add_time_info, 
                             sequence_length=seq_num, 
                             pred_num_quarters=pred_num_quarters,
                             skip_quarters=skip_quarters)

    # Check if 'applying_data.csv' exists
    applying_data_path = data_format.data_root + 'applying_data.csv'
    if os.path.exists(applying_data_path):
        print("Applying data exists, will load it")
        load_apply_data = True
    else:
        print("Applying data does not exist, will create it")
        load_apply_data = False
    boundary_quarter, test_boundary_quarter, apply_data_boundary = calculate_quarters(pred_num_quarters, seq_num, 
                                                                                      start_quarter=start_quarter,
                                                                                      skip_quarters=skip_quarters)
    
    train_filename = data_format.data_root + 'training_data.csv' 
    scaled_filename = data_format.data_root + 'scaled_data.csv' 
    apply_filename = data_format.data_root + 'applying_data.csv'
    
    data_format.load_scaled_data_val(train_filename= train_filename,
                                     scaled_filename=scaled_filename,
                                     apply_filename=apply_filename,
                                     load_apply_data=load_apply_data, 
                                     test_date=apply_data_boundary,
                                     on_apply_data=(validation_type=='Val'),
                                     start_quarter=start_quarter)
    # test data is test data starting date. 
    full_dataset = data_format.full_df

    # Create the dataloader
    dataloader = DataLoader(full_dataset, **params)
    input_dim = len(x_features) + int(if_add_time_info) + 4 + skip_quarters
    net = RNNNet(cat_feat=data_format.cat_features, cat_mapping=data_format.cat_mapping,
             embed_dim_mapping=data_format.embed_dim_mapping,
             input_dim=input_dim, hidden_dim=hidden_dim, output_dim=pred_num_quarters,
             n_layers=n_layers, drop_prob=drop_prob, rnn_type=rnn_type,
             bidirectional=bidirectional, num_heads=num_heads, 
             if_skip=if_skip, if_feed_drop=if_feed_drop, if_feed_norm=if_feed_norm,
             MSE=(MSE_or_GaussianNLLLoss=='MSE'), use_bn=use_bn, 
             activation_num=activation_num)

    # load the model
    # model_path = r'./model/model.pth'
    model_path = r'./model/model.pth' if tune_folder is None else f'./{tune_folder}/model/model.pth'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print("Model loaded")

    # validate the model
    outputs, seats, stds, keys = validation(dataloader, net, data_format.seat_scaler, 
                                            device, MSE=(MSE_or_GaussianNLLLoss=='MSE'),
                                            tune_folder=tune_folder,
                                            on_apply_data=(validation_type=='Val'))

    route_dict = create_route_dict(outputs, seats, stds, keys)
    orig_df = data_format.test_df
    route_df = route_dict_to_df(route_dict)

    # save the processed data
    # ana_df_name = "./results/data_to_ana_apply.csv"
    if tune_folder is None:
        ana_df_name = "./results/data_to_ana_apply.csv"
    else:
        ana_df_name = f"./{tune_folder}/results/data_to_ana_apply.csv"
    route_df.to_csv(ana_df_name)

    # Create the DataAna object
    ana = DataAna(ana_df_name, tune_folder=tune_folder)
    ana.merge_previous_data(orig_df)

    find_best_routes(tune_folder=tune_folder) # find the best routes
    
    if tune == False:
        # Ask the user to input the airline and route
        while True:
            user_input = input("Enter airline and route, separated by comma, or 'c' to exit: ")
            if user_input.lower() == 'c':
                break
            try:
                airline, route = user_input.split(',')
                airline = airline.strip()  # remove possible leading/trailing whitespaces
                route = route.strip()  # remove possible leading/trailing whitespaces
                ana.plot_prediction(airline, route, on_apply_data=(validation_type=='Val'))
            except ValueError:
                print("Invalid input, please enter the airline and route separated by a comma or 'continue' to proceed.")

    # record the end time
    end_time = time.time()
    time_used = (end_time - start_time) / 60
    print(f"Time used: {time_used} minutes")
    print("-------- End ----------")


if __name__ == "__main__":
    # Set basic parameters
    folder_path = r'C:\Users\qilei.zhang\OneDrive - Frontier Airlines\Documents\Data\USconti'
    seats_file_name = r'\Schedule_Monthly_Summary_Report_Conti.csv'
    perf_file_name = r'\Airline_Performance_Report_Conti.csv'
    apply_file_name = '\Schedule_Monthly_Summary_2023Q1234.csv'

    # Load parameters from the JSON file.
    if not os.path.exists('parameters.json'):
        print("parameters.json does not exist, Find the file and put it in the same folder as this file")
    with open('parameters.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))

    main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name)
