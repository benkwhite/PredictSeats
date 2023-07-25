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
from RNN_ana import DataAna


# Inherit the DatasetFormation class
class Validation(DatasetFormation):
    def __init__(self, folder_path, seats_file_name, perf_file_name, x_features, apply_file_name, *args, **kwargs):
        super().__init__(folder_path, seats_file_name, perf_file_name, x_features, *args, **kwargs)

        self.le_airports = joblib.load('le_airports.pkl')
        self.le_airlines = joblib.load('le_airlines.pkl')
        
        # Load the mappings
        with open("cat_mapping.pkl", "rb") as f:
            self.cat_mapping = pickle.load(f)
        with open("embed_dim_mapping.pkl", "rb") as f:
            self.embed_dim_mapping = pickle.load(f)
        print("Dimension mapping loaded.")

        # apply_file_name = 'Schedule_Monthly_Summary_2023Q12.csv'
        self.apply_sch_df = pd.read_csv(self.root + apply_file_name)

    def create_real_test_data(self, test_date='Q1 2021'):
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
        test_date_num = int(test_date.split(' ')[1]) * 4 + int(test_date.split(' ')[0][1])
        self.df['SortDate'] = self.df['Date'].apply(lambda x: int(x.split(' ')[1]) * 4 + int(x.split(' ')[0][1]))
        self.test_df = self.df[self.df['SortDate'] >= test_date_num]
        
        # reset the index
        self.test_df.reset_index(drop=True, inplace=True)

        # save the test data
        self.test_df.to_csv('applying_data.csv', index=False)

        print("The test data is created.")

    def load_scaled_data_val(self, train_filename='training_data.csv', 
                             test_filename='testing_data.csv', 
                             scaled_filename='scaled_data.csv', 
                             apply_filename='applying_data.csv',
                             load_apply_data=True,
                             test_date='Q1 2021'):
        # load the original training data first to get the scaler
        self.train_df = pd.read_csv(train_filename)
        if not load_apply_data:
            # self.test_df = pd.read_csv(test_filename) # Not needed
            self.create_real_test_data(test_date)
        else:
            self.test_df = pd.read_csv(apply_filename)

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
        print("Validation data scaled.")
        self.create_date_features_val()
        print("Date features created.")
        # self.save_scaled_data_val()  # temporary comment
        
        # prepare the validation data
        self.final_preparation_val()
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

        print("Apply data scaled.")

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

    def final_preparation_val(self):
        # Create binary features for the quarter
        # self.scaled_df = pd.get_dummies(self.scaled_df, columns=['year', 'quarter'])
        # group the dataframe for each airline and each route
        self.scaled_df = pd.get_dummies(self.scaled_df, columns=['quarter'])
        datasets = []
        for key, route_df in self.scaled_df.groupby(["Mkt Al", "Orig", "Dest"]):
            # Get the value of the seats from applying data
            seats_df = self.apply_sch_df[(self.apply_sch_df['Mkt Al'] == key[0]) & (self.apply_sch_df['Orig'] == key[1]) & (self.apply_sch_df['Dest'] == key[2])]
            if len(seats_df) > self.n_future:
                seats_df = seats_df.iloc[:self.n_future, :]

            if len(route_df) < self.seq_len:
                continue
            
            # Get the seats values to list
            seats_list = seats_df['Seats'].values

            # fill out 0 for the seats_list if the seats_list is not long enough
            if len(seats_df) < self.n_future:
                seats_list = list(seats_list) + [0] * (self.n_future - len(seats_df))

            route_df = route_df.sort_values("Date_delta")
            datasets.append(FlightDataset(route_df, self.seq_len, self.num_features,
                                          self.cat_features, self.embed_dim_mapping,
                                          time_add=self.time_add, seats_values=seats_list,
                                          n_future=self.n_future))
        self.full_df = torch.utils.data.ConcatDataset(datasets)

    def save_scaled_data_val(self, filename='scaled_data_apply.csv'):
        self.scaled_df.to_csv(filename, index=False)
        print("Scaled applying data saved.")


class FlightDataset(Dataset):
    """
    Create a dataset that can be used for dataloading.
    """
    def __init__(self, df, sequence_length, num_feat, cat_feat, embed_dim_mapping, time_add=True, seats_values=None, n_future=2):
        self.df = df
        self.sequence_length = sequence_length
        self.num_features = num_feat
        self.cat_features = cat_feat

        self.dummy_quarter = [*[f"quarter_{i}" for i in range(1, 5)]]

        if time_add:
            self.num_features = self.num_features + self.dummy_quarter + ['time_scaled']
        else:
            self.num_features = self.num_features + self.dummy_quarter

        if seats_values is None:
            self.seats_values = [0] * n_future
        else:
            self.seats_values = seats_values

    def __len__(self):
        return len(self.df) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Get the relevant slice of the dataframe
        df_slice = self.df.iloc[idx : idx + self.sequence_length]
        
        # Construct the sequence data
        sequence_data = df_slice[self.num_features].astype(float).values
        cat_sequence_data = torch.LongTensor(df_slice[self.cat_features].values)

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
        target_data = self.seats_values

        # Return the sequence data and the target value. "Seats" is the 19th column (0-indexed)
        # return torch.from_numpy(sequence_data[:-1]), torch.tensor(sequence_data[-1, 18])  
        return torch.from_numpy(sequence_data), cat_sequence_data, torch.tensor(target_data), time_range, loc_key
    

def validation(loader, net, seat_scaler, device='cpu', n_times=10, MSE=True):
    # Set the network to evaluation mode
    net = net.to(device) 

    if MSE:
        net.train()
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
    error_table_filename = 'error_table_apply.csv'
    final_error_table.to_csv(error_table_filename, index=False)

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


def route_dict_to_df(route_dict):
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


# inherit the DataAna Class
class DataAna(DataAna):
    def __init__(self, ana_df_name):
        super().__init__(ana_df_name)
        self.boundary_quarter = "Q1 2023"
        self.boundary_num = int(self.boundary_quarter.split(' ')[1]) * 4 + int(self.boundary_quarter.split(' ')[0][1])

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
        # self.remove_other_parameters()

        # Drop `Flights` and  `num_comp` column
        # self.df.drop(columns=['Flights', 'num_comp'], inplace=True)

        # Sort the dataframe by route and date
        self.df.sort_values(by=['Mkt Al', 'route', 'Date'], inplace=True)

        self.df_plot = self.df.copy()

        # Calculate the metrics 
        self.calculate_competitors()
        self.classify_competitors()
        self.classify_market_size()
        self.calculate_metrics_new()

        self.df.to_csv("data_after_ana_app.csv", index=False)

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
        result_df.to_csv("result_apply.csv", index=False)

        return result_df
    
    def plot_prediction(self, al, alpha):
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
        line2, = plt.plot(df_filtered['Date'], df_filtered['pred'], label='Prediction 2023', 
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

        # Increase tick label size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot
        plt.tight_layout() # This ensures that all labels are visible in the saved file
        plt.savefig(f'{al}_{alpha}_ind.png', dpi=300) # Set dpi for higher resolution

        # Display the plot
        plt.show()

    def plot_prediction_n(self, al, alpha):
        """
        Use it after run calculate_metrics_new
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


def main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name):
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
    validation_type = 'Val'

    if args is None:
        print("Using default parameters since no arguments are provided.")
        # resume_training=False
        MSE_or_GaussianNLLLoss="MSE"
        rnn_type="LSTM"
        pred_num_quarters=4
        seq_num=10
        if_add_time_info=False
        n_layers=4
        drop_prob=0.2
        num_heads=6

        # start_year = 2016
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
        drop_prob = 0.3
        num_heads = args.num_heads

        # start_year = args.start_year

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
                             pred_num_quarters=pred_num_quarters)

    # Check if 'applying_data.csv' exists
    applying_data_path = 'applying_data.csv'
    if os.path.exists(applying_data_path):
        print("Applying data exists, will load it")
        load_apply_data = True
    else:
        print("Applying data does not exist, will create it")
        load_apply_data = False
    boundary_quarter, test_boundary_quarter, apply_data_boundary = calculate_quarters(pred_num_quarters, seq_num)
    data_format.load_scaled_data_val(load_apply_data=load_apply_data, test_date=apply_data_boundary)
    # test data is test data starting date. 
    full_dataset = data_format.full_df

    # Create the dataloader
    dataloader = DataLoader(full_dataset, **params)
    input_dim = len(x_features) + int(if_add_time_info) + 4
    net = RNNNet(cat_feat=data_format.cat_features, cat_mapping=data_format.cat_mapping,
             embed_dim_mapping=data_format.embed_dim_mapping,
             input_dim=input_dim, hidden_dim=300, output_dim=pred_num_quarters,
             n_layers=n_layers, drop_prob=drop_prob, rnn_type=rnn_type,
             bidirectional=True, num_heads=num_heads, 
             if_skip=True, if_feed_drop=True, if_feed_norm=True)

    # load the model
    # model_path = r'C:\Users\qilei.zhang\OneDrive - Frontier Airlines\Documents\VSCode\SeatPredict\model.pth'
    model_path = r'model.pth'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print("Model loaded")

    # validate the model
    outputs, seats, stds, keys = validation(dataloader, net, data_format.seat_scaler, 
                                            device, MSE=(MSE_or_GaussianNLLLoss=='MSE'))

    route_dict = create_route_dict(outputs, seats, stds, keys)
    orig_df = data_format.test_df
    route_df = route_dict_to_df(route_dict)
    ana_df_name = "data_to_ana_apply.csv"
    route_df.to_csv(ana_df_name)

    # Create the DataAna object
    ana = DataAna(ana_df_name)
    ana.merge_previous_data(orig_df)

    ana.plot_prediction('AA', 'LAXSFO')
    ana.plot_prediction('DL', 'ATLSFO')
    ana.plot_prediction('F9', 'DENLAS')
    # ana.plot_prediction('AA', 'ATLDFW')
    ana.plot_prediction('AA', 'DENDFW')
    ana.plot_prediction('AA', 'ATLPHL')


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
