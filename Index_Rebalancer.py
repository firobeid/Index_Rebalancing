#!/usr/bin/env python
from mimetypes import init
import os
import sys
import unittest
import warnings
from datetime import datetime
import numpy as np
import pandas as pd  # pip install openpyxl
from openpyxl import load_workbook
from scipy.optimize import least_squares, minimize
from functions import *
warnings.filterwarnings("ignore")

path = "Tasks for Python Test (003).xlsx" #can be set from command line, can be changed to sys.args later on for broader efficiency

if not os.path.isfile(path):
    sys.exit("Data file does not exist in the path provided or wrong file name")

def get_data(path):
    global universe
    universe = pd.read_excel(path, 
                  sheet_name="Start Universe",
                  parse_dates = True, 
                 index_col=[0,2])#load in our universe, parse the dates, create multi index df(date,asset ID]
    universe = universe[~universe.index.duplicated()] #safe pratice to drop rows based on unique (date,ticker) combo
    return universe
def intializer(assets = 50):
    global universe_date, choose_rebalance_date, target
    global date #need this to set 20% buffer or turn it off for 2017-06-30 date
    target = int(assets)
    available_dates = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in universe.index.get_level_values(0).unique().values]
    keys = list(map(str,list(range(0,len(available_dates)))))
    available_dates = dict(zip(keys,available_dates))
    for k,v in available_dates.items():
        print(k,"-->",v)
    choose_rebalance_date = input("Please choose from the above available number options to rebabance the index for the\
    respective date:")
    if int(choose_rebalance_date) not in [0,1,2]:
        sys.exit("Please choose only from the provided options, %s is invalid"%choose_rebalance_date)
    date = available_dates[str(choose_rebalance_date)]
        
    #get the only column that we need to run search algo on to save memory in the search process
    universe_date = universe.loc[date:date,["Z_Value"]]\
    .sort_values(by = "Z_Value", ascending=False).copy()
    return universe_date

def intial_40_selection():
    global selection
    selection = pd.DataFrame()
    ranking_enforcer(universe_date) 
    # universe_date = universe_date.iloc[:39] STRESS TEST FIRST CONDITION TO CHECK universe_date<=40 
    if universe_date.shape[0] > 40:
        selection = universe_date.iloc[:40]
    else: 
        selection = universe_date.iloc[:] #select all at this point if equal or less then 40
    print("***%d assets have been selected***"% selection.shape[0])
    return selection

def selection_10_buffer():
    global selection
    start_assets = selection.shape[0]
    ranking_enforcer(universe_date)
    if universe_date.shape[0] > 60:
        temp_10_buffer = universe_date.iloc[40:60].copy() # slice from index 40 to 59 (total 60 assets ranked)
        selection = selection.append(temp_10_buffer.iloc[:10])
    elif universe_date.shape[0] > 40: 
        temp_10_buffer = universe_date.iloc[40:].copy() #if less or equal then 60 assets in the universe, select all from 40 onwards
        if temp_10_buffer.shape[0] >= 10:
            selection = selection.append(temp_10_buffer.iloc[:10]) #given the original universe has less the 60 assets
        else:
            selection = selection.append(temp_10_buffer) # at this point we still need assets to reach 50 in our universe
    else:
        print("No assets have been added due to limitation of our universe")
        pass #this captures if we actually dont have more then 40 assets in the universe
    if selection.shape[0] > 50: #precautionary layer to make sure only 50 selected up to now
        selection = selection.iloc[0:50] 
    new_assets = selection.shape[0]
    print("***%d new assets have been added***"% (new_assets -start_assets))
    print("***%d assets have been total detected***"% (selection.shape[0]))
    return selection
    
def new_candidates_topUp50():
    global selection
    
    start_assets = selection.shape[0]
    get_data(path) #path can be changed if new excel file different from old one
    print("***Please choose same date as before for same market value rebalance!***")
    intializer() 
    ranking_enforcer(universe_date)
    least_ZValue = selection.iloc[-1][0] #get the last Z_Value from last step
    temp_remaining = universe_date[universe_date.Z_Value < least_ZValue].copy()#assuming new scores are lower then 
    # the one we have NOT higher since the intial run should have had the highest z_values
    ranking_enforcer(temp_remaining)
    temp_remaining = temp_remaining.loc[temp_remaining.index.difference(selection.index),:] #make sure no duplicates from new/old universe
    ranking_enforcer(temp_remaining)
    assets_remaining= target - len(selection) #target is set at runtime
    selection = selection.append(temp_remaining.iloc[:assets_remaining])
    new_assets = selection.shape[0]
    print("***%d new assets have been added***"% (new_assets -start_assets))
    print("***%d assets have been total found***"% (selection.shape[0]))
    return selection

def intial_weights():
    '''
    Adjust intial guess weights to fit constraints
    '''
    global x0
    #get indicies where sectors assets belong too same sector
    all_weights = []
    x0 = np.array(selection["Max Wt"].values).round(4) #what we have to start with
    sectors = set(selection["Sector Code"])
    for i in sectors:
        if isinstance(pd.Index(selection["Sector Code"]).get_loc(i), int):
            indexes = [pd.Index(selection["Sector Code"]).get_loc(i)]
        else:
            indexes = np.where(pd.Index(selection["Sector Code"]).get_loc(i))[0].tolist()
        ub = 0.5
    #     print(i);print(indexes);print(sum(results[[indexes]]))
        all_weights.append({"Sector": i, "Assets Index": indexes, "Sector Weights Sum":sum(x0[[indexes]])})
    for i in all_weights:
        indexes = i["Assets Index"]
        # print(indexes)
        if sum(x0[indexes]).round(4) > 0.5:
            print("Adjusting sector weights: %s" % i["Sector"])
            decrease_amount = (sum(x0[indexes]).round(4) - 0.5).round(4)
            x0[indexes] = x0[indexes] - decrease_amount/len(x0[indexes])
    # make sure no weights are negative and all sum is = to 1
    x0[x0 < 0] = 0.
    x0 = (((x0 -((sum(x0)-1)/len(x0))).round(4))) 
    x0[x0 < 0] = 0. #needed to be repeated to make sure the min is non-negative
    print("Sum of intial weights is %f" % sum(x0).round(4))
    
def minimization_init():
    global Uncapped_Wt, stock_floor, stock_cap, bounds
    try:
        Uncapped_Wt = selection['Uncapped Wt'].values
        # x0 = list(selection["Max Wt"].values)
        # x0 = Uncapped_Wt
        # x0 = len(selection) * [1. / len(selection),] #Intial guess (1/50 each asset)
        stock_floor = 0.05/100 #0.05% floor
        stock_cap = selection["Max Wt"].values
        bounds = tuple((stock_floor, x) for x in stock_cap) 
    except:
        print("Something went wrong, please check if dataframe `selection` is available as a global\
        variable and columns: `Max Wt` & `Uncapped Wt` exist")
        sys.exit()
    return "Minimization variables init Done..."
        
def min_function(Capped_Wt):
    '''
    To minimize the sum of [(squared difference between Capped Wt and Uncapped Wt)/Uncapped Wt for each stock]
    '''
    # OBJECTIVE FUNCTION
    return sum(((Capped_Wt - Uncapped_Wt)**2)/ Uncapped_Wt) # to use with pandas, indices should have the same name

def fun_Constraints(groupCodes):
    #The only thing I had to outsource:
    #https://stackoverflow.com/questions/62109983/setting-up-group-constraints-for-scipy-minimize-optimization-problem
    #But some minor improvments
    uniqueGroupCodes = set(groupCodes)
    #All Weights should sum up to 1
    group_cons = [{'type': 'eq', 'fun': lambda x:  1. - sum(x)}]
    #Implementing to each sector assets constraint: 50% - SUM(weights of asset in same sector) >= 0
    for code in uniqueGroupCodes:
        if isinstance(pd.Index(groupCodes).get_loc(code), int):
            indexes = [pd.Index(groupCodes).get_loc(code)]
        else:
            indexes = np.where(pd.Index(groupCodes).get_loc(code))[0].tolist()
        ub = 0.5
        ubdict = {'type': 'eq', 'fun': lambda x: np.array(0.5 - (np.sum(x[[indexes]])))}
        # lbdict = {'type': 'ineq', 'fun': lambda x: np.array(0 + (np.sum(x[[indexes]])))}
        _ = group_cons.append(ubdict); #group_cons.append(lbdict)

    return group_cons
def opt_constraint_checker(weights):
    '''
    Optimization post constraint checker; If all constraints where met
    '''
    global all_weights
    all_weights = []
    sectors = set(selection["Sector Code"])
    for i in sectors:
        if isinstance(pd.Index(selection["Sector Code"]).get_loc(i), int):
            indexes = [pd.Index(selection["Sector Code"]).get_loc(i)]
        else:
            indexes = np.where(pd.Index(selection["Sector Code"]).get_loc(i))[0].tolist()
        ub = 0.5
    #     print(i);print(indexes);print(sum(results[[indexes]]))
        all_weights.append({"Sector": i, "Assets Index": indexes, "Sector Weights Sum":sum(weights[[indexes]])})
    sector_weights_sum = [i['Sector Weights Sum'].round(4) for i in all_weights]
    total_sum = sum(sector_weights_sum)
    # Check if pass total assets weights = 100%
    if abs(round((total_sum - 1), 1)) == 0.0: #tolerance for [0.90 - 1.04] sum of weights
        print("Passed constraint: Total weights ~ 100%")
    else:
        print("Failed constraint: Total weights ~ 100%")
    # Check if pass sector cap weights sum 
    if max(sector_weights_sum).round(1) <= 0.5:
        print("Passed constraint: Sector Cap = 50%")
    else:
        print("Failed constraint: Sector Cap = 50%")
    #Check if pass asset individual cap limits
    if all(stock_cap.round(4) >= results):
        print("Passed constraint: Stock Cap = Min (5%, 20*FCap Wt)")
    else:
        print("Failed constraint: Stock Cap = Min (5%, 20*FCap Wt)")
    #Check if pass asset individual floor limit = 0.05%
    if all(stock_floor <= results):
        print("Passed constraint: Stock Floor = 0.05%")
    else:
        print("Failed constraint: Stock Floor = 0.05%")

###Execution####
def run():
    try:
        global selection
        intializer()
        intial_40_selection()
        # print(validate_data(selection.shape[0]-1))
        if date != "2017-06-30":#since No rebalance buffer is needed for the first rebalance at 6/30/2017 in the instructions stated
            print(f'Applying 20% buffer rule for date: {date}....')
            selection_10_buffer()
        else:
            print(f"Skipping 20% buffer rule for date: {date} .....")
        if selection.shape[0] ==target:
            pass
        else:
            new_candidates_topUp50() #any remaing assets not chosen in step 2 and 3 go to this function
        if isinstance(selection,pd.DataFrame):
            print("Returning asset selection...")
            if len(selection) > 1:
                try:
                # get back all other columns from the original universe based on common index combo
                    selection = universe.loc[selection.index.intersection(universe.index), :]
                    #return final dataframe
                    selection = selection.reset_index()[["Ref Date", "Company Name","RIC","Sector Code","FCap Wt","Z_Value"]]
                except:
                    sys.exit("Encountered error is returing dataframe")
            else:
                sys.exit("No data found....")
        print("Done....confirming that the assets for the index are %d" % (selection.shape[0]))
    except KeyError:
        print("Something went wrong, please re-run and check that you followed the on screen instructions...")
        pass
def prepare_parameters():
    '''
    Calculates needed column needed for the optimization process
    '''
    global selection
    print("Creating new columns!")
    try:
        selection["FCap Wt*(1+Z_Value)"] = selection.apply(lambda x: x['FCap Wt']*(1 + x["Z_Value"]),axis=1) 
        sum_of_FCap_Zvalue = sum(selection["FCap Wt*(1+Z_Value)"]) #denominator needed for the Uncapped Wt calculation
        selection["Uncapped Wt"] = selection.apply(lambda x: x["FCap Wt*(1+Z_Value)"]/sum_of_FCap_Zvalue,axis=1)
        #Stock Cap = Min (5%, 20*FCap Wt):
        selection["Max Wt"] = selection.apply(lambda x: min([0.05, 20*x["FCap Wt"]]), axis=1)
        selection["Ref Date"] = selection["Ref Date"].apply(lambda x:str(x).split()[0]) #convert back to strings with no minutes/hourly/seconds data
    except:
        print("Something went wrong, please check if dataframe `selection` is available as a global\
        variable and columns: `'FCap Wt` & `Z_Value` exist")
        sys.exit()
    print("Done...")
    return selection


def minimization_run(rounding = 4):
    np.random.seed(42)
    global results, weights
    prepare_parameters()
    try:
        minimization_init()
        intial_weights() #get initial weights
        linear_constraint = fun_Constraints(selection["Sector Code"].values)
        weights = minimize(min_function , x0 = x0,method='SLSQP', 
                       bounds = bounds,
                       options={'ftol': 1e-9, 'disp': True,'maxiter': 1000},constraints = (linear_constraint))
        print(weights.message," with objective function =  %.4f"%(weights.fun * 100), "%")
        results = weights.x.round(rounding)
        opt_constraint_checker(weights=results)
    except:
        sys.exit("Ran into an error in the optimization process")

def deliver_output():
    global final_results #if results need to be seen globally in df output
    final_results = pd.concat([selection[selection.columns[:-3]],pd.DataFrame(results, columns=["Weights"])], axis = 1)
    #get existing output sheet tab in the given excel - remember th `path` variable is the excel file pointer
    existing_output = pd.read_excel(path, 
                      sheet_name="Output Sheet",
                      parse_dates = True)
    #check unique dates in the output sheet, if avialble, to not add same rebalance weights if they where previously added
    existing_output_weights = [i for i in existing_output["Ref Date"].unique()] #str(i).split("T")[0]:note to self
    if date in existing_output_weights: #check if we already did this before for the chosen rebalancing date
        sys.exit("Rebalanced Weights already saved for the specified date %s" % date)
    else:
    #     with pd.ExcelWriter(path=path, engine= "openpyxl", mode= "a", if_sheet_exists="overlay") as writer:  
    #         selection.to_excel(writer, sheet_name='Output Sheet',  
    #                            header=False,index = False, 
    #                            startrow=len(existing_output) +1)
        #solution to avoid overwriting the contents inside the excel file while uploading the results to the output sheet
        book = load_workbook(path)
        writer =  pd.ExcelWriter(path=path, engine= "openpyxl")
        writer.book = book
        writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
        final_results.to_excel(writer, sheet_name='Output Sheet',  
                           header=False,index = False, 
                           startrow=len(existing_output) +1)
        writer.save()
        print("Rebalanced Weights have been uploaded for the market value date: %s" % date)



if __name__ == "__main__":
    get_data(path)
    run()
    prepare_parameters()
    minimization_run()
    deliver_output()
    # unittest.main()
