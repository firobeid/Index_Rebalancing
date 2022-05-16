#!/usr/bin/env python
def check_ranking(df):
    '''
    Check if the dataframe is ranked in descending order
    returns: True if the columns ranked in descending otherwise False
    '''
    if (df.shift(-1) > df).nunique() == 1:
        return True
    return False
def ranking_enforcer(df):
    #Check if the  dataframe is actually ordered in descending order
    if bool(df.apply(check_ranking).Z_Value) == True:
        pass
    else:
        _ = df.sort_values(by = "Z_Value", ascending=False, inplace = True)

def validate_data(index,selection,universe_date):
    '''
    Validate data at any point to make sure it matches original df data at 
    any index level
    '''
    try:
        return selection.iloc[index][0] == universe_date.iloc[index][0]
    except:
        print("Out of range index")
