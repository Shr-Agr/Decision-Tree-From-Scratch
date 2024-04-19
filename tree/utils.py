import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if(str(y.dtype)=="float64" or str(y.dtype)=="float32"):
        return True
    elif(str(y.dtype)=="category"):
        return False
    return True


def entropy_formula(p):
    if(p==0):
        return 0
    return -p * np.log2(p)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    Y_weighted=Y.value_counts(normalize=True)
    
    return Y_weighted.apply(entropy_formula).sum()    


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the Mean Squared Error
    """
    avg = Y.sum()/len(Y)
    return ((Y-avg)**2).sum()/len(Y)


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    Y_weighted=Y.value_counts(normalize=True)
    return 1-((Y_weighted**2).sum())


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    # attr is the series of values of one attribute
    
    ##################################################################
    
    # For Discrete Input Discrete Output
    # Parameter -> Entropy
    if str(Y.dtype)=="category" and str(attr.dtype)=="category":
        Y_weighted=Y.value_counts(normalize=True)
        total_entropy=Y_weighted.apply(entropy_formula).sum()
        attr_entropy=0
        attr_vals=attr.unique()
        for attr_val in attr_vals:
            Y_at_attr_val=Y[attr==attr_val]
            subset_entropy=Y_at_attr_val.value_counts(normalize=True).apply(entropy_formula).sum()
            attr_entropy+=((attr==attr_val).sum()/len(Y))*subset_entropy
        info_gain = total_entropy-attr_entropy
    
    ##################################################################
        
    # For Discrete Input Real Output
    # Parameter -> Mean Squared Error
    elif str(Y.dtype)=="float64" and str(attr.dtype)=="category":
        sample_mse=mse(Y)
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        weighted_mse=0
        for value in attr.unique():
            weighted_mse = weighted_mse + mse(df_samples[attr==value]['Y'])*(len(df_samples[attr==value]['Y'])/len(Y))
        info_gain = sample_mse - weighted_mse

    ##################################################################

    # For Real Input Discrete Output
    # Parameter -> Entropy with splitting the Input
        # sort by the attribute and find the split point where maximum info gain occurs
    elif str(Y.dtype)=="category" and str(attr.dtype)=="float64":
        split_point = -np.inf
        max_gain = -np.inf
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        df_sorted = df_samples.sort_values('Attribute')
        sorted_attributes = attr.sort_values()
        lower_ind = sorted_attributes.index[0]

        for upper_ind in sorted_attributes.index[1:]:
            middle_val = (sorted_attributes[lower_ind] + sorted_attributes[upper_ind])/2
            split_gain = entropy(Y) - entropy(df_sorted.loc[attr<=middle_val]['Y']) * (len(df_sorted.loc[attr<=middle_val]['Y'])/len(Y)) - entropy(df_sorted.loc[attr>middle_val]['Y']) * (len(df_sorted.loc[attr>middle_val]['Y'])/len(Y))
            lower_ind=upper_ind
            if split_gain>max_gain:
                split_point=middle_val
                max_gain=split_gain
        info_gain = (max_gain,split_point)

    ##################################################################
    
    # For Real Input Real Output
    # Parameter -> Mean Squared Error with splitting the Input
    else:
        split_point = -np.inf
        max_gain = -np.inf
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        df_sorted = df_samples.sort_values('Attribute')
        sorted_attributes = attr.sort_values()
        lower_ind = sorted_attributes.index[0]

        for upper_ind in sorted_attributes.index[1:]:
            middle_val = (sorted_attributes[lower_ind] + sorted_attributes[upper_ind])/2
            split_gain = mse(Y) - mse(df_sorted.loc[attr<=middle_val]['Y']) * (len(df_sorted.loc[attr<=middle_val]['Y'])/len(Y)) - mse(df_sorted.loc[attr>middle_val]['Y']) * (len(df_sorted.loc[attr>middle_val]['Y'])/len(Y))
            lower_ind=upper_ind
            if split_gain>max_gain:
                split_point=middle_val
                max_gain=split_gain
        info_gain = (max_gain,split_point)

    return info_gain