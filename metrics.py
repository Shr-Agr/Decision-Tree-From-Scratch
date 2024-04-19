from typing import Union
import pandas as pd

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    """
    assert (y_hat.size == y.size)
    assert (y_hat.size > 0)
    # return (y_hat==y).sum()/y.size
    return (y_hat.reset_index(drop=True)==y.reset_index(drop=True)).sum()/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    bool_indexes = y_hat==cls
    bool_ind2 = y==cls
    good_vals = (y_hat.reset_index(drop=True)[bool_indexes.reset_index(drop=True)] == y.reset_index(drop=True)[bool_indexes.reset_index(drop=True)]).sum()
    if bool_indexes.sum()>0:
        return good_vals/bool_indexes.sum()
    else:
        if bool_ind2.sum() > 0:
            return 0
        else:
            return None             # since the ground truth is never equal to class, so precision would be None and not zero
    

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    bool_indexes = y==cls
    good_vals = (y_hat.reset_index(drop=True)[bool_indexes.reset_index(drop=True)] == y.reset_index(drop=True)[bool_indexes.reset_index(drop=True)]).sum()
    if bool_indexes.sum()>0:
        return good_vals/bool_indexes.sum()
    else:
        return None                 # since the ground truth is never equal to class, so recall would be None and not zero


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return (((y.reset_index(drop=True)-y_hat.reset_index(drop=True))**2).mean())**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return abs(y.reset_index(drop=True)-y_hat.reset_index(drop=True)).mean()