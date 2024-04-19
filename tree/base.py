"""
This code can make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import information_gain,gini_index,entropy,mse

np.random.seed(42)


# Tree Node class:
class TreeNode:
    def __init__(self, splitting_attr = None, val = None, depth = None):
        self.splitting_attr = splitting_attr  
        self.depth = depth              
        self.val = val

        # dictionary storing child tree nodes of a particular node
        self.ChildTreeNodes = {}
        # Discrete Attribute Case: # key of dict:   attribute value          
        #                          # value of dict: child node correspoonding to that particular attribute value 
        # Real Attribute Case:     # key of dict:   a tuple with first value as the splitting point and
        #                                           second value as a string either "lower" or "higher" depending on the case
        #                          # value of dict: child nodes

        self.attribute_split_point=None         # necessary for the Leaf_Node_Val function (regression case)

@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    # max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None                      # root node of decision tree


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.root = self.tree_construct(X, y, depth = 0)


    def tree_construct(self, X, y, depth = 0):
        # X: pd.Dataframe
        # y: pd.Series

        # Base Cases

        # if y is empty then just return
        if y.empty:
            return
        
        # if X has no attributes left to split the nodes on or the max_depth has been reached
        if X.shape[1]<=0 or depth>=self.max_depth:
            if str(y.dtype=="category"):
                # Classification: Max tree depth reached, so taking the mode (most frequent) of the available outputs

                return TreeNode(val=y.mode()[0],depth=depth)
            else:
                # Regression: Max tree depth reached, so taking the mean of the available outputs
                return TreeNode(val=y.mean(),depth=depth)
        
        # if y takes only 1 unique value
        if(len(y.unique())==1):
            return TreeNode(val=y.unique()[0],depth=depth)


        attribute_split_point=None                  # split point of the attribute if the input/attribute is real or continuous
        info_gain_max=-np.inf

        # finding the attribute for max information gain
        for attr in X.columns:
            attribute_info_gain=None
            split_point=None        # for the attribute in case of real attribute
            attribute_info_gain=information_gain(y, X[attr])

            if(type(attribute_info_gain)==tuple):
                temp = attribute_info_gain
                attribute_info_gain = None
                attribute_info_gain, split_point = temp[0],temp[1]

            if(attribute_info_gain>info_gain_max):
                splitting_attribute = attr
                info_gain_max = attribute_info_gain
                attribute_split_point = split_point

        # create the node of tree with specifying the splitting attribute to differentiate the leaf nodes
        Node = TreeNode(splitting_attr = splitting_attribute)


        # Implementing ID3 Algorithm 

        # for discrete attribute case
        if str(X[splitting_attribute].dtype) == "category":
            value_counts = X[splitting_attribute].value_counts()
            unique_values = value_counts.index
            df = pd.DataFrame({'Value': unique_values, 'Count': value_counts.values})
            
            for value in df['Value']:
                bool_indexes = X[splitting_attribute] == value              # making a series of bools to filter out the X dataframe and y series for recursion

                if True in bool_indexes.values:
                    Node.ChildTreeNodes[value]=self.tree_construct(X[bool_indexes], y[bool_indexes], depth + 1)
            # Remove the attribute column from X about which you are splitting (ID3 algorithm)
            X = X.drop(splitting_attribute,axis=1)

        # for continuous attribute case:
        else:
            Node.attribute_split_point=attribute_split_point
            bool_indexes_higher = X[splitting_attribute] > attribute_split_point
            bool_indexes_lower = X[splitting_attribute] < attribute_split_point
            Node.ChildTreeNodes[(attribute_split_point, "lower")] = self.tree_construct(X[bool_indexes_lower], y[bool_indexes_lower], depth+1)
            Node.ChildTreeNodes[(attribute_split_point, "higher")] = self.tree_construct(X[bool_indexes_higher], y[bool_indexes_higher], depth+1)

        return Node


    def Leaf_Node_Val(self, Node, X):        
        # Base case if the current node is a leaf node, return its value
        if Node.splitting_attr == None:
            return Node.val

        # Check if the splitting attribute is discrete or continuous
        if Node.attribute_split_point==None:                                # Classification
            attribute_value = X[Node.splitting_attr]
            if attribute_value in Node.ChildTreeNodes:
                return self.Leaf_Node_Val(Node.ChildTreeNodes[attribute_value], X)

        else:                                                                # Regression
            if (X[Node.splitting_attr] < Node.attribute_split_point):
                child_key2="lower"
            else:
                child_key2="higher"
            child_node=Node.ChildTreeNodes[(Node.attribute_split_point,child_key2)]
            return self.Leaf_Node_Val(child_node,X)
        
        # return the current node's value, if no matching child node is found (this is a failsafe)
        return Node.val


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        # Traverse the tree constructed to return the predicted vals for the given test inputs.

        prediction = []
        for ind in X.index:
            prediction.append(self.Leaf_Node_Val(self.root,X.loc[ind]))
        
        return pd.Series(prediction)
        # pass
        

    def plot(self, depth=0, Node = None) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        # base cases
        if (Node == None):
            Node = self.root
        if (Node.splitting_attr == None):
            print("    " * depth + "     value = " + str(Node.val) + ", depth = " + str(Node.depth))
            return
        for child_node_dict_keys in Node.ChildTreeNodes:
            # if child node was formed by regression
            if (type(child_node_dict_keys)==tuple):
                if (child_node_dict_keys[1] == "lower"):
                    print("    " * depth + "?( X( Attribute(" + str(Node.splitting_attr) + ") ) < " + str(child_node_dict_keys[0]) + "):")
                elif (child_node_dict_keys[1] == "higher"):
                    print("    " * depth + "?( X( Attribute(" + str(Node.splitting_attr) + ") ) > " + str(child_node_dict_keys[0]) + "):")

            # if child node was formed by classification
            else:
                print("    " * depth + "?( X( Attribute(" + str(Node.splitting_attr) + ") ) = " + str(child_node_dict_keys) + "):")
            
            self.plot(depth + 1, Node.ChildTreeNodes[child_node_dict_keys])