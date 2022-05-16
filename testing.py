#!/usr/bin/env python
from dataclasses import dataclass
from Index_Rebalancer import *

class TestSomeFunctions(unittest.TestCase):
    '''
    python -m unittest -v testing.py
    '''
    @classmethod
    def setUpClass(self):
        """We only want to pull this data once for each TestCase since it is an expensive operation"""
        self.df = get_data(path)
        self.df1 = intializer()
        self.df2 = intial_40_selection()
        self.df3 = prepare_parameters()

        
 
    def test_columns_present(self):
        """ensures that the expected columns are all present"""
        self.assertIn("Company Name", self.df.columns)
        self.assertIn("Sector Code", self.df.columns)
        self.assertIn("FCap Wt", self.df.columns)
        self.assertIn("Z_Value", self.df.columns)
 
    def test_non_empty(self):
        """ensures that there is more than one row of data"""
        self.assertNotEqual(len(self.df1.index), 0)
        self.assertNotEqual(len(self.df2.index), 0)
        self.assertNotEqual(len(self.df3.index), 0)
    
    def test_input_value(self):
        """stress test the inputs"""
        self.assertRaises(TypeError,minimization_init, True)
        # self.assertIsInstance(prepare_parameters, pd.DataFrame)
        # self.assertRaises(TypeError,run, True)
        # self.assertRaises(KeyError,run, True)
 




