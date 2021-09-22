import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit


class regression:
    def __init__(self, csv, tar_col, index=True):
        if index:
            self.data = pd.read_csv(csv, index_col=0)
        else:
            self.data = pd.read_csv(csv)
        self.target = self.data[tar_col]
        self.cols = self.data.columns
        self.out = None

    def linearRegression(self, cols, intercept=True):
        exog = sm.add_constant(self.data[cols]) if intercept else self.data[cols]
        endog = self.target
        model = sm.OLS(endog, exog).fit()
        self.out = model.summary()

    def logisticRegression(self, cols, intercept=True):
        exog = sm.add_constant(self.data[cols]) if intercept else self.data[cols]
        endog = self.target
        if len(np.unique(endog)) > 2:
            Exception('仅支持二分类')
        endog /= np.max(endog)
        model = Logit(endog, exog).fit()
        self.out = model.summary()

    def coxRegression(self, cols):
        sf = sm.SurvfuncRight(self.data[cols], self.target)
        return sf.summary()