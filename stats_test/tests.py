import pandas as pd
from scipy.stats import chi2_contingency


class test:
    def __init__(self, csv, index=True):
        if index:
            self.data = pd.read_csv(csv, index_col=0)
        else:
            self.data = pd.read_csv(csv)
        self.col = self.data.columns

    def chi_square(self, cols, threshold=0.05):
        # 构建列联表
        test_data1, test_data2 = self.data[cols[0]], self.data[cols[1]]
        contingency = pd.crosstab(test_data1, test_data2)
        try:
            c, p, dof, expected = chi2_contingency(contingency)
            desc = f'卡方检验统计量为{c}，p值为{p}，自由度为{dof}'
            desc += f',{threshold}<{p}，不能拒绝原假设' if threshold < p else f',{threshold}>{p}，拒绝原假设'
            return desc
        except:
            raise Exception('数据应为类型变量，请检查数据')

    def t_test(self, cols, threshold=0.05):
        from scipy.stats import ttest_ind
        test_data1, test_data2 = self.data[cols[0]], self.data[cols[1]]
        statistic, p_value = ttest_ind(test_data1, test_data2)
        desc = f't检验统计量为{statistic}，p值为{p_value}'
        desc += f',{threshold}<{p_value}，不能拒绝原假设' if threshold < p_value else f',{threshold}>{p_value}，拒绝原假设'
        return desc

    def correlation(self, cols=None):
        return self.data[cols].corr() if cols else self.data.corr()

    def f_test(self, cal_col, cols):
        res = []
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        for x in cols:
            df_temp = pd.melt(self.data, id_vars=cal_col, value_vars=x)
            model = ols(f'value ~ C({cal_col})', data=df_temp).fit()
            anova_table = sm.stats.anova_lm(model, typ=1)
            res.append(anova_table)
        return res

    def variance(self, cal_col, cols):
        res = []
        for col in cols:
            df_temp = self.data[[col, cal_col]]
            res.append(df_temp.groupby(cal_col).agg(['count', 'mean', 'std']))
        return res

    def normality_test(self, cols):
        from scipy.stats import kstest, shapiro
        res = pd.DataFrame()
        res['样本量'] = self.data[cols].count()
        res['均值'] = self.data[cols].mean()
        res['标准差'] = self.data[cols].std()
        res['偏度'] = self.data[cols].skew()
        res['峰度'] = self.data[cols].kurt()
        col_name = pd.MultiIndex.from_product([['Kolmogorov-Smirnov检验', 'Shapro-Wilk检验'], ['统计量', 'P值']])
        ks_all, p_ks = [], []
        sw_all, p_sw = [], []
        for col in cols:
            ks, p = kstest(self.data[col], 'norm')
            sw, p_ = shapiro(self.data[col])
            ks_all.append(ks)
            p_ks.append(p)
            sw_all.append(sw)
            p_sw.append(p_)
        res[col_name[0]] = ks_all
        res[col_name[1]] = p_ks
        res[col_name[2]] = sw_all
        res[col_name[3]] = p_sw
        return res
