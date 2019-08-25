import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class HousingImpute:
    '''
    initializes an instance based on a csv file.
    
    use self.columns_missing to see columns that need imputing that we have an imputation for
    use self.new_missing to see columns that have missing data which we do not have a protocol for based on training.csv
    
    self.df accesses the dataframe of the object
    self.run_imputers() will auto impute for all possible features

    self.left_to_impute() will show what is left to impute
    
    this class does not handle some completely at random imputing i.e. ID333 in train.csv
    '''
    
    def __init__(self,filename):
        #create df from csv file and check for missing values to impute
        self.df = pd.read_csv(filename, index_col=0)
        self.columns_missing, self.new_missing = self.check_for_missing_columns()

    def save_df(self,filename):
        #saves df to csv format but must choose a filename
        self.df.to_csv(filename+'.csv')

    def left_to_impute(self):
        #call this method after run_imputers() to see what is left to manually impute
        null_counter = self.df.isnull().sum()
        for i in range(len(null_counter)):
            if null_counter[i] > 0:
                print(self.df.columns[i] , null_counter[i])
                print('-'*20)
                temp_column = self.df.columns[i]
                print(self.df.loc[self.df[temp_column].isnull(),temp_column])
                print('-'*20)
    
    def run_imputers(self):
        #active all the imputers needed
        embedded_imputer_columns = ['MasVnrArea','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                              'GarageYrBlt','GarageFinish','GarageQual','GarageCond']
        for column in self.columns_missing:
            try:
                getattr(self,column+'_imputer')()
            except:
                if column not in embedded_imputer_columns:
                    print('no imputer for {}'.format(column))
                else:
                    print('{} imputer embedded in another imputer'.format(column))
    
    def check_for_missing_columns(self):
        train_missing = ['LotFrontage','Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                'Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
                 'PoolQC','Fence','MiscFeature']
        
        null_counter = self.df.isnull().sum()
        columns_missing = []
        new_missing= []
        for i in range(len(null_counter)):
            if null_counter[i] > 0:
                print(self.df.columns[i] , null_counter[i])
                columns_missing.append(self.df.columns[i])    
        for column in columns_missing:
            if (column not in train_missing):
                print('{} does not have a current impute method'.format(column))
                new_missing.append(column)
                columns_missing.remove(column)
                
        return columns_missing, new_missing
        
    def Electrical_imputer(self):
        #simply imputing with most common value of Electrical feature
        self.df.loc[self.df.Electrical.isnull(),'Electrical'] = 'SBrkr'
    
    def MasVnrType_imputer(self):
        #these are datapoints with no type so area must also be zero
        missing_mas = self.df[self.df.MasVnrType.isnull()].index
        self.df.loc[missing_mas,'MasVnrType']='None'
        self.df.loc[missing_mas,'MasVnrArea']=0
    
    def BsmtQual_imputer(self):
        #impute for all missing bsmt rows that are due to no bsmt, not completely random missing values
        missing_basement_indices = self.df[(self.df.BsmtQual.isnull())&(self.df.BsmtCond.isnull())].index
        self.df.loc[missing_basement_indices,'BsmtQual'] = 'No_Bsmt'
        self.df.loc[missing_basement_indices,'BsmtCond'] = 'No_Bsmt'
        self.df.loc[missing_basement_indices,'BsmtExposure'] = 'No_Bsmt'
        self.df.loc[missing_basement_indices,'BsmtFinType1'] = 'No_Bsmt'
        self.df.loc[missing_basement_indices,'BsmtFinType2'] = 'No_Bsmt'
    
    def GarageType_imputer(self):
        #impute for missing garage values due to not having a garage
        missing_garage_indices = self.df[(self.df.GarageType.isnull())&(self.df.GarageQual.isnull())&(self.df.GarageCond.isnull())].index
        self.df.loc[missing_garage_indices,'GarageType']='No_G'
        
        #most like a garage built the year the house is built rather than a garage in the year the house had a remodeling
        self.df.loc[missing_garage_indices,'GarageYrBlt']= self.df.loc[self.df.GarageYrBlt.isnull(),'YearBuilt'] 
        
        self.df.loc[missing_garage_indices,'GarageFinish']='No_G'
        self.df.loc[missing_garage_indices,'GarageQual'] = 'No_G'
        self.df.loc[missing_garage_indices,'GarageCond'] = 'No_G'
    
    def Alley_imputer(self):
        #impute all the missing Alleys as they do not have alleys
        self.df.loc[self.df.Alley.isnull(),'Alley']='No_Alley'
        
    def FireplaceQu_imputer(self):
        #only impute missing FireplaceQu for the ones that have a 0 value for fireplaces
        self.df.loc[(self.df.FireplaceQu.isnull())&(self.df.Fireplaces==0),'FireplaceQu'] = 'No_FP'
        
    def PoolQC_imputer(self):
        #only impute missing PoolQC for the ones that have a 0 value for fireplaces
        self.df.loc[(self.df.PoolQC.isnull())&(self.df.PoolArea==0),'PoolQC']='No_Pool'
        
    def Fence_imputer(self):
        #impute all missing fence values as the house having no fence
        self.df.loc[self.df.Fence.isnull(),'Fence']='No_Fence'
        
    def MiscFeature_imputer(self):
        #impute all missing misc feature values as the house having no miscfeature
        self.df.loc[self.df.MiscFeature.isnull(),'MiscFeature']='No_MF'
        
    def LotFrontage_imputer(self):
        #linear regression for lotfrontage vs lotarea after removing outliers, setting a max at 200 based on visualization
        lr = LinearRegression()
        lr.coef_ = np.array([0.00215388])
        lr.intercept_ = 48.640713607035664
        
        impute_pred = pd.DataFrame(lr.predict(self.df.LotArea[self.df.LotFrontage.isnull()].values.reshape(-1,1)),columns=['LR_Pred'])
        impute_pred['Max'] = 200
        
        self.df.loc[self.df.LotFrontage.isnull(),'LotFrontage'] = impute_pred.min(1).values