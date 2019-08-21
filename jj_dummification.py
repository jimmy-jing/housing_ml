import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class LabelEncoders():
    '''
    class to return various label encoder instances based a dict of df columns
    only to be used within class HousingCategorical

    to understand each instance method, please use the self.label_dict to identify relevant features
    '''
    def __init__(self):
        #inits with dict of column names and partial method names to access instance methods
        #not every key has to be used but new ones must be manually added

        self.label_dict = {'ExterQual':'quality',
                     'ExterCond': 'quality',
                     'HeatingQC': 'quality',
                     'KitchenQual': 'quality',
                     'BsmtQual': 'bsmt',
                     'BsmtCond': 'bsmt',
                     'BsmtExposure': 'bsmt_exposure',
                     'FireplaceQu': 'fireplace',
                     'GarageCond': 'garage',
                     'GarageQual': 'garage',
                     'GarageFinish': 'garage_finish',
                     'PoolQC': 'pool'}
    
    def quality_encoder(self):
        quality_encoder = LabelEncoder()
        quality_encoder.classes_ = ['Po','Fa','TA','Gd','Ex']
        return quality_encoder
    
    def bsmt_encoder(self):
        bsmt_encoder = LabelEncoder()
        bsmt_encoder.classes_ = ['No_Bsmt','Po','Fa','TA','Gd','Ex']
        return bsmt_encoder
    
    def bsmt_exposure_encoder(self):
        bsmt_exposure_encoder = LabelEncoder()
        bsmt_exposure_encoder.classes_ = ['No_Bsmt','No','Mn','Av','Gd']
        return bsmt_exposure_encoder
    
    def fireplace_encoder(self):
        fireplace_encoder = LabelEncoder()
        fireplace_encoder.classes_ = ['No_FP','Po','Fa','TA','Gd','Ex']
        return fireplace_encoder
        
    def garage_encoder(self):
        garage_encoder = LabelEncoder()
        garage_encoder.classes_ = ['No_G','Po','Fa','TA','Gd','Ex']
        return garage_encoder
        
    def garage_finish_encoder(self):
        garage_finish_encoder = LabelEncoder()
        garage_finish_encoder.classes_ = ['No_G','Unf','RFn','Fin']
        return garage_finish_encoder
        
    def pool_encoder(self):
        pool_encoder = LabelEncoder()
        pool_encoder.classes_ = ['No_Pool','Fa','TA','Gd','Ex']
        return pool_encoder


class HousingCategorical():
    '''
    class to deal with all categorical features in dataset either via OHE or LE

    user can init first and overwrite self.label_encode_features and self.ohe_features with new list to test different combinations
    '''
    def __init__(self):
        '''
        all three instance attributes can be overwritten after init for customization.
        please use list_checker if user overwrites to make sure information is compliant
        '''
        self.df = pd.read_csv('train_imputed.csv', index_col=0)
        self.label_encode_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',
                                      'HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual',
                                      'GarageCond','PoolQC']
    
        self.ohe_features = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
                             'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
                             'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                             'Foundation','BsmtFinType1','BsmtFinType2','Heating','CentralAir','Electrical',
                             'Functional','GarageType','PavedDrive','Fence','MiscFeature','SaleType',
                             'SaleCondition','MoSold','YrSold']

        
    def one_hot_encode(self):
        #function to create instance of OHE, train on the necessary features, transform those features and replace with transformed results

        ohe = OneHotEncoder(drop='first', sparse=False)
        ohe_df=ohe.fit_transform(self.df[self.ohe_features])
    
        #create a list to store new names after dummification so self.df is still legible
        new_column_names = []
        for i in range(len(ohe.categories_)):
            for j in ohe.categories_[i]:
                if (list(ohe.categories_[i]).index(j)) == 0:
                    pass
                else:
                    new_column_names.append(self.ohe_features[i]+'_'+str(j))
        
        ohe_df = pd.DataFrame(ohe_df, columns=new_column_names)
        ohe_df.index = range(1,1461)
        self.df = pd.merge(self.df.drop(columns=self.ohe_features),ohe_df, left_index=True, right_index=True)
        self.move_sale_price_to_right()
        
    def label_encode(self):
        #create instance of LabelEncoders to label encode all necessary features from 0-n
        #user should beware of the addition assumption of linear spacing within each feature
        label_instance = LabelEncoders()    
        for i in self.label_encode_features:
            labeler = getattr(label_instance, label_instance.label_dict[i]+'_encoder')()
            self.df[i] = labeler.transform(self.df[i])
            
    def move_sale_price_to_right(self):
        #moves target variable y to the last column
        self.df = pd.concat([self.df.drop(columns='SalePrice'), self.df.SalePrice], axis=1)
    
    def list_checker(self):
        #checks for overlap among the two feature lists as well as if all categorical features are accounted for
        for i in self.label_encode_features:
            if i in self.ohe_features:
                print(i+' is in both feature lists')
        
        for i in self.ohe_features:
            if i in self.label_encode_features:
                print(i+' is in both feature lists')

        if len(self.ohe_features)+len(self.label_encode_features) != 45:
        	print('Beware the number of features you are OHE and LE transforming!\n')
        	print('Categorical Features might be missed or continuous variables might be overridden!\n')
        	print('value_counts() analysis recommended!')