import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import random
from wurlitzer import pipes
import argparse
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
def read_csv(path_list):
    df = pd.DataFrame()
    for path in path_list:
        df = pd.concat([df,pd.read_csv(path)])
    return df.reset_index(drop=True)

def accuracy_1(conf_matrix):
    conf_sum = 0
    c = 0
    for i in range(len(conf_matrix)):
        row_sum = sum(conf_matrix[i])
        i_minus_1 = conf_matrix[i][i-1] if i-1>=0 else 0
        i_exact = conf_matrix[i][i]
        i_plus_1 = conf_matrix[i][i+1] if i+1<len(conf_matrix) else 0
        conf_sum += row_sum
        c += i_minus_1 + i_exact + i_plus_1
    return c/conf_sum

def make_dataset(df,n,size,by_player=False):
    random_state_list = random.sample(np.arange(0,size).tolist(),size)

    if by_player:
        base_l = []
        for group in df['group'].unique():
            target = df[df['group']==group]
            players = target['Target'].value_counts()
            players = players[players>=20].index
            for player in players[:size]:
                data = target[target['Target']==player]
                for i in range(5):
                    sample = data.sample(n=n,random_state=random_state_list[i])
                    x = sample.drop(columns=[
                        'group','Target','Opponent','TargetRank','OpponentRank','Color','file'
                        ]).mean().tolist()
                    base_l.append([sample['group'].values[0]]+x)
        base = pd.DataFrame(data=base_l,columns=['group']+df.drop(columns=['group','Target','Opponent','TargetRank','OpponentRank','Color','file']).columns.tolist())
        return base
    else:
        base_l = []
        for group in df['group'].unique():
            target = df[df['group']==group]
            for i in range(size):
                ddf = target.sample(n=n,random_state=random_state_list[i])
                x = ddf.drop(columns=['group','Target','Opponent','TargetRank','OpponentRank','Color','file']).mean().tolist()
                base_l.append([ddf['group'].values[0]]+x)
        base = pd.DataFrame(data=base_l,columns=['group']+df.drop(columns=['group','Target','Opponent','TargetRank','OpponentRank','Color','file']).columns.tolist())
        return base

def train_model(df,columns,params):
    X = np.array(df[columns])
    y = np.array(df[['group']]).flatten()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)
    
    callbacks = [
        lgb.early_stopping(50,verbose=False),
        lgb.log_evaluation(period=0)
    ]
    with pipes(stdout=None,stderr=None):
        lgb_train = lgb.Dataset(X_train,y_train)
        lgb_val = lgb.Dataset(X_val,y_val,reference=lgb_train)
    
    model = lgb.train(
        params,lgb_train,num_boost_round=500,
        valid_sets=[lgb_train,lgb_val],
        callbacks=callbacks
        )
    
    return model

def test_model(df,columns,model):
    X_test = np.array(df[columns])
    y_test = np.array(df[['group']]).flatten()
    y_pred = model.predict(X_test)
    y_pred = [round(a) for a in y_pred]
    
    accuracy = accuracy_score(y_test,y_pred)
    std_err = np.sqrt(accuracy*(1-accuracy)/len(y_pred))
    conf_matrix = confusion_matrix(y_test,y_pred)
    accuracy1 = accuracy_1(conf_matrix)
    
    accuracy = format(accuracy*100,'.1f')
    accuracy1 = format(accuracy1*100,'.1f')
    
    return {'accuracy':accuracy,'accuracy1':accuracy1,'stderr':std_err,'conf':conf_matrix}

def train_and_test(train,test,columns,params,type,by_player=False):
    results = []
    if type == 'go':
        train_size = 5000
        test_size = 500
    elif type == 'chess':
        train_size = 13000
        test_size = 500
    
    models = []
    if by_player: 
        n_list = [5,10,15]
    else:          
        n_list = [1,5,10,15,20]
    
    for n in n_list:
        train_data = make_dataset(df=train,n=n,size=train_size)
        test_data = make_dataset(df=test,n=n,size=test_size,by_player=by_player)
        
        model = train_model(train_data,columns,params)
        models.append(model)
        result = test_model(test_data,columns,model)
        results.append(result)
        print(f'finished: type={type} n={n} columns={columns}')
    
    return results

def plot_cofusion_matrix(conf,n,name,target):
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf, annot=True, vmin=0,vmax=500, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label",fontsize=18)
    plt.ylabel("True Label",fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'results/{target}/confusion_matrix_{name}_{n}.png',bbox_inches='tight')
    plt.show()
    
def save_result(results,name,target,by_player=False):
    if by_player:
        n_list = [5,10,15]
    else:
        n_list = [1,5,10,15,20]
    accuracy_list = [result['accuracy'] for result in results]
    accuracy1_list = [result['accuracy1'] for result in results]
    # stderr_list = [result['stderr'] for result in results]
    conf_list = [result['conf'] for result in results]
    results_df = pd.DataFrame(
        data={'accuracy':accuracy_list,
              'accuracy1':accuracy1_list,
            #   'stderr':stderr_list
              },
        index=n_list,
    )
    if not os.path.exists('results'):
        os.mkdir('results')
    results_df.to_csv(f'results/{target}/{name}.csv',index=False)
    for i in range(len(conf_list)):
        conf = conf_list[i]
        n = n_list[i]
        plot_cofusion_matrix(conf,n,name,target)
    return

def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target',type=str,required=True)
    args = parser.parse_args()
    
    target_columns_go = [
        # all columns
        [
            's-score_mean',
            'prior_9d_gmean','prior_7d_gmean','prior_5d_gmean',
            'prior_3d_gmean','prior_1d_gmean','prior_2k_gmean', 
            'prior_4k_gmean','prior_6k_gmean','prior_8k_gmean',
            'prior_10k_gmean', 
            'loss_50_mean','loss_median'
        ],
        # w/o s-score
        [
            'prior_9d_gmean','prior_7d_gmean','prior_5d_gmean',
            'prior_3d_gmean','prior_1d_gmean','prior_2k_gmean', 
            'prior_4k_gmean','prior_6k_gmean','prior_8k_gmean',
            'prior_10k_gmean', 
            'loss_50_mean','loss_median'  
        ],
        # w/o prior
        [
            's-score_mean',
            'loss_50_mean','loss_median'
        ],
        # w/o loss
        [
            's-score_mean',
            'prior_9d_gmean','prior_7d_gmean','prior_5d_gmean',
            'prior_3d_gmean','prior_1d_gmean','prior_2k_gmean', 
            'prior_4k_gmean','prior_6k_gmean','prior_8k_gmean',
            'prior_10k_gmean', 
        ]
    ]
    target_columns_chess = [
        # all columns
        [
            's-score_mean',
            'prior_1100_gmean','prior_1200_gmean','prior_1300_gmean',
            'prior_1400_gmean','prior_1500_gmean','prior_1600_gmean',
            'prior_1700_gmean','prior_1800_gmean','prior_1900_gmean',
            'loss_50_mean','loss_50_std'
        ],
        # w/o s-score
        [
            'prior_1100_gmean','prior_1200_gmean','prior_1300_gmean',
            'prior_1400_gmean','prior_1500_gmean','prior_1600_gmean',
            'prior_1700_gmean','prior_1800_gmean','prior_1900_gmean',
            'loss_50_mean','loss_50_std'
        ],
        # w/o prior
        [
            's-score_mean',
            'loss_50_mean','loss_50_std'
        ],
        # w/o loss
        [
            's-score_mean',
            'prior_1100_gmean','prior_1200_gmean','prior_1300_gmean',
            'prior_1400_gmean','prior_1500_gmean','prior_1600_gmean',
            'prior_1700_gmean','prior_1800_gmean','prior_1900_gmean',
        ]
    ]
    
    if args.target == 'chess':
        group_list = [
            '1000_1200','1200_1400','1400_1600','1600_1800',
            '1800_2000','2000_2200','2200_2400','2400_2600'
        ]
        target_columns_set = target_columns_chess
    elif args.target == 'go':
        group_list = [
            '3-5k','1-2k','1d','2d','3d',
            '4d','5d','6d','7d','8d','9d'
        ]
        target_columns_set = target_columns_go
    train_paths = [f'{args.target}/training/{group}_game_info.csv' for group in group_list]
    test_paths = [f'{args.target}/testing/{group}_game_info.csv' for group in group_list]
    test_by_player_paths = [f'{args.target}/testing_by_player/{group}_game_info.csv' for group in group_list]
    
    train_dataset = read_csv(train_paths)
    test_dataset = read_csv(test_paths)
    test_by_player_dataset = read_csv(test_by_player_paths)
    
    params_regression = {
        'objective':'regression',
        'metric':'mse',
        'verbosity':-1,
        'seed':'42',
        'force_col_wise':True
    }
    
    name_labels = ['all_columns','without_s-score','without_prior','without_loss']
    
    results = []
    # main result
    for idx,target_columns in enumerate(target_columns_set):
        result = train_and_test(train_dataset,test_dataset,target_columns,params_regression,type=args.target)
        save_result(result,name_labels[idx],args.target)
        results.append(result)
    
    results = []
    ## by player 
    for idx,target_columns in enumerate(target_columns_set):
        result = train_and_test(train_dataset,test_by_player_dataset,target_columns,params_regression,type=args.target,by_player=True)
        save_result(result,name_labels[idx]+f'_by_player',args.target,by_player=True)
        results.append(result)

if __name__ == '__main__':
    main()