# coding : utf-8

'''

加速度xyzの値から右足・左足のジェスチャーを判定する機械学習（デモ版）
特徴量は加速度3つ
ラベルは'0'（右足）と'1'（左足）の二つ

'''

# データ処理に必須のライブラリ
import pandas as pd # テーブルを'DataFrame'というオブジェクトとして扱うため
import numpy as np  # 計算用ライブラリ


# scikit-learn（機械学習ライブラリ）
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# メイン関数（if __name__ == '__main__:' に関数'main()'を置くのが一般的）
def main():

    '''   データの呼び出し   '''
    # csv（学習データ）の呼び出し
    # X : 特徴量（加速度のデータ）．1列目以降のデータ
    # y : ラベル（右足'0' or 左足'1'）．0列目のデータ
    with open('dataset/accel.csv', 'r', encoding='shift_jis') as f:
        df = pd.read_csv(f)
    print(df)
    X = df.iloc[:, 1:].values
    y = df.iloc[:,0].values
    

    '''   データ分割   '''
    # トレーニングデータとテストデータに分割
    # 'test_size'によってテストデータの大きさを指定する（今回は全体の2/6）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 2/6, random_state=0
    )


    '''  データ変形   '''
    # データの標準化を行う
    # 'fit()'によって適合させ，'transform()'で変形を行う
    # 重要なのは，トレーニングデータで適合したものをテストデータでも使うこと
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    '''   推定機の準備   '''
    # 推定機の初期設定を行う（今回はSVMを使用，ライブラリ名は'SVC'だがSVMを指す）
    # 引数（ハイパーパラメータ）は適当．意味については割愛
    # ニューラルネットワーク版を使用する場合は'MLPClassifier'を使う
    clf = SVC(kernel='rbf', gamma='auto', C=1.0, random_state=0)
    #clf = MLPClassifier(hidden_layer_sizes=(2,), solver='adam')


    '''   学習   '''
    # 'fit()'によって学習を行う
    # 入力はトレーニングデータ'X_train_std'と正解ラベル'y_train'
    clf.fit(X_train_std, y_train)


    '''   推定　　 '''
    # 'predict()'によって，入力データのラベルを推定する
    # 'accuracy_score(正解ラベル, 予測ラベル)'によって正解率を計算できる
    y_pred = clf.predict(X_test_std)
    score = accuracy_score(y_test, y_pred)

    print('入力したテストデータ')
    print(X_test)
    print('正解ラベル')
    print(y_test)
    print('予測ラベル')
    print(y_pred)
    print('正解率 : %.3f' % score )


    '''   交差検定   '''
    # 一応，K分割交差の検定についても記述（今回は3分割）
    # 'StratifiedKFold()'クラスを利用．'split(X, y)'によって，テストデータを重複なく分割できる
    # 'n_splits'の値が分割回数
    kfold = StratifiedKFold(n_splits=3, random_state=1)
    
    # for文で分割回数だけ回す
    # 'train'，'test'に交差検定用のインデックス番号が入る
    # 内部でやっていることは一緒．'scores'に集計して，最後に平均を求める
    scores = []
    for train, test in kfold.split(X, y):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[train], y[train]
        
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        clf = SVC(kernel='rbf', gamma='auto', C=1.0, random_state=0)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)

        # 今回は適合率を求めるため'precision_score()'を利用する
        # 'accuracy_score()'と基本は同じだが，'pos_label'の指定によってどのラベルの
        # 適合率を計算するか指定しなければいけない
        # 再現率の場合は'racall_score()'，F値の場合は'f1_score()'を利用する
        score = precision_score(y_true=y_test, y_pred=y_pred, pos_label=1)
        scores.append(score)

    print('適合率の平均 : %.3f' % np.mean(scores))


if __name__ == '__main__':
    main()