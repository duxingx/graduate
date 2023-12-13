import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import torch
import torchtuples as tt
from pycox.models import CoxTime, CoxCC, CoxPH, PMF, MTLR, DeepHitSingle, PCHazard, LogisticHazard
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


DATABASE_NAME = 'mimiciv'  # 数据库名
HOST = "localhost"  # 主机地址
PORT = 5432  # 端口
USER_NAME = 'postgres'  # 用户名
PASSWORD = "root"  # 密码

MONTH1 = 30
MONTH3 = 90
MONTH5 = 150

# 系统变量
RESULT_FILE = "D:\\Xingx\\Project\\20220207-研究生毕业论文-杜兴兴\\result"
FILE_SUFFIX = datetime.now().strftime('%Y-%m-%d')
OUTPUT_FILE = os.path.join(RESULT_FILE, FILE_SUFFIX)
try:
    os.makedirs(OUTPUT_FILE)
except:
    pass

np.random.seed(123456)
_ = torch.manual_seed(123456)

mapcode = {"Missing": 0, "Low": 1, "Lower": 2, "Medium": 3, "Higher": 4, "High": 5}

groupmap = {0: 'Missing', 1: 'Low', 2: 'Lower', 3: 'Medium', 4: 'Higher', 5: 'High'}

modelnamedict = {'coxtime': 'Cox-Time',
                 'coxcc': 'Cox-CC',
                 'deephit': 'DeepHit',
                 'coxph': 'DeepSurv',
                 'logistichazard': 'LogisticHazard',
                 'pchazard': 'PChazard',
                 'mtlr': 'MTLR',
                 'pmf': 'PMF'}

MAPCODE2 = {'Medical Intensive Care Unit (MICU)': 'MICU',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'MSICU',
            'Coronary Care Unit (CCU)': 'CCU',
            'Cardiac Vascular Intensive Care Unit (CVICU)': 'CVICU',
            'Surgical Intensive Care Unit (SICU)': 'SICU',
            'Trauma SICU (TSICU)': 'TSICU',
            'Neuro Intermediate': 'NI',
            'Neuro Stepdown': 'NS',
            'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'NSICU'}

modelnames = ["coxtime", "coxcc", "deephit", "coxph",
                  "logistichazard", "pchazard", "mtlr", "pmf"]


def dataframe_from_pgsql(schema, tablename):
    """
    从pgsql数据库载入数据
    :param schema: 模式
    :param tablename: 表名
    :return: dataframe
    """
    pg_conn = psycopg2.connect(host=HOST, port=PORT, dbname=DATABASE_NAME, user=USER_NAME, password=PASSWORD)
    print("Opened database successfully")

    cur = pg_conn.cursor()

    query = "set search_path to {};SELECT * FROM {};".format(schema, tablename)
    df = pd.read_sql_query(query, pg_conn, index_col=None)

    cur.close()
    pg_conn.close()
    return df


def check_variabletype(df=None, dictdat=None, count=20, exstr=['_max', '_min', '_mean']):
    """
    监测数据集变量类型:离散变量/连续变量
    :param dictdat:是否存在元数据
    :param df:数据集名
    :param count:设置变量的分类数,默认分类数最大为20,若大于则为连续变量
    :param exstr:设置需要排除的连续变量变量名所包含的字符串
    :return:
    """
    if dictdat is not None:
        continvari = list(
            dictdat.loc[(dictdat.VariableType == "Continue") & (dictdat.CategoryCD != 0), "RawDataColumns1"].values)
        discvari = list(
            dictdat.loc[(dictdat.VariableType == "Discrate") & (dictdat.CategoryCD != 0), "RawDataColumns1"].values)
    else:
        freq_list, discvari = [], []
        for col in df.columns:
            x = df.loc[:, col].value_counts().shape[0]
            if x <= count:
                freq_list.append(col)

                if all([col.find(_) < 0 for _ in exstr]):
                    discvari.append(col)
        continvari = list(set(df.columns).difference(set(discvari)))
    print("查找到的连续变量总数:", len(continvari))
    print("查找到的离散变量总数:", len(discvari))

    return continvari, discvari


def calc_missrate(df):
    """
    计算变量的缺失率
    :param df:数据
    :return:返回每个变量的缺失率
    """
    missrate = (df.isnull().sum() / df.shape[0])

    return missrate


def varible_classfi(dictdata, missrate, threshodmin, threshodmax, cls):
    """
    用于变量分类:重要变量,不重要变量,缺失率高的变量,缺失率低的变量
    :param threshodmin:缺失率阈值的下限
    :param threshodmax:缺失率阈值的上限
    :param missrate:各个变量缺失率的序列
    :param cls:能够标识变量Category的列表
    :param dictdata:元数据,能够索引到变量的重要性
    :return:同时满足缺失率要求和重要性要求的变量交集
    """

    varicls = dict()
    # 缺失率筛选
    varicls["mr"] = list(missrate[(missrate < threshodmax) & (missrate > threshodmin)].index)
    # 重要性筛选
    varicls["imp"] = list(dictdata.loc[dictdata.Category.isin(cls), "RawDataColumns1"].values)

    cls_res = set(varicls["mr"]).intersection(set(varicls["imp"]))
    return list(cls_res)


def impute_missdeal(df, varis):
    """
    插值填充缺失值
    :param df:数据
    :param varis:需要填补的变量
    :return:指定变量填补后的数据
    """
    tempdat = df.sort_values("admitcustime")
    # tempdat = tempdat.loc[:, varis + idcols]
    for f in varis:
        tempdat[f] = tempdat[f].interpolate()
        tempdat[f] = tempdat[f].fillna(method="ffill")
        tempdat[f] = tempdat[f].fillna(method="bfill")
    res = tempdat
    return res


def cutbox_missdeal(df, varis):
    """
    用分箱来处理缺失值
    :param df:数据
    :param varis:需要填补的变量
    :return:指定变量填补后的数据
    """
    tempdat = df.sort_values("admitcustime")
    # tempdat = tempdat.loc[:, varis + idcols]

    group_names = ['Low', 'Lower', 'Medium', 'Higher', 'High']
    for f in varis:
        tempdat[f] = pd.cut(tempdat[f], 5, labels=group_names)
        tempdat[f] = tempdat[f].cat.add_categories(['Missing']).fillna("Missing")
    res = tempdat

    return res


def unif_test(seri, seed=123):
    """
    简化的均匀分布检验
    :param seri:series数据
    :param seed:随机种子号
    :return:
    """
    # 随机数据中随机抽取数据，并且保证下次抽取时与此次抽取结果一样
    sam1 = seri.dropna().sample(n=1000, random_state=seed, axis=0)
    sam2 = seri.dropna().sample(n=1000, random_state=seed * 2, axis=0)

    cus1 = 0
    for i in sam1.index:
        mindis = abs(sam1.drop(i) - sam1[i]).min()
        cus1 += mindis

    cus2 = 0
    for i in sam2.index:
        mindis = abs(sam2.drop(i) - sam2[i]).min()
        cus2 += mindis

    testval = cus1 / (cus1 + cus2 + 0.0001)

    return (testval - 0.5) <= 0.05


def samply_missdeal(df, varis):
    """
    数据符合均匀分布用均值,数据存在倾斜分布用中位数
    :return:
    """
    tempdat = df.sort_values("admitcustime")
    # tempdat = tempdat.loc[:, varis + idcols]
    for f in varis:
        if unif_test(tempdat[f]):
            tempdat[f] = tempdat[f].fillna(tempdat[f].mean())
            print(f, ":均匀分布用均值")
        else:
            tempdat[f] = tempdat[f].fillna(tempdat[f].median())
            print(f, ":倾斜分布用中位数")

    res = tempdat
    return res


def delete_missdeal(df, varis):
    """
    直接删除变量
    :param df:数据
    :param varis:需要填补的变量
    :return:指定变量填补后的数据
    """
    tempdat = df.sort_values("admitcustime")
    # tempdat = tempdat.loc[:, varis + idcols]

    tempdat = tempdat.drop(varis, axis=1)
    res = tempdat

    return res


def median_missdeal(df, varis):
    """
    数据符合均匀分布用均值,数据存在倾斜分布用中位数
    :return:
    """
    tempdat = df.sort_values("admitcustime")
    # tempdat = tempdat.loc[:, varis + idcols]
    for f in varis:
        tempdat[f] = tempdat[f].fillna(tempdat[f].median())

    res = tempdat
    return res


def conbine_missdeal(df, varis, conbi_flag):
    """
    替换为Missing并入最低频数的类别
    :param df:数据
    :param varis:需要填补的变量
    :return:指定变量填补后的数据
    """
    tempdat = df.sort_values("admitcustime")

    for f in varis:
        tempdat[f] = tempdat[f].fillna("Missing")
        vc = tempdat[f].value_counts()
        if conbi_flag:
            oldval = vc[vc < 10].index + ["Missing"]
        else:
            oldval = vc[vc < 10].index

        tempdat[f] = tempdat[f].replace(oldval, "Other")
    res = tempdat
    return res


def check_oultier_3sigma(var):
    """
    3 sigma 准则查找异常值
    :param var:
    :return: 如果存在异常值,返回True,并返回上下限; 不存在返回False;
    """
    mean1 = var.quantile(q=0.25)
    mean2 = var.quantile(q=0.75)
    mean3 = mean2 - mean1
    topnum2 = mean2 + 3 * mean3
    bottomnum2 = mean2 - 3 * mean3

    top_flag = any(var > topnum2)
    bot_flag = any(var < bottomnum2)
    return (top_flag or bot_flag), bottomnum2, topnum2


def check_oultier_mad(rawdata):
    pass


def deal_oultier(df, varis, method):
    tempdat = df.copy()
    for f in varis:
        if (tempdat[f].dtypes == "int64") or (tempdat[f].dtypes == "float64"):
            oult_flag, bottomnum, topnum = check_oultier_3sigma(tempdat[f])
            if oult_flag:
                print(f, "\tbottom value:", bottomnum, "\tbottom value:", topnum)
                if method == "median":
                    replace_value = tempdat[f][(tempdat[f] > topnum) & (tempdat[f] < bottomnum)].max()
                    tempdat.loc[(tempdat[f] > topnum) & (tempdat[f] < bottomnum), f] = replace_value
                else:
                    replace_value1 = tempdat[f][tempdat[f] <= topnum].max()
                    tempdat.loc[tempdat[f] > topnum, f] = replace_value1
                    replace_value2 = tempdat[f][tempdat[f] >= bottomnum].min()
                    tempdat.loc[tempdat[f] < bottomnum, f] = replace_value2

    return tempdat


def order_code(df, varis, mapcode):
    """
    等级变量编码
    :param df:
    :param varis:
    :param mapcode:
    :return:
    """
    tempdat = df.copy()
    for f in varis:
        tempdat[f] = tempdat[f].map(mapcode)

    res = tempdat
    return res


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    print(v, "存储至:", filename)
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def import_deal_dat():
    # 编码前的数据
    encodedat = pd.read_csv(os.path.join(OUTPUT_FILE, "datdelet1.csv"), index_col=0)
    encodedat = encodedat.drop(["hadm_id", "subject_id", "stay_id"], axis=1)
    encodedat = encodedat.rename(columns={"admitcustime": "duration", "death_flag": "event"})
    encodedat = encodedat.loc[
                (encodedat["duration"] > 0) & (encodedat["age"] >= 18) & (encodedat["duration"] < 300), :]

    # 编码后的数据（包括one-hot、order等编码）
    decodedat = pd.read_csv(os.path.join(OUTPUT_FILE, "datoutput.csv"), index_col=0)
    decodedat = decodedat.drop(["hadm_id", "subject_id", "stay_id"], axis=1)
    decodedat = decodedat.rename(columns={"admitcustime": "duration", "death_flag": "event"})
    decodedat = decodedat.loc[
                (decodedat["duration"] > 0) & (decodedat["age"] >= 18) & (decodedat["duration"] < 300), :]

    # 变量
    cols_standardize = load_variavle(os.path.join(OUTPUT_FILE, "cols_standardize.pkl"))  # 连续变量
    cols_categorical = load_variavle(os.path.join(OUTPUT_FILE, "cols_categorical.pkl"))  # 分类变量：有序变量（order）
    cols_leave = load_variavle(os.path.join(OUTPUT_FILE, "cols_leave.pkl"))  # 分类变量：经过one-hot编码后的变量
    cols_onehot = load_variavle(os.path.join(OUTPUT_FILE, "cols_onehot.pkl"))  # 分类变量：未经过one-hot编码的变量

    # 类型转换
    decodedat = decodedat.astype("float32")
    decodedat["event"] = decodedat["event"].astype("int32")

    # 复制一个可操作的dataframe
    encodedat1 = encodedat.copy()
    decodedat1 = decodedat.copy()

    return decodedat1, encodedat1, cols_standardize, cols_categorical, cols_leave, cols_onehot


def split_dat(df, frac=0.2):
    df_test = df.sample(frac=frac)  # 提取测试集
    df_train = df.drop(df_test.index)  # 剩余训练集
    df_val = df_train.sample(frac=frac)  # 提取验证集
    df_train = df_train.drop(df_val.index)  # 剩余训练集
    return df_train, df_val, df_test


def feature_transfer(cols_standardize, cols_leave, cols_categorical, df_train, df_val, df_test):
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + leave)
    x_mapper_long = DataFrameMapper(categorical)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)
    x_train = np.concatenate((x_train[0], x_train[1]), axis=1).astype("float32")
    x_val = np.concatenate((x_val[0], x_val[1]), axis=1).astype("float32")
    x_test = np.concatenate((x_test[0], x_test[1]), axis=1).astype("float32")
    return x_train, x_val, x_test


def mode_all(datparams, netparams, modelparams, trainparams):
    # 数据处理
    df_train = datparams["df_train"]
    df_val = datparams["df_val"]
    df_test = datparams["df_test"]
    x_val = datparams["x_val"]
    x_train = datparams["x_train"]
    # 构建网络
    in_features = netparams["in_features"]
    num_nodes = netparams["num_nodes"]
    batch_norm = netparams["batch_norm"]
    dropout = netparams["dropout"]
    output_bias = netparams["output_bias"]
    # 构造模型
    optimizer = modelparams["optimizer"]
    tolerance = modelparams["tolerance"]
    modelname = modelparams["modelname"]
    alpha = modelparams["alpha"]
    sigma = modelparams["sigma"]
    # 训练模型
    batch_size = trainparams["batch_size"]
    epochs = trainparams["epochs"]
    callbacks = trainparams["callbacks"]
    verbose = trainparams["verbose"]

    num_durations = 20
    if modelname == "coxtime":
        labtrans = CoxTime.label_transform()
    elif modelname == "deephit":
        labtrans = DeepHitSingle.label_transform(num_durations)
    elif modelname == "logistichazard":
        labtrans = LogisticHazard.label_transform(num_durations)
    elif modelname == "pmf":
        labtrans = PMF.label_transform(num_durations)
    elif modelname == "mtlr":
        labtrans = MTLR.label_transform(num_durations)
    elif modelname == "pchazard":
        labtrans = PCHazard.label_transform(num_durations)
    else:
        labtrans = None

    # 目标变量转换
    get_target = lambda df: (df['duration'].values, df['event'].values)
    durations_test, events_test = get_target(df_test)

    if labtrans:
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))
        out_features = labtrans.out_features
    else:
        y_train = get_target(df_train)
        y_val = get_target(df_val)
        out_features = 1

    val = tt.tuplefy(x_val, y_val)

    if str.lower(modelname) == "coxtime":
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
        model = CoxTime(net, optimizer, labtrans=labtrans)
    elif str.lower(modelname) == "coxcc":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = CoxCC(net, optimizer)
    elif str.lower(modelname) == "coxph":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = CoxPH(net, optimizer)
    elif str.lower(modelname) == "deephit":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = DeepHitSingle(net, optimizer, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    elif str.lower(modelname) == "logistichazard":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
    elif str.lower(modelname) == "pmf":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = PMF(net, optimizer, duration_index=labtrans.cuts)
    elif str.lower(modelname) == "mtlr":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = MTLR(net, optimizer, duration_index=labtrans.cuts)
    elif str.lower(modelname) == "pchazard":
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = PCHazard(net, optimizer, duration_index=labtrans.cuts)
    else:
        print("请输入正确模型名称")
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = PCHazard(net, optimizer, duration_index=labtrans.cuts)

    print("模型:", model)

    lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=tolerance)
    model.optimizer.set_lr(lrfinder.get_best_lr())
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val.repeat(10).cat())

    return net, model, lrfinder, log, durations_test, events_test


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def generate_label(data, time):
    label_list = []
    for i in range(len(data)):
        if data['event'].iloc[i] == 0:
            label = 0
        else:
            if data['duration'].iloc[i] > time:
                label = 0
            else:
                label = 1
        label_list.append(label)
    return label_list
