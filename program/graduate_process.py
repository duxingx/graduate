from utils import *

if __name__ == '__main__':
    # 从数据库载入数据
    rawdata = dataframe_from_pgsql("public", "res_all")
    dictdata = dataframe_from_pgsql("public", "dict_all")
    dictdata = dictdata.loc[dictdata.VariableType.values != "DateTime", :]
    dat = rawdata.loc[:, dictdata.RawDataColumns1.values]
    print("rawdata shape:", rawdata.shape)
    print("dat shape:", dat.shape)
    print("dictdata shape:", dictdata.shape)

    dat["first_careunit"] = dat["first_careunit"].map(MAPCODE2)
    dat["last_careunit"] = dat["last_careunit"].map(MAPCODE2)
    dat["marital_status"] = dat["marital_status"].str.capitalize()
    dat["ethnicity"] = dat["ethnicity"].str.capitalize()

    # 确定连续变量和离散变量
    continvari, discvari = check_variabletype(dictdat=dictdata)
    # dat[discvari] = dat[discvari].astype("category")
    missrate = calc_missrate(dat)

    # 连续变量
    importantcls = ["实验室检查-肝肾功能", "实验室检查-血常规", "实验室检查-动脉血气分析", "实验室检查-血电解质", "实验室检查-凝血指标",
                    "实验室检查-血电解质", "生命体征", "ICU信息", "尿常规", "透析", "诊断结果"]
    nonimportantcls = ["合并症", "常用重症评分", "出入院信息", "用药情况", "患者基础资料"]
    idcols = list(dictdata.loc[dictdata.VariableType == "ID", "RawDataColumns1"].values)
    # 重要性高,缺失率低的变量:通过计算/经验/业务估计填补[模型填补/插值法填补]
    varis1 = varible_classfi(dictdata, missrate, -1, 0.2, cls=importantcls)
    varis1 = list(set(varis1).intersection(set(continvari)))
    datmis1 = impute_missdeal(dat, varis1)
    # 重要性低,缺失率低的变量:不做处理或简单填充[数据符合均匀分布用均值,数据存在倾斜分布用中位数]
    varis2 = varible_classfi(dictdata, missrate, -1, 0.2, cls=nonimportantcls)
    varis2 = list(set(varis2).intersection(set(continvari)))
    datmis2 = samply_missdeal(datmis1, varis2)

    # 重要性高,缺失率较高的变量:进行分箱的变量分箱,缺失值单独作为一个箱子
    varis3 = varible_classfi(dictdata, missrate, 0.199999999, 0.5, cls=importantcls)
    varis3 = list(set(varis3).intersection(set(continvari)))
    datmis3 = cutbox_missdeal(datmis2, varis3)
    # 重要性低,缺失率较高的变量:不做处理或简单填充[数据符合均匀分布用均值,数据存在倾斜分布用中位数]
    varis4 = varible_classfi(dictdata, missrate, 0.199999999, 0.5, cls=nonimportantcls)
    varis4 = list(set(varis4).intersection(set(continvari)))
    datmis4 = samply_missdeal(datmis3, varis4)

    # 重要性高,缺失率高的变量:进行分箱的变量分箱,缺失值单独作为一个箱子
    varis5 = varible_classfi(dictdata, missrate, 0.49999999, 1, cls=importantcls)
    varis5 = list(set(varis5).intersection(set(continvari)))
    datmis5 = cutbox_missdeal(datmis4, varis5)
    # 重要性低,缺失率高的变量:删除变量
    varis6 = varible_classfi(dictdata, missrate, 0.49999999, 1, cls=nonimportantcls)
    varis6 = list(set(varis6).intersection(set(continvari)))
    datmis6 = delete_missdeal(datmis5, varis6)

    # 离散变量
    # 哑变量填充:将是否缺失单独作为一个子哑变量
    # 重要性高,缺失率低的变量:替换为Missing
    varis7 = varible_classfi(dictdata, missrate, 0, 0.5, cls=importantcls)
    varis7 = list(set(varis7).intersection(set(discvari)))
    datmis7 = conbine_missdeal(datmis6, varis7, conbi_flag=False)
    # 重要性低,缺失率低的变量:替换为Missing并入最低频数的类别
    varis8 = varible_classfi(dictdata, missrate, 0, 0.5, cls=nonimportantcls)
    varis8 = list(set(varis8).intersection(set(discvari)))
    datmis8 = conbine_missdeal(datmis7, varis8, conbi_flag=True)

    # 重要性高,缺失率高的变量:将是否缺失单独作为一个子哑变量
    varis9 = varible_classfi(dictdata, missrate, 0.499, 1, cls=importantcls)
    varis9 = list(set(varis9).intersection(set(discvari)))
    datmis9 = conbine_missdeal(datmis8, varis9, conbi_flag=False)
    # 重要性低,缺失率高的变量:删除变量
    varis10 = varible_classfi(dictdata, missrate, 0.499, 1, cls=nonimportantcls)
    varis10 = list(set(varis10).intersection(set(discvari)))
    datmis10 = delete_missdeal(datmis9, varis10)

    # 格式内容清洗
    # 时间,日期,数值,半全角等显示格式不一致
    # 内容中有不该存在的字符
    # 内容与该字段应有内容不符
    # rawdata = format_deal(rawdata)

    # 逻辑错误清洗
    # 去重
    # 去除不合理值[离群点处理]
    # 箱线图快速发现异常
    # sigma原则:若数据存在正态分布,偏离均值的3sigma之外
    # 基于绝对离差中位数(MAD)
    # 根据异常点的数量和影响:考虑是否将记录删除
    # 若对数据做log-scale变换后消除异常值,则此方法生效,且不损失信息
    # 平均值或中位数代替异常点,简单高效
    oulvaris = set(continvari).difference(set(varis3 + varis5 + varis6))
    datout1 = deal_oultier(datmis10, oulvaris, method='median')

    # 修正矛盾内容:多个类似变量展示的含义自相矛盾

    # 非需求数据清洗
    dattype1 = datout1.copy()
    dattype1[discvari] = dattype1[discvari].astype("category")
    # 删除不需要的字段
    notneed = ["icd_code"]
    datdelet1 = dattype1.drop(notneed, axis=1)

    # 编码
    datdecode1 = datdelet1.copy()
    catvari = datdecode1.dtypes[datdecode1.dtypes == "category"].index.tolist()
    # 等级编码:这些变量是由于连续变量进行离散化后产生的
    cols_categorical = list(set(catvari).intersection(set(continvari)))

    # one-hot编码:剩下的这些变量是本身原本就是分类变量(包括0-1变量和多分类变量)
    varis12 = list(set(catvari).difference(set(continvari)))
    # cols_leave:0-1变量;cols_onehot:多分类变量需要进行one-hot编码
    cols_01, cols_onehot = [], []
    for f in varis12:
        vc = datdecode1[f].value_counts()
        if (f not in ["hadm_id", "subject_id", "stay_id", "admitcustime", "death_flag"]):
            if len(vc) == 2 and (set(vc.index) == {0, 1}):
                cols_01.append(f)
            else:
                cols_onehot.append(f)

    datdecode2 = pd.get_dummies(datdecode1[cols_onehot])
    cols_onehot = list(datdecode2.columns)
    cols_leave = cols_onehot + cols_01
    # cols_standardize:其他剩余的连续变量
    cols_standardize = datdecode1.dtypes[datdecode1.dtypes != "category"].index.drop(
        ["hadm_id", "subject_id", "stay_id", "admitcustime", "death_flag"]).tolist()
    mapcode = {
        "Missing": 0,
        "Low": 1,
        "Lower": 2,
        "Medium": 3,
        "Higher": 4,
        "High": 5
    }
    datdecode1 = order_code(datdecode1, cols_categorical, mapcode)
    # 合并one-hot编码后的数据与患者唯一识别id
    idvar = ["hadm_id", "subject_id", "stay_id"]
    datmerg1 = pd.merge(datdecode1[idvar], datdecode2,
                        left_index=True, right_index=True, how='outer')
    # 与等级编码的变量合并
    datmerg2 = pd.merge(datmerg1, datdecode1[cols_categorical],
                        left_index=True, right_index=True, how='outer')
    # 与0-1变量合并
    datmerg3 = pd.merge(datmerg2, datdecode1[cols_01],
                        left_index=True, right_index=True, how='outer')
    # 合并其他连续变量
    datmerg4 = pd.merge(datmerg3, datdecode1[cols_standardize],
                        left_index=True, right_index=True, how='outer')
    # 合并目标变量
    datmerg5 = pd.merge(datmerg4, datdecode1[dictdata.loc[dictdata.VariableType == "Target", "RawDataColumns1"].values],
                        left_index=True, right_index=True, how='outer')
    # 合并住院时间变量
    datmerg6 = pd.merge(datmerg5, rawdata[["hadm_id", "subject_id", "stay_id", "admittime"]],
                        on=["hadm_id", "subject_id", "stay_id"], how='outer')
    datdeletmerge = pd.merge(datdelet1, rawdata[["hadm_id", "subject_id", "stay_id", "admittime"]],
                        on=["hadm_id", "subject_id", "stay_id"], how='outer')
    # 离散化处理


    # 数据导出
    decodedat = datmerg6.copy()
    encodedat = datdeletmerge.copy()


    encodedat.to_csv(os.path.join(OUTPUT_FILE, "encodedat.csv"))
    decodedat.to_csv(os.path.join(OUTPUT_FILE, "decodedat.csv"))

    save_variable(cols_standardize, os.path.join(OUTPUT_FILE, "cols_standardize.pkl"))
    save_variable(cols_leave, os.path.join(OUTPUT_FILE, "cols_leave.pkl"))
    save_variable(cols_categorical, os.path.join(OUTPUT_FILE, "cols_categorical.pkl"))
    save_variable(varis12, os.path.join(OUTPUT_FILE, "cols_onehot.pkl"))



