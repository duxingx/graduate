import matplotlib.ticker as mtick
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib import pyplot as plt
from pycox.evaluation import EvalSurv
from sklearn.metrics import roc_curve, auc

from utils import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['figure.figsize'] = (6.0, 4.0)

if __name__ == '__main__':
    # 导入数据
    decodedat1, encodedat1, cols_standardize, cols_categorical, cols_leave, cols_onehot = import_deal_dat()

    # 划分数据集
    df_train, df_val, df_test = split_dat(decodedat1)

    # 特征转换
    x_train, x_val, x_test = feature_transfer(cols_standardize, cols_leave, cols_categorical, df_train, df_val, df_test)

    res_net = dict()
    res_model = dict()
    res_lrfinder = dict()
    res_log = dict()
    res_surv = dict()
    res_predictdf = dict()
    res_durations_test = dict()
    res_events_test = dict()

    for c in modelnames:
        datparams = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "x_val": x_val,
            "x_train": x_train,
        }
        netparams = {
            "in_features": x_train.shape[1],
            "num_nodes": [32, 32],
            "batch_norm": True,
            "dropout": 0.1,
            "output_bias": False,
        }
        modelparams = {
            "optimizer": tt.optim.Adam,
            "tolerance": 10,
            "alpha": 0.2,
            "sigma": 0.1,
            "modelname": c
        }

        trainparams = {
            "batch_size": 256,
            "epochs": 512,
            "callbacks": [tt.callbacks.EarlyStopping()],
            "verbose": True
        }

        res_net[c], res_model[c], res_lrfinder[c], res_log[c], res_durations_test[c], res_events_test[c] = mode_all(
            datparams, netparams, modelparams, trainparams)

        if c in ["coxcc", "coxtime", "coxph"]:
            print("正在计算预测结果")
            _ = res_model[c].compute_baseline_hazards()
            res_surv[c] = res_model[c].predict_surv_df(x_test)
        else:
            print("正在计算预测结果")
            res_surv[c] = res_model[c].predict_surv_df(x_test)

        res_predictdf[c] = 1 - res_surv[c].T


    def roc_plot(predict_df, df_test, modelc):
        print("正在绘制1月，3月，5月的ROC曲线")

        near_1_idx = find_nearest(predict_df.columns, MONTH1)
        predict_1_month = list(predict_df[near_1_idx])
        near_3_idx = find_nearest(predict_df.columns, MONTH3)
        predict_3_month = list(predict_df[near_3_idx])
        near_5_idx = find_nearest(predict_df.columns, MONTH5)
        predict_5_month = list(predict_df[near_5_idx])

        label_1_month = generate_label(df_test, MONTH1)
        label_3_month = generate_label(df_test, MONTH3)
        label_5_month = generate_label(df_test, MONTH5)

        fpr_1, tpr_1, thresholds_1 = roc_curve(label_1_month, predict_1_month, pos_label=1)
        fpr_3, tpr_3, thresholds_3 = roc_curve(label_3_month, predict_3_month, pos_label=1)
        fpr_5, tpr_5, thresholds_5 = roc_curve(label_5_month, predict_5_month, pos_label=1)

        model_auc_1 = auc(fpr_1, tpr_1)
        model_auc_3 = auc(fpr_3, tpr_3)
        model_auc_5 = auc(fpr_5, tpr_5)

        fig, ax = plt.subplots()

        ax.plot(fpr_1, tpr_1, label="1个月:%.3f" % model_auc_1)
        ax.plot(fpr_3, tpr_3, label="3个月:%.3f" % model_auc_3)
        ax.plot(fpr_5, tpr_5, label="5个月:%.3f" % model_auc_5)
        ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
        ax.set_xlim([0.0, 1.0])
        ax.set_xlim([0.0, 1.05])
        ax.set(xlabel='FPR({})'.format(modelc))
        ax.set(ylabel='TPR({})'.format(modelc))
        ax.legend(title="AUC值", loc="lower right")
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "{}_135的ROC曲线.svg".format(modelc)), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "{}_135的ROC曲线.pdf".format(modelc)), dpi=600)


    # 绘制roc曲线图（1-3-5）：8个模型分开绘制
    for c in modelnames:
        roc_plot(res_predictdf[c], df_test=df_test, modelc=modelnamedict[c])


    def lr_finder_plot(res_lrfinder, modelnames):
        print("正在绘制LR寻优图")

        fig = plt.figure(figsize=(12, 8))
        flag = 0
        for c in modelnames:
            ax = fig.add_subplot(2, 4, flag + 1)

            res_lrfinder[c].plot(ax=ax)

            ax.set_xlabel("学习率({})".format(modelnamedict[c]))
            if flag in [0, 4]:
                ax.set_ylabel("训练损失")
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            flag += 1
        plt.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "模型训练LR寻优图.svg"), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "模型训练LR寻优图.pdf"), dpi=600)


    # 绘制LR寻优图：8个模型绘制到一张图中
    lr_finder_plot(res_lrfinder, modelnames)


    def lost_plot(res_log, modelnames):
        print("正在绘制训练损失图")

        fig = plt.figure(figsize=(12, 8))
        flag = 0
        for c in modelnames:
            ax = fig.add_subplot(2, 4, flag + 1)
            tempdat = res_log[c].to_pandas()

            ax.plot(tempdat.index, tempdat["train_loss"], label="Train Loss")
            ax.plot(tempdat.index, tempdat["val_loss"], label="Val Loss")

            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            ax.set_xlabel("训练批次({})".format(modelnamedict[c]))
            if flag in [0, 4]:
                ax.set_ylabel("训练损失")
            else:
                ax.set_ylabel("")

            plt.legend(loc='upper right')
            flag += 1
        plt.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "模型训练损失图.svg"), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "模型训练损失图.pdf"), dpi=600)


    # 绘制训练损失图：8个模型分开绘制
    lost_plot(res_log, modelnames)


    def monthroc_plot(month, modelnames, res_predictdf, df_test):
        print("正在绘制按月的roc曲线图")
        fig = plt.figure()
        ax = fig.add_subplot()

        flag = 0
        for c in modelnames:
            near_idx = find_nearest(res_predictdf[c].columns, month)
            predict_month = list(res_predictdf[c][near_idx])
            label_month = generate_label(df_test, month)
            fpr, tpr, thresholds = roc_curve(label_month, predict_month, pos_label=1)
            model_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label='{}:'.format(modelnamedict[c]) + '%0.3f' % model_auc)
            ax.legend(title="AUC值", loc="lower right")
            flag += 1
        ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR({})'.format(modelnamedict[c]))
        ax.set_ylabel('TPR({})'.format(modelnamedict[c]))
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.svg".format(month)), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.pdf".format(month)), dpi=600)


    # 绘制roc曲线图（1）：8个模型绘制到一张图中
    monthroc_plot(MONTH1, modelnames, res_predictdf, df_test)


    def monthroc_plot(month, modelnames, res_predictdf, df_test):
        print("正在绘制按月的roc曲线图")
        fig = plt.figure()
        ax = fig.add_subplot()

        flag = 0
        for c in modelnames:
            near_idx = find_nearest(res_predictdf[c].columns, month)
            predict_month = list(res_predictdf[c][near_idx])
            label_month = generate_label(df_test, month)
            fpr, tpr, thresholds = roc_curve(label_month, predict_month, pos_label=1)
            model_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label='{}:'.format(modelnamedict[c]) + '%0.3f' % model_auc)
            ax.legend(title="AUC值", loc="lower right")
            flag += 1
        ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR({})'.format(modelnamedict[c]))
        ax.set_ylabel('TPR({})'.format(modelnamedict[c]))
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.svg".format(month)), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.pdf".format(month)), dpi=600)


    # 绘制roc曲线图（3）：8个模型绘制到一张图中
    monthroc_plot(MONTH3, modelnames, res_predictdf, df_test)


    def monthroc_plot(month, modelnames, res_predictdf, df_test):
        print("正在绘制按月的roc曲线图")
        fig = plt.figure()
        ax = fig.add_subplot()

        flag = 0
        for c in modelnames:
            near_idx = find_nearest(res_predictdf[c].columns, month)
            predict_month = list(res_predictdf[c][near_idx])
            label_month = generate_label(df_test, month)
            fpr, tpr, thresholds = roc_curve(label_month, predict_month, pos_label=1)
            model_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label='{}:'.format(modelnamedict[c]) + '%0.3f' % model_auc)
            ax.legend(title="AUC值", loc="lower right")
            flag += 1
        ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR({})'.format(modelnamedict[c]))
        ax.set_ylabel('TPR({})'.format(modelnamedict[c]))
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.svg".format(month)), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.pdf".format(month)), dpi=600)


    # 绘制roc曲线图（5）：8个模型绘制到一张图中
    monthroc_plot(MONTH5, modelnames, res_predictdf, df_test)


    def timeroc_plot(modelnames, res_predictdf, df_test):

        xvalue = np.arange(0, df_test["duration"].max(), 10).tolist()

        fig = plt.figure()
        ax = fig.add_subplot()

        for c in modelnames:
            model_auc = []
            for xv in xvalue:
                near_idx = find_nearest(res_predictdf[c].columns, xv)
                predict = list(res_predictdf[c][near_idx])
                label = generate_label(df_test, xv)
                fpr, tpr, thresholds = roc_curve(label, predict, pos_label=1)

                model_auc.append(auc(fpr, tpr))
            plt.plot(xvalue, model_auc, label=modelnamedict[c])
            ax.legend(loc="lower left")

        ax.set_xlabel('时间(天)')
        ax.set_ylabel('AUC')
        ax.set_xlim([0.0, 230])
        ax.set_ylim([0.0, 1.05])
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.svg"), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.pdf"), dpi=600)


    # 绘制time-roc曲线图：8个模型绘制到一张图中
    timeroc_plot(modelnames, res_predictdf, df_test)


    def eval_res_table(modelnames, res_surv, df_test):
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_test, events_test = get_target(df_test)

        cindex_td = []
        ibs = []
        inbll = []
        bs_1_month, bs_3_month, bs_5_month, bs_all_month = [], [], [], []
        nbll_1_month, nbll_3_month, nbll_5_month, nbll_all_month = [], [], [], []
        modelname = []
        for c in modelnames:
            time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
            ev = EvalSurv(res_surv[c], durations_test, events_test, censor_surv='km')

            cindex_td.append("%.3f" % ev.concordance_td('antolini'))
            ibs.append("%.3f" % ev.integrated_brier_score(time_grid))
            inbll.append("%.3f" % ev.integrated_nbll(time_grid))

            bs = ev.brier_score(time_grid)

            bs_1_month.append("%.3f" % bs[bs.index < MONTH1].mean() + "(" + "%.3f" % (
                    bs[bs.index < MONTH1].mean() - 1.96 * bs[bs.index < MONTH1].std()) + "," + "%.3f" % (
                                      bs[bs.index < MONTH1].mean() + 1.96 * bs[bs.index < MONTH1].std()) + ")")
            bs_3_month.append("%.3f" % bs[bs.index < MONTH3].mean() + "(" + "%.3f" % (
                    bs[bs.index < MONTH3].mean() - 1.96 * bs[bs.index < MONTH3].std()) + "," + "%.3f" % (
                                      bs[bs.index < MONTH3].mean() + 1.96 * bs[bs.index < MONTH1].std()) + ")")
            bs_5_month.append("%.3f" % bs[bs.index < MONTH5].mean() + "(" + "%.3f" % (
                    bs[bs.index < MONTH5].mean() - 1.96 * bs[bs.index < MONTH5].std()) + "," + "%.3f" % (
                                      bs[bs.index < MONTH5].mean() + 1.96 * bs[bs.index < MONTH1].std()) + ")")
            bs_all_month.append("%.3f" % bs.mean() + "(" + "%.3f" % (bs.mean() - 1.96 * bs.std()) + "," + "%.3f" % (
                    bs.mean() + 1.96 * bs.std()) + ")")

            nbll = ev.nbll(time_grid)
            nbll_1_month.append("%.3f" % nbll[nbll.index < MONTH1].mean() + "(" + "%.3f" % (
                    nbll[nbll.index < MONTH1].mean() - 1.96 * nbll[nbll.index < MONTH1].std()) + "," + "%.3f" % (
                                        nbll[nbll.index < MONTH1].mean() + 1.96 * nbll[
                                    nbll.index < MONTH1].std()) + ")")
            nbll_3_month.append("%.3f" % nbll[nbll.index < MONTH3].mean() + "(" + "%.3f" % (
                    nbll[nbll.index < MONTH3].mean() - 1.96 * nbll[nbll.index < MONTH3].std()) + "," + "%.3f" % (
                                        nbll[nbll.index < MONTH3].mean() + 1.96 * nbll[
                                    nbll.index < MONTH3].std()) + ")")
            nbll_5_month.append("%.3f" % nbll[nbll.index < MONTH5].mean() + "(" + "%.3f" % (
                    nbll[nbll.index < MONTH5].mean() - 1.96 * nbll[nbll.index < MONTH5].std()) + "," + "%.3f" % (
                                        nbll[nbll.index < MONTH5].mean() + 1.96 * nbll[
                                    nbll.index < MONTH5].std()) + ")")
            nbll_all_month.append(
                "%.3f" % nbll.mean() + "(" + "%.3f" % (nbll.mean() - 1.96 * nbll.std()) + "," + "%.3f" % (
                        nbll.mean() + 1.96 * nbll.std()) + ")")

            modelname.append(modelnamedict[c])
        evalu_res = {
            "模型": modelname,
            "C-Index(td)": cindex_td,
            "IBS": ibs,
            "IBLL": inbll,

            "BS(1个月)": bs_1_month,
            "BS(3个月)": bs_3_month,
            "BS(5个月)": bs_5_month,
            "BS(全生存期)": bs_all_month,

            "NBLL(1个月)": nbll_1_month,
            "NBLL(3个月)": nbll_3_month,
            "NBLL(5个月)": nbll_5_month,
            "NBLL(全生存期)": nbll_all_month

        }

        evalu_res_df = pd.DataFrame(evalu_res)
        return evalu_res_df


    # 各个指标的评价：
    evalu_res_df = eval_res_table(modelnames, res_surv, df_test)
    evalu_res_df.to_csv(os.path.join(OUTPUT_FILE, "各个模型的评价指标.csv"), encoding="utf_8_sig")


    def km_pairs_plot(dat, cols, type="standardize"):
        if type == "standardize":
            encodedat = dat.copy()
            for col in cols:
                encodedat[col] = dat[col].apply(lambda x: "Low" if x < dat[col].median() else "High")
        elif type == "onehot":
            encodedat = dat.copy()
            for col in cols:
                if set(encodedat[col].unique().tolist()) <= set(["Missing", 0, 1]):
                    encodedat[col] = dat[col].map({0: "False", 1: "Yes"})
                if col == "first_careunit" or col == "last_careunit":
                    encodedat[col] = dat[col].map(MAPCODE2)
        else:
            encodedat = dat.copy()

        logranklist = []
        featurelist = []
        flag = 0
        f = 0
        for feature in cols:
            if flag % 8 == 0:
                flag = 0
                fig = plt.figure(figsize=(12, 8))

            figax = plt.subplot(2, 4, flag + 1)
            for v in encodedat[feature].unique():
                df_tmp = encodedat.loc[encodedat[feature] == v]
                if len(df_tmp) == 0:
                    continue
                # KaplanMeier检验
                kmf = KaplanMeierFitter()
                kmf.fit(df_tmp["duration"], df_tmp["event"], label=v)
                # 绘制生存曲线
                kmf.plot_survival_function()

            logrank = multivariate_logrank_test(encodedat["duration"], encodedat[feature], encodedat["event"])
            logranklist.append([logrank.test_statistic, logrank.p_value])
            featurelist.append(feature)

            if flag in [0, 4]:
                plt.ylabel("S(t)")
            else:
                plt.ylabel("")
            plt.xlabel("{}:时间(天)".format(feature.replace("_", " ")))

            plt.legend(loc="upper right",
                       title="P值:{}".format(['<0.001' if logrank.p_value < 0.001 else '%.4F' % logrank.p_value][0]))

            flag += 1

            if flag % 8 == 0 or (f == 8 * (len(cols) // 8) and flag == len(cols) % 8):
                f += 8
                plt.autoscale(tight=True)
                plt.savefig(os.path.join(OUTPUT_FILE, "{}类变量的生存曲线图{}.svg".format(type, f)), dpi=600, format='svg')
                plt.savefig(os.path.join(OUTPUT_FILE, "{}类变量的生存曲线图{}.pdf".format(type, f)), dpi=600)

        return logranklist, featurelist


    # 绘制各个变量的K-M曲线
    categorical_logranklist, categorical_featurelist = km_pairs_plot(encodedat1, cols=cols_categorical,
                                                                     type="categorical")
    standardize_logranklist, standardize_featurelist = km_pairs_plot(encodedat1, cols=cols_standardize,
                                                                     type="standardize")
    onehot_logranklist, onehot_featurelist = km_pairs_plot(encodedat1, cols=cols_onehot, type="onehot")

    # 表格：各个变量的log-rank检验以及样本基本信息表
    cat_testvalue = []
    cat_pvalue = []
    for i in categorical_logranklist:
        cat_testvalue.append(i[0])
        cat_pvalue.append(i[1])

    sta_testvalue = []
    sta_pvalue = []
    for i in standardize_logranklist:
        sta_testvalue.append(i[0])
        sta_pvalue.append(i[1])

    one_testvalue = []
    one_pvalue = []
    for i in onehot_logranklist:
        one_testvalue.append(i[0])
        one_pvalue.append(i[1])

    cat_dict = {
        "变量名": categorical_featurelist,
        "统计量值": cat_testvalue,
        "P值": cat_pvalue
    }
    sta_dict = {
        "变量名": standardize_featurelist,
        "统计量值": sta_testvalue,
        "P值": sta_pvalue
    }
    one_dict = {
        "变量名": onehot_featurelist,
        "统计量值": one_testvalue,
        "P值": one_pvalue
    }

    cat_logrank_df = pd.DataFrame(cat_dict)
    sta_logrank_df = pd.DataFrame(sta_dict)
    one_logrank_df = pd.DataFrame(one_dict)

    cat_logrank_df.to_csv(os.path.join(OUTPUT_FILE, "分类数据logrank检验结果.csv"), encoding="utf_8_sig")
    sta_logrank_df.to_csv(os.path.join(OUTPUT_FILE, "标准化数据logrank检验结果.csv"), encoding="utf_8_sig")
    one_logrank_df.to_csv(os.path.join(OUTPUT_FILE, "onehot数据logrank检验结果.csv"), encoding="utf_8_sig")


    # 可视化绘图：整体生存曲线与直方图
    def all_hist_KM_plot(encodedat1):
        print("正在绘制生存时间直方图")

        fig, ax = plt.subplots()

        ax.hist(encodedat1.loc[encodedat1["event"] == 0, "duration"], bins=40, label='未发生院内死亡')
        ax.hist(encodedat1.loc[encodedat1["event"] == 1, "duration"], bins=40, label='发生院内死亡')
        ax.set_xlabel("频次")
        ax.set_ylabel("时间(天)")
        ax.legend(loc='upper right')
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "生存时间直方图.svg"), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "生存时间直方图.pdf"), dpi=600)

        print("正在绘制生存时间KM图")
        kmf = KaplanMeierFitter()
        kmf.fit(encodedat1['duration'], event_observed=encodedat1['event'])
        fig, ax = plt.subplots()
        kmf.plot_survival_function(label='Kaplan-Meier生存曲线')

        plt.hlines(0.5, ls='--', color="red", xmin=0, xmax=kmf.median_survival_time_)
        plt.vlines(kmf.median_survival_time_, ls='--', color="red", ymin=0, ymax=0.5, label='中位生存时间')
        ax.set_xlabel("时间(天)")
        ax.set_ylabel("S(t)")
        plt.legend(loc='upper right')
        ax.autoscale(tight=True)
        plt.savefig(os.path.join(OUTPUT_FILE, "生存时间KM图.svg"), dpi=600, format='svg')
        plt.savefig(os.path.join(OUTPUT_FILE, "生存时间KM图.pdf"), dpi=600)


    # 绘制整体生存曲线
    all_hist_KM_plot(encodedat1)
