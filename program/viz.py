from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib import pyplot as plt
from pycox.evaluation import EvalSurv
from sklearn.metrics import roc_curve, auc

from utils import *

# plt.style.use("ggplot")


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = "serif"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tdir = 'in'
major = 5.0
minor = 3.0
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir

plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor


# plt.rcParams.update(
#     {
#         'text.usetex': False,
#         'font.family': 'stixgeneral',
#         'mathtext.fontset': 'stix',
#     }
# )
#
# CSFONT = {'fontname': 'Times New Roman'}
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.labelweight'] = 'bold'



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
    ax.legend(title="AUC值")
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "{}_135的ROC曲线.svg".format(modelc)), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "{}_135的ROC曲线.pdf".format(modelc)), dpi=600)


def time_roc_plot(res_predictdf, modelnames, df_test, modelc, figname="TimeROC"):
    print("正在绘制依赖时间的ROC曲线：")

    xvalue = np.arange(0, df_test["duration"].max(), 10)



    fig, ax = plt.subplots()

    for c in modelnames:
        model_auc = []
        for xv in xvalue:
            near_idx = find_nearest(res_predictdf[c].columns, xv)
            predict = list(res_predictdf[c][near_idx])
            label = generate_label(df_test, xv)
            fpr, tpr, thresholds = roc_curve(label, predict, pos_label=1)
            model_auc.append(auc(fpr, tpr))

        ax.plot(xvalue, model_auc, label=c)
        ax.legend(loc="lower right")
    ax.set_xlim([0.0, 230.0])
    ax.set_xlim([0.0, 1.05])
    ax.set(xlabel='时间(天)')
    ax.set(ylabel='AUC')
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.svg"), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.pdf"), dpi=600)


# 可视化绘图：整体生存曲线与直方图
def all_KM_plot(encodedat1):
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


# 可视化绘图：分类变量生存曲线【有序】
def km_pairs_plot(dat, cols, rows, columns, nrowthd, ncolthd, type="standardize"):
    if type == "standardize":
        encodedat = dat.copy()
        for col in cols:
            encodedat[col] = dat[col].apply(lambda x: "Low" if x < dat[col].median() else "High")
    else:
        encodedat = dat.copy()

    # fig, ax = plt.subplots(rows, columns, figsize=(50, 50), sharey=True)

    fig, ax = plt.subplots(rows, columns)
    logranklist = []
    featurelist = []
    for nrow in range(rows):
        for ncol in range(columns):
            if nrow >= nrowthd and ncol >= ncolthd:
                break
            feature = cols[ncol * 2 + nrow]

            for v in encodedat[feature].unique():
                df_tmp = encodedat.loc[encodedat[feature] == v]
                if len(df_tmp) == 0:
                    continue
                print(feature, v)
                # KaplanMeier检验
                kmf = KaplanMeierFitter()
                kmf.fit(df_tmp["duration"], df_tmp["event"], label=v)
                logrank = multivariate_logrank_test(encodedat["duration"], encodedat[feature], encodedat["event"])
                logranklist.append([logrank.test_statistic, logrank.p_value])
                featurelist.append(feature)
                p_value_text = \
                ['p-value < 0.001' if logrank.p_value < 0.001 else 'p-value = %.4F' % logrank.p_value][0]

                # 绘制生存曲线
                kmf.plot_survival_function(ax=ax[nrow][ncol])
                ax[nrow][ncol].set_title("{}\n LogRank检验P值:{}".format(feature.replace("_", " "), p_value_text))
                ax[nrow][ncol].set_xlabel("时间(天)")
                ax[nrow][ncol].set_ylabel("S(t)")
                ax[nrow][ncol].legend(loc="upper right")
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "{}类变量的生存曲线图.svg".format(type)), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "{}类变量的生存曲线图.pdf".format(type)), dpi=600)

    return logranklist, featurelist


def lr_finder_plot(res_lrfinder, modelnames):
    print("正在绘制LR寻优图")
    fig, ax = plt.subplots(2, 4)

    # fig, ax = plt.subplots(2, 4, figsize=(15, 10), sharey=True, sharex=True)
    flag = 0
    for c in modelnames:
        ax = fig.add_subplot(2, 4, flag + 1)
        tempdat = res_lrfinder[c].to_pandas()

        ax.plot(tempdat.index, tempdat["train_loss"], label=c)
        ax.set_xlabel("学习率")
        ax.set_ylabel("训练损失")
        ax.legend(loc='upper right')
        flag += 1
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "模型训练LR寻优图.svg"), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "模型训练LR寻优图.pdf"), dpi=600)


def lost_plot(res_log, modelnames):

    fig, ax = plt.subplots(2, 4)

    # fig = plt.figure(figsize=(40, 25))
    flag = 0
    for c in modelnames:
        ax = fig.add_subplot(2, 4, flag + 1)
        tempdat = res_log[c].to_pandas()

        ax.plot(tempdat.index, tempdat["train_loss"], label="Train Loss")
        ax.plot(tempdat.index, tempdat["val_loss"], label="Val Loss")
        ax.set_xlabel("训练损失{}".format(c))
        ax.set_ylabel("训练批次{}".format(c))
        plt.legend(loc='upper right')
        flag += 1
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "模型训练损失图.svg"), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "模型训练损失图.pdf"), dpi=600)


def monthroc_plot(month, modelnames, res_predictdf, df_test):

    fig, ax = plt.subplots(2, 4)
    flag=0
    for c in modelnames:
        ax = fig.add_subplot(2, 4, flag + 1)

        near_idx = find_nearest(res_predictdf[c].columns, month)
        predict_month = list(res_predictdf[c][near_idx])
        label_month = generate_label(df_test, month)
        fpr, tpr, thresholds = roc_curve(label_month, predict_month, pos_label=1)
        model_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label='{}:'.format(c)+'%0.3f' % model_auc)
        ax.legend(title="AUC值")
        flag += 1
    ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR{}'.format(c))
    ax.set_ylabel('TPR{}'.format(c))
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.svg".format(month)), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "{}的ROC曲线.pdf".format(month)), dpi=600)


def timeroc_plot(modelnames, res_predictdf, df_test):

    xvalue = np.arange(0, df_test["duration"].max(), 10).tolist()

    fig, ax = plt.subplots(2, 4)

    for c in modelnames:
        model_auc = []
        for xv in xvalue:
            near_idx = find_nearest(res_predictdf[c].columns, xv)
            predict = list(res_predictdf[c][near_idx])
            label = generate_label(df_test, xv)
            fpr, tpr, thresholds = roc_curve(label, predict, pos_label=1)
            model_auc.append(auc(fpr, tpr))

        ax.plot(xvalue, model_auc, label=c)
        ax.legend(loc="lower right")

    ax.set_xlabel('时间(天)')
    ax.set_ylabel('AUC')
    ax.set_xlim([0.0, 230])
    ax.set_ylim([0.0, 1.05])
    ax.autoscale(tight=True)
    plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.svg"), dpi=600, format='svg')
    plt.savefig(os.path.join(OUTPUT_FILE, "依赖时间的ROC曲线.pdf"), dpi=600)


def eval_res_table(modelnames, res_surv, df_test):
    get_target = lambda df: (df['duration'].values, df['event'].values)
    durations_test, events_test = get_target(df_test)

    cindex_td = []
    ibs = []
    inbll = []
    bs_1_month, bs_3_month, bs_5_month, bs_all_month = [], [], [], []
    nbll_1_month, nbll_3_month, nbll_5_month, nbll_all_month = [], [], [], []

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
                                    nbll[nbll.index < MONTH1].mean() + 1.96 * nbll[nbll.index < MONTH1].std()) + ")")
        nbll_3_month.append("%.3f" % nbll[nbll.index < MONTH3].mean() + "(" + "%.3f" % (
                nbll[nbll.index < MONTH3].mean() - 1.96 * nbll[nbll.index < MONTH3].std()) + "," + "%.3f" % (
                                    nbll[nbll.index < MONTH3].mean() + 1.96 * nbll[nbll.index < MONTH3].std()) + ")")
        nbll_5_month.append("%.3f" % nbll[nbll.index < MONTH5].mean() + "(" + "%.3f" % (
                nbll[nbll.index < MONTH5].mean() - 1.96 * nbll[nbll.index < MONTH5].std()) + "," + "%.3f" % (
                                    nbll[nbll.index < MONTH5].mean() + 1.96 * nbll[nbll.index < MONTH5].std()) + ")")
        nbll_all_month.append("%.3f" % nbll.mean() + "(" + "%.3f" % (nbll.mean() - 1.96 * nbll.std()) + "," + "%.3f" % (
                nbll.mean() + 1.96 * nbll.std()) + ")")

    evalu_res = {
        "modelname": modelnames,
        "cindex_td": cindex_td,
        "ibs": ibs,
        "inbll": inbll,

        "bs_1_month": bs_1_month,
        "bs_3_month": bs_3_month,
        "bs_5_month": bs_5_month,
        "bs_all_month": bs_all_month,

        "nbll_1_month": nbll_1_month,
        "nbll_3_month": nbll_3_month,
        "nbll_5_month": nbll_5_month,
        "nbll_all_month": nbll_all_month

    }

    evalu_res_df = pd.DataFrame(evalu_res)
    return evalu_res_df
