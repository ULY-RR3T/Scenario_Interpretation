from util import *
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
from matplotlib import rcParams
from datetime import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
rcParams.update({'figure.autolayout': True})
sns.set()
sns.set_style("whitegrid")



quantiles = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])

parser = argparse.ArgumentParser(description="Returns a plot and the values of the difference of two scenarios"
                                             'in CDC\' challenge via the methods '
                                             "from the paper DFRV in special cases")
parser.add_argument('-r', '--round', type=int, help="The round that we extract the results from")
parser.add_argument('-m', '--model_name', type=str, help="The name of the model begin evaluated")
parser.add_argument('-sx', '--scenarioX', type=str, help="The scenario X (X is larger than Y if assumed "
                                                "(semi)infomrative)")
parser.add_argument('-sy', '--scenarioY', type=str, help="The scenario X (X is larger than Y if assumed "
                                                "(semi)infomrative)")
parser.add_argument('-a','--alpha', type=float, help="The confidence level used in evaluatino")
parser.add_argument('-t','--t_app', type=int, help="The number of weeks after the initial prediction time the scenarios"
                                            "begin to take effect")
parser.add_argument('-e','--e', type=float, help="The epsilon that is going to be assumed for models")
parser.add_argument('-tt','--target_type', type=str, help='The type of target being evaluated. Can be one of case,death,hosp')
parser.add_argument('-ic','--inc_cum', type=str, help='increment or culmulative results')
args = parser.parse_args()


def read_data(link):
    return pd.read_csv(link).dropna()

def generate_epsilon_distribution(figsize = (6, 5)):
    names = ['USC_SIkJalpha', 'Ensemble', 'Ensemble_LOP', 'Ensemble_LOP_untrimmed', 'JHUAPL-Bucky', 'JHU_IDD-CovidSP']
    date_str = "-2021-09-14"
    vare_l_dist = []
    vare_u_dist = []
    e_u_dist = []
    e_l_dist = []

    fig,ax = plt.subplots(2,1,figsize=(6,6))

    tapp = 4
    for name in names:
        df = read_data(f"data/round9/{name}.csv")
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name='A', num_weeks=12, target_type='case', inc_cum='cum')
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name='C', num_weeks=12, target_type='case', inc_cum='cum')
        (vare_l_temp, vare_u_temp) = epsilon_distribution(scenarioX, scenarioY, quantiles, tapp,
                                                          epsilon_method="APPROXIMATE")
        (e_l_temp, e_u_temp) = epsilon_distribution(scenarioX, scenarioY, quantiles, tapp, epsilon_method="ESTIMATE")
        vare_l_dist.append(vare_l_temp)
        vare_u_dist.append(vare_u_temp)
        e_u_dist.append(e_u_temp)
        e_l_dist.append(e_l_temp)

    for i in range(len(names)):
        ax[0].scatter(np.arange(tapp), vare_l_dist[i], s=60, label=names[i])
    for i in range(len(names)):
        ax[1].scatter(np.arange(tapp), vare_u_dist[i], s=60, label=names[i])

    ax[1].set_xlabel(r"Weeks Ahead")
    ax[1].set_ylabel(r"$\varepsilon_u$")
    ax[0].set_xlabel(r"Weeks Ahead")
    ax[0].set_ylabel(r"$\varepsilon_l$")
    ax[0].set_ylim(-0.005, 0.08)
    ax[1].set_ylim(-0.005, 0.08)
    ax[1].legend(prop={'size': 8.5}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0.)
    ax[0].legend(prop={'size': 8.5}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0.)
    fig.savefig(r"plots/publish/fig2.pdf")
    fig2,ax2 = plt.subplots(2,1,figsize=(6,6))

    for i in range(len(names)):
        ax2[0].scatter(np.arange(tapp), e_l_dist[i], s=60, label=names[i])

    # plt.savefig(r"plots/round9/eudist.png")
    for i in range(len(names)):
        ax2[1].scatter(np.arange(tapp), e_u_dist[i], s=60, label=names[i])

    ax2[0].legend(prop={'size': 8.5}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
             ncol=2, mode="expand", borderaxespad=0.)
    ax2[0].set_xlabel(r"Weeks Ahead")
    ax2[0].set_ylabel(r"$\tilde{\epsilon}_l$")
    ax2[0].set_ylim(0, 0.16)
    ax2[1].legend(prop={'size': 8.5}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
             ncol=2, mode="expand", borderaxespad=0.)
    ax2[1].set_xlabel(r"Weeks Ahead")
    ax2[1].set_ylabel(r"$\tilde{\epsilon}_u$")
    ax2[1].set_ylim(0, 0.16)
    fig2.savefig(r"plots/publish/fig3.pdf")
    plt.show()

def extract_data(df, date_str, scenario_name, num_weeks, target_type, inc_cum):
    """

    extract_data(int,int) => andas.Dataframep

    Keyword arguments:
    scenario_id -> 0 if we are using the first scenario, 1 if we are looking to extract the second scenario
    """
    rslt = []
    scenario = scenario_name + date_str
    for target_wk in range(1,num_weeks+1):
        target = f"{str(target_wk)} wk ahead {inc_cum} {target_type.lower()}"
        rslt_curr = df[(df['scenario_id'] == scenario) & (df['target'] == target)
                       & (df['location'] == 'US') & (df['quantile'] != 0) & (df['quantile'] != 1)]
        rslt.append(round(rslt_curr['value'],-1))
    return np.array(rslt)

def epsilon_distribution(QX_all, QY_all, q, t_app, epsilon_method="ESTIMATE"):
    if epsilon_method.upper() == "ESTIMATE":
        # If the application time of scenario is immediate, then we can't estimate epsilon
        if t_app == 0:
            return 0

        # Vectors to contain possible values of e_u, e_l, then take max on these two vectors
        e_u = []
        e_l = []

        # Number of quantiles avaliable to X and Y
        num_quantiles = len(q)

        for t in range(t_app):

            # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
            QY = QY_all[t]
            QX = QX_all[t]
            e_u_curr = []
            e_l_curr = []

            # Max on all matchings of X and Y
            for i in range(1, num_quantiles):
                if QX[i] >= QY[i]:
                    QU = QX
                    QL = QY
                else:
                    QU = QY
                    QL = QX

                QL_i_prev = QL[i - 1] if i != 0 else 0
                QU_i_next = QU[i + 1] if i != num_quantiles - 1 else np.inf

                q_i = q[i]
                # argmax_{j:j<=i}
                alpha = last_max([k & (j <= (i - 1)) for j, k in enumerate(QU >= QL_i_prev)])
                # print(QX >= QY_i_prev)
                beta = first_min([k & (j >= (i + 1)) for j, k in enumerate(QL >= QU_i_next)])

                q_alpha = q[alpha]
                q_beta = q[beta]
                e_l_curr.append(q_i - q_alpha)
                e_u_curr.append(q_beta - q_i)

            e_l.append(round(max(e_l_curr),3))
            e_u.append(round(-min(e_u_curr),3))

        return e_l,e_u

    elif epsilon_method.upper() in ['APPROX','APPROXIMATE']:
        # If the application time of scenario is immediate, then we can't estimate epsilon
        if t_app == 0:
            return 0

        # Vectors to contain possible values of e_u, e_l, then take max on these two vectors
        e_u = []
        e_l = []


        # Number of quantiles avaliable to X and Y
        num_quantiles = len(q)

        for t in range(t_app):

            # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
            QY = QY_all[t]
            QX = QX_all[t]
            e_l_curr = []
            e_u_curr = []
            PCHIPX_INV = interpolate.interp1d(QX, q, fill_value=(0, 1), bounds_error=False)
            PCHIPY_INV = interpolate.interp1d(QY, q, fill_value=(0, 1), bounds_error=False)

            # Max on all matchings of X and Y
            for i in range(num_quantiles):
                if QX[i] >= QY[i]:
                    PCHIPU_INV = PCHIPX_INV
                    QU = QX
                    PCHIPL_INV = PCHIPY_INV
                    QL = QY
                else:
                    PCHIPU_INV = PCHIPY_INV
                    QU = QY
                    PCHIPL_INV = PCHIPX_INV
                    QL = QX

                e_l_curr.append(q[i] - PCHIPU_INV(QL[i]))
                e_u_curr.append(q[i] - PCHIPL_INV(QU[i]))

            e_l.append(round(max(e_l_curr), 4))
            e_u.append(round(-min(e_u_curr), 4))

        return e_l,e_u

    else:
        raise "epsilon_method must be either 'ESTIMATE' or 'APPROXIMATE'"

def epsilon(QX_all, QY_all, q, t_app, epsilon_method="ESTIMATE"):
    if epsilon_method.upper() == "ESTIMATE":
        return estimate_epsilon(QX_all, QY_all, t_app, q)
    elif epsilon_method.upper() in ["APPROX","APPROXIMATE"]:
        return approximate_epsilon(QX_all,QY_all,t_app,q)
    else:
        raise "epsilon_method must be either ESTIMATE or APPROXIMATE!"

def estimate_epsilon(QX_all, QY_all, t_app, q):
    """ Compute epsilon. By default, (x_i > y_j) for semi-informative models

    :param X: The values of X with the corresponding quantiles specified in q [0:[q1,q2,q3,q4],1:[],2:[],...,]
    :param Y: The quantiles of Y with the corresponding quantiles specified in q
    :param t: The number of steps ahead when t is applied (t_app - t_0)
    :param q: The list of quantiles avaliable
    :param model_type: ['uninformative','semi-informative','informative']
    :return: (e_u,e_l) if model_type is 'uninformative'; e is model_type is 'semi-informative'; 0 if model_type is 'informative'
    """
    # If the application time of scenario is immediate, then we can't estimate epsilon
    if t_app == 0:
        return 0

    # Vectors to contain possible values of e_u, e_l, then take max on these two vectors
    e_u = [0]
    e_l = [0]

    # Number of quantiles avaliable to X and Y
    num_quantiles = len(q)

    for t in range(t_app):

        # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
        QY = QY_all[t]
        QX = QX_all[t]

        # Max on all matchings of X and Y
        for i in range(1,num_quantiles):
            if QX[i] >= QY[i]:
                QU = QX
                QL = QY
            else:
                QU = QY
                QL = QX

            QL_i_prev = QL[i-1] if i!=0 else 0
            QU_i_next = QU[i+1] if i!=num_quantiles-1 else np.inf

            q_i = q[i]
            #argmax_{j:j<=i}
            alpha = last_max([k & (j <= (i - 1)) for j,k in enumerate(QU >= QL_i_prev)])
            # print(QX >= QY_i_prev)
            beta = first_min([k & (j >= (i + 1)) for j,k in enumerate(QL >= QU_i_next)])

            q_alpha = q[alpha]
            q_beta = q[beta]
            e_l.append(round(q_i - q_alpha,3))
            e_u.append(round(q_i - q_beta,3))

    return round(max(e_l),3),round(-min(e_u),3)

def approximate_epsilon(QX_all, QY_all, t_app, q):
    # If the application time of scenario is immediate, then we can't estimate epsilon
    if t_app == 0:
        return 0

    # Vectors to contain possible values of e_u, e_l, then take max on these two vectors
    e_u = [0]
    e_l = [0]

    # Number of quantiles avaliable to X and Y
    num_quantiles = len(q)

    for t in range(t_app):

        # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
        QY = QY_all[t]
        QX = QX_all[t]
        PCHIPX_INV = interpolate.interp1d(QX,q,fill_value=(0,1),bounds_error=False)
        PCHIPY_INV = interpolate.interp1d(QY,q,fill_value=(0,1),bounds_error=False)

        # Max on all matchings of X and Y
        for i in range(num_quantiles):
            if QX[i] >= QY[i]:
                PCHIPU_INV = PCHIPX_INV
                QU = QX
                PCHIPL_INV = PCHIPY_INV
                QL = QY
            else:
                PCHIPU_INV = PCHIPY_INV
                QU = QY
                PCHIPL_INV = PCHIPX_INV
                QL = QX

            e_l.append(round(q[i] - PCHIPU_INV(QL[i]),3))
            e_u.append(round(q[i] - PCHIPL_INV(QU[i]),3))

    return round(max(e_l),3),round(-min(e_u),3)

def DFRV_epsilon(QX_all, QY_all, q, e_l, e_u, alpha, dfrv_method="EXACT"):
    if dfrv_method.upper() == "EXACT":
        return DFRV_epsilon_exact(QX_all, QY_all, q, e_l, e_u, alpha)
    elif dfrv_method.upper() in ['APPROX', 'APPROXIMATE', "ESTIMATE"]:
        return DFRV_epsilon_approx(QX_all, QY_all, q, e_l, e_u, alpha)
    else:
        raise "method has to be either 'EXACT' or 'APPROXIMATE'!"

def DFRV_epsilon_exact(QX_all, QY_all, q, e_l, e_u, alpha):
    total_num_weeks = len(QX_all)
    rslt = []
    for t in range(total_num_weeks):
        Z_u = []
        Z_l = []
        QX = QX_all[t]
        QY = QY_all[t]
        for i in range(iteration_N):
            i_N = np.random.uniform(0, 1)
            q_l = last_max(q <= (i_N - e_l))
            q_u = first_min(q >= (i_N + e_u))
            Z_u.append(QX[q_u] - QY[q_l])
            Z_l.append(QX[q_l]-QY[q_u])
        Z_u = np.array(Z_u)
        Z_l = np.array(Z_l)

        # Calculate u and l based on the confidence interval given
        upper = 0.5 + alpha / 2
        lower = 0.5 - alpha / 2
        Z_u_upper = round(np.quantile(Z_u,upper),3)
        Z_l_lower = round(np.quantile(Z_l,lower),3)
        median_index_u = round(np.quantile(Z_u, 0.5), 3)
        median_index_l = round(np.quantile(Z_l, 0.5), 3)
        rslt.append([Z_l_lower,median_index_l,np.mean(Z_l),np.mean(Z_u),median_index_u,Z_u_upper])

    return np.array(rslt)

def DFRV_epsilon_approx(QX_all, QY_all, q, e_l, e_u, alpha):
    total_num_weeks = len(QX_all)
    rslt = []
    for t in range(total_num_weeks):
        Z_u = []
        Z_l = []
        QX = QX_all[t]
        QY = QY_all[t]
        PCHIPX = interpolate.PchipInterpolator(q,QX)
        PCHIPY = interpolate.PchipInterpolator(q,QY)
        for i in range(iteration_N):
            i_N = np.random.uniform(0, 1)
            q_l = i_N - e_l
            q_u = i_N + e_u
            Z_u.append(PCHIPX(q_u) - PCHIPY(q_l))
            Z_l.append(PCHIPX(q_l) - PCHIPY(q_u))
        Z_u = np.array(Z_u)
        Z_l = np.array(Z_l)

        # Calculate u and l based on the confidence interval given
        upper = 0.5 + alpha / 2
        lower = 0.5 - alpha / 2

        Z_u_upper = round(np.quantile(Z_u, upper), 3)
        Z_l_lower = round(np.quantile(Z_l, lower), 3)
        median_index_u = round(np.quantile(Z_u, 0.5), 3)
        median_index_l = round(np.quantile(Z_l, 0.5), 3)
        rslt.append([Z_l_lower,median_index_l,np.mean(Z_l),np.mean(Z_u),median_index_u,Z_u_upper])

    return np.array(rslt)

def DFRV(QX_all, QY_all, q, t_app, alpha, epsilon_method="ESTIMATE", dfrv_method="EXACT"):
    if epsilon_method.upper() == "ESTIMATE":
        e_l, e_u = estimate_epsilon(QX_all, QY_all, t_app, q)
        print(f"Estimated e_l,e_u = {e_l, e_u}")
    elif epsilon_method.upper() in ['APPROX', 'APPROXIMATE']:
        e_l, e_u = approximate_epsilon(QX_all, QY_all, t_app, q)
        print(f"Approximated e_l,e_u = {e_l, e_u}")
    else:
        raise "epsilon_method has to be either estimate or approximate!"

    return DFRV_epsilon(QX_all, QY_all, q, e_l, e_u, alpha=alpha, dfrv_method=dfrv_method)

def download_data(round=11):
    if round == 9:
        df_list = {
            # 'USC_SIkJalpha' : 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/USC-SIkJalpha/2021-12-19-USC-SIkJalpha.csv?raw=true',
            'Ensemble': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble/2021-09-11-Ensemble.csv?raw=true',
            'Ensemble_LOP': 'https://raw.githubusercontent.com/midas-network/covid19-scenario-modeling-hub/master/data-processed/Ensemble_LOP/2021-09-11-Ensemble_LOP.csv',
            'Ensemble_LOP_untrimmed': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP_untrimmed/2021-09-11-Ensemble_LOP_untrimmed.csv?raw=true',
            'JHUAPL-Bucky': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHUAPL-Bucky/2021-09-13-JHUAPL-Bucky.csv?raw=true',
            'JHU_IDD-CovidSP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHU_IDD-CovidSP/2021-09-14-JHU_IDD-CovidSP.csv?raw=true',
            'MOBS_NEU-GLEAM_COVID': 'https://raw.githubusercontent.com/midas-network/covid19-scenario-modeling-hub/master/data-processed/MOBS_NEU-GLEAM_COVID/2021-09-14-MOBS_NEU-GLEAM_COVID.csv',
        }
    elif round == 11:
        df_list = {
            'USC_SIkJalpha': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/USC-SIkJalpha/2021-12-19-USC-SIkJalpha.csv?raw=true',
            'Ensemble': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble/2021-12-18-Ensemble.csv?raw=true',
            'Ensemble_LOP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP/2021-12-18-Ensemble_LOP.csv?raw=true',
            'Ensemble_LOP_untrimmed': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP_untrimmed/2021-12-18-Ensemble_LOP_untrimmed.csv?raw=true',
            'JHU_IDD-CovidSP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHU_IDD-CovidSP/2021-12-21-JHU_IDD-CovidSP.csv?raw=true',
            'MOBS_NEU-GLEAM_COVID': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/MOBS_NEU-GLEAM_COVID/2021-12-17-MOBS_NEU-GLEAM_COVID.csv?raw=true',
            'NotreDame-FRED': 'https://github.com/midas-network/covid19-scenario-modeling-hub/raw/master/data-processed/NotreDame-FRED/2021-12-21-NotreDame-FRED.csv',
        }
    for k,v in df_list.items():
        print(k)
        pd.read_csv(v).to_csv(f"data/round{round}/{k}.csv")

def result2():
    date_str = "-2021-12-21"
    date_obj = datetime.strptime(date_str[1:],'%Y-%m-%d')

    curr_model = 'USC_SIkJalpha'
    df = pd.read_csv(f"data/round11/{curr_model}.csv").dropna()

    currel = 0.1
    curre_list = [(0,0),(0.05,0.05),(0.15,0.15)]
    curralpha_list = [0.5,0.75,0.85,0.95]
    dfrv_method = "EXACT"
    if dfrv_method.upper() == "EXACT":
        dfrv_method_label="Exact"
    else:
        dfrv_method_label="Approximate"
    tgt_type = 'Case'
    scenarios = ('B','A')
    ylim = (0,6*1e7)
    curralpha = 0.80
    fig,ax = plt.subplots(3,2,figsize=(10,14))
    for i,curre in enumerate(curre_list):
        currel = curre[0]
        curreu = curre[1]
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum')
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum')
        # result = DFRV(scenarioX, scenarioY, t_app=4, q=quantiles, alpha=0.75, epsilon_method="APPROXIMATE",
        #                dfrv_method="APPROX")
        result = DFRV_epsilon(scenarioX,scenarioY,quantiles,currel,curreu,alpha=curralpha,dfrv_method=dfrv_method)
        wks = [date_obj + timedelta(weeks=i) for i in list(range(1,result.shape[0] + 1))]
        ax[i,0].plot(wks, result[:, 1], label='$Z^L median$', linewidth=2)
        ax[i,0].plot(wks, result[:, 2], '--',label='$Z^L mean$', linewidth=2)
        ax[i,0].plot(wks, result[:, 3], '--',label='$Z^U mean$', linewidth=2)
        ax[i,0].plot(wks, result[:, 4], label='$Z^U median$', linewidth=2)
        ax[i,0].fill_between(wks, result[:, 0], result[:, 5], color='b', alpha=0.15)
        ax[i,0].legend(prop={'size': 12})
        ax[i,0].set_ylabel("DFRV", fontsize=13)
        ax[i,0].set_xlabel("Date",fontsize=13)
        ax[i,0].tick_params(labelrotation=20)
        ax[i,0].yaxis.set_tick_params(labelsize=13)
        ax[i,0].set_ylim(0,6*1e7)
        plt.ylim(*ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        title = f"{curr_model} Scenario {scenarios[0]} - {scenarios[1]} Round 11 {tgt_type} Culmulative"
        # ax[i].set_title(title,y=1.04)
        ax[i,0].annotate(f"$\epsilon_l = {currel}, \epsilon_u = {curreu}, " + r"\alpha" + f" = {curralpha}$ \nmethod={dfrv_method_label}",
                       xy=(0.35,0.05),xycoords='axes fraction',fontsize=14)
    dfrv_method = "APPROX"
    if dfrv_method.upper() == "EXACT":
        dfrv_method_label="Exact"
    else:
        dfrv_method_label="Approximate"
    for i,curre in enumerate(curre_list):
        currel = curre[0]
        curreu = curre[1]
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum')
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum')
        # result = DFRV(scenarioX, scenarioY, t_app=4, q=quantiles, alpha=0.75, epsilon_method="APPROXIMATE",
        #                dfrv_method="APPROX")
        result = DFRV_epsilon(scenarioX,scenarioY,quantiles,currel,curreu,alpha=curralpha,dfrv_method=dfrv_method)
        wks = [date_obj + timedelta(weeks=i) for i in list(range(1,result.shape[0] + 1))]
        ax[i,1].plot(wks, result[:, 1], label='$Z^L median$', linewidth=2)
        ax[i,1].plot(wks, result[:, 2], '--',label='$Z^L mean$', linewidth=2)
        ax[i,1].plot(wks, result[:, 3], '--',label='$Z^U mean$', linewidth=2)
        ax[i,1].plot(wks, result[:, 4], label='$Z^U median$', linewidth=2)
        ax[i,1].fill_between(wks, result[:, 0], result[:, 5], color='b', alpha=0.15)
        ax[i,1].legend(prop={'size': 12})
        ax[i,1].set_ylabel("DFRV", fontsize=13)
        ax[i,1].set_xlabel("Date",fontsize=13)
        ax[i,1].tick_params(labelrotation=20)
        ax[i,1].yaxis.set_tick_params(labelsize=13)
        ax[i,1].set_ylim(0,6*1e7)
        plt.ylim(*ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        title = f"{curr_model} Scenario {scenarios[0]} - {scenarios[1]} Round 11 {tgt_type} Culmulative"
        # ax[i].set_title(title,y=1.04)
        ax[i,1].annotate(f"$\epsilon_l = {currel}, \epsilon_u = {curreu}, " + r"\alpha" + f" = {curralpha}$ \nmethod={dfrv_method_label}",
                       xy=(0.35,0.05),xycoords='axes fraction',fontsize=14)
        # plt.savefig(f"plots/round11/dfrv/{dfrv_method}/{title}+{curreu}+{currel}+{curralpha}.png")]
    plt.suptitle(title,fontsize=18)
    plt.savefig(f"plots/publish/fig1.pdf")
    plt.show()

def result3():
    date_str = "-2021-12-21"
    date_obj = datetime.strptime(date_str[1:],'%Y-%m-%d')

    curr_model = 'Ensemble'
    df = pd.read_csv(f"data/round11/{curr_model}.csv").dropna()

    # currel = 0.1
    # curre_list = [(0,0),(0.05,0.05),(0.15,0.15)]
    currel = 0.1
    curreu = 0.1
    curralpha_list = [0.5,0.75,0.95]
    dfrv_method = "EXACT"
    if dfrv_method.upper() == "EXACT":
        dfrv_method_label="Exact"
    else:
        dfrv_method_label="Approximate"
    tgt_type = 'Case'
    scenarios = ('B','A')
    ylim = (0,2*1e7)

    fig,ax = plt.subplots(1,3,figsize=(15,4))
    for i,curralpha in enumerate(curralpha_list):
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum')
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum')
        result = DFRV_epsilon(scenarioX,scenarioY,quantiles,currel,curreu,alpha=curralpha,dfrv_method=dfrv_method)
        wks = [date_obj + timedelta(weeks=i) for i in list(range(1,result.shape[0] + 1))]
        ax[i].plot(wks, result[:, 1], label='$Z^L median$', linewidth=2)
        ax[i].plot(wks, result[:, 2], '--',label='$Z^L mean$', linewidth=2)
        ax[i].plot(wks, result[:, 3], '--',label='$Z^U mean$', linewidth=2)
        ax[i].plot(wks, result[:, 4], label='$Z^U median$', linewidth=2)
        ax[i].fill_between(wks, result[:, 0], result[:, 5], color='b', alpha=0.15)
        ax[i].legend(prop={'size': 12})
        ax[i].set_ylabel("DFRV", fontsize=13)
        ax[i].set_xlabel("Date",fontsize=13)
        ax[i].tick_params(labelrotation=20)
        ax[i].yaxis.set_tick_params(labelsize=13)
        ax[i].set_ylim(*ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        title = f"{curr_model} Scenario {scenarios[0]} - {scenarios[1]} Round 11 {tgt_type} Culmulative"
        # ax[i].set_title(title,y=1.04)
        ax[i].annotate(f"$\epsilon_l = {currel}, \epsilon_u = {curreu}, " + r"\alpha" + f" = {curralpha}$ \nmethod={dfrv_method_label}",
                       xy=(0.35,0.05),xycoords='axes fraction',fontsize=14)
        # plt.savefig(f"plots/round11/dfrv/{dfrv_method}/{title}+{curreu}+{currel}+{curralpha}.png")]
    plt.suptitle(title,fontsize=18)
    plt.savefig(f"plots/publish/rslt7.pdf")
    plt.show()


if __name__ == "__main__":
    # download_data(9)
    # download_data(11)
    # result2()
    # generate_epsilon_distribution()
    names = ['USC_SIkJalpha', 'Ensemble', 'Ensemble_LOP', 'Ensemble_LOP_untrimmed', 'JHUAPL-Bucky', 'MOBS_NEU-GLEAM_COVID']
    date_str = "-2021-09-14"
    date_obj = datetime.strptime(date_str[1:],'%Y-%m-%d')
    curr_model = 'Ensemble'
    currel = 0.1
    curreu = 0.1
    curralpha_list = [0.5,0.75,0.95]
    dfrv_method = "EXACT"
    if dfrv_method.upper() == "EXACT":
        dfrv_method_label="Exact"
    else:
        dfrv_method_label="Approximate"
    tgt_type = 'Case'
    scenarios = ('B','A')
    ylim = (0,2*1e7)
    e_l = []
    e_u = []
    vare_l = []
    vare_u = []
    epsilon_method_list = ["Estimate","Approximate"]
    fig,ax = plt.subplots(2,1,figsize=(6,6))
    for curr_model in names:
        df = pd.read_csv(f"data/round9/{curr_model}.csv").dropna()
        df.value = round(df.value,-4)
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum')
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum')
        curr_el, curr_eu = epsilon(scenarioX, scenarioY, quantiles, 4, epsilon_method="Estimate")
        print(curr_el,curr_eu)
        continue
        curr_varel, curr_vareu = epsilon(scenarioX, scenarioY, quantiles, 4, epsilon_method="Approximate")
        e_l.append(curr_el)
        e_u.append(curr_eu)
        vare_l.append(curr_varel)
        vare_u.append(curr_vareu)
    for j in range(len(names)):
        ax[0].plot(e_u[j],e_l[j],marker="o",label=names[j])
        ax[1].plot(vare_u[j],vare_l[j],marker="o",label=names[j])
    to_print = np.array([names,e_l,e_u,vare_l,vare_u])
    for i in range(len(names)):
        print(to_print[:,i])
    ax[0].set_ylim(-0.005,0.06)
    ax[1].set_ylim(-0.005,0.06)
    ax[0].legend(prop={'size': 8.5},bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
    ax[1].legend(prop={'size': 8.5},bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
    ax[0].set_ylabel(r"$\tilde{\epsilon}_u$")
    ax[1].set_ylabel(r"$\varepsilon_u$")
    ax[0].set_xlabel(r"$\tilde{\epsilon}_l$")
    ax[1].set_xlabel(r"$\varepsilon_l$")
    plt.savefig(f"plots/publish/fig4.pdf")
    plt.show()
