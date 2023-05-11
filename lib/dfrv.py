from scipy.interpolate import PchipInterpolator,interp1d
from .util import *
from .config import *
import matplotlib.pyplot as plt
import warnings
import random

def epsilon(QX_all, QY_all, t_app, q, epsilon_method="ESTIMATE"):
    if epsilon_method.upper() == "ESTIMATE":
        return estimate_epsilon(QX_all, QY_all, t_app, q)
    elif epsilon_method.upper() in ["APPROX","APPROXIMATE"]:
        return approximate_epsilon(QX_all,QY_all,t_app,q)
    else:
        raise "epsilon_method must be either ESTIMATE or APPROXIMATE!"

def estimate_dev(QX_all, QY_all, t_app, q):
    # Updated Method
    if t_app == 0:
        return 0

    dev = [0]

    # Number of quantiles avaliable to X and Y
    num_quantiles = len(q)

    for t in range(t_app-1):

        # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
        QY = QY_all[t]
        QX = QX_all[t]
        if (len(QY) != len(QX)):
            warnings.warn("QY and QX must be the same size!")
            continue
        if len(QY) == 0:
            warnings.warn("QY and QX can't be empty!")
            continue

        f = np.min([np.max(QX) - np.min(QX),np.max(QY) - np.min(QY)])

        # Max on all matchings of X and Y
        for i in range(len(q)):
            dev.append(np.abs(QX[i] - QY[i])/np.abs(f))

    return np.max(dev)


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

    # Number of quantiles avaliable to X and Yr
    num_quantiles = len(q)

    for t in range(t_app-1):

        # The quantiles of X and Y at time t before t_app (Max on all t < t_app)
        QY = QY_all[t]
        QX = QX_all[t]
        if (len(QY) != len(QX)):
            warnings.warn("QY and QX must be the same size!")
            continue
        if len(QY) == 0:
            warnings.warn("QY and QX can't be empty!")
            continue

        # Max on all matchings of X and Y
        for i in range(1,num_quantiles):
            if QX[i] >= QY[i]:
                QU = QX
                QL = QY
            else:
                QU = QY
                QL = QX

            #argmax_{j:j<=i}
            alpha = last_max([cmp_rslt & (k < i) for k,cmp_rslt in enumerate(QU < QL[i])])
            beta = first_min([cmp_rslt & (k > i) for k,cmp_rslt in enumerate(QL > QU[i])])

            e_l.append(round(q[i] - q[alpha],3))
            e_u.append(round(q[i] - q[beta],3))
            # if max(e_l) >= 0.1 or max(e_u) >= 0.1:
                # print("Here")

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
        PCHIPX_INV = interp1d(QX,q,fill_value=(0,1),bounds_error=False)
        PCHIPY_INV = interp1d(QY,q,fill_value=(0,1),bounds_error=False)

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

def DFRV_epsilon_exact(QX_all, QY_all, q, e_l, e_u, alpha):
    q = np.array(q)
    total_num_weeks = len(QX_all)
    rslt = []
    for t in range(total_num_weeks):
        Z_u = []
        Z_l = []
        QX = QX_all[t]
        QY = QY_all[t]
        # print([(qx,qy) for qx,qy in zip(QX,QY)])
        for i in range(config['iteration_N']):
            i_N = np.random.uniform(0, 1)
            q_l = last_max(q <= (i_N - e_l))
            q_u = first_min(q >= (i_N + e_u))
            Z_u.append(QX[q_u] - QY[q_l])
            Z_l.append(QX[q_l] - QY[q_u])
        Z_u = np.array(Z_u)
        Z_l = np.array(Z_l)

        # Calculate u and l based on the confidence interval given
        upper = 0.5 + alpha / 2
        lower = 0.5 - alpha / 2
        Z_u_upper = round(np.quantile(Z_u,upper),3)
        Z_l_lower = round(np.quantile(Z_l,lower),3)
        median_u = round(np.median(Z_u),3)
        median_l = round(np.median(Z_l),3)
        rslt.append([Z_l_lower,median_l,np.mean(Z_l),np.mean(Z_u),median_u,Z_u_upper])

    return np.array(rslt)

def DFRV_epsilon_approx(QX_all, QY_all, q, e_l, e_u, alpha):
    total_num_weeks = len(QX_all)
    rslt = []
    for t in range(total_num_weeks):
        Z_u = []
        Z_l = []
        QX = QX_all[t]
        QY = QY_all[t]
        PCHIPX = PchipInterpolator(q,QX)
        PCHIPY = PchipInterpolator(q,QY)
        for i in range(config['iteration_N']):
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

def DFRV_epsilon(QX_all, QY_all, q, e_l, e_u, alpha, dfrv_method="EXACT"):
    if dfrv_method.upper() == "EXACT":
        return DFRV_epsilon_exact(QX_all, QY_all, q, e_l, e_u, alpha)
    elif dfrv_method.upper() in ['APPROX', 'APPROXIMATE', "ESTIMATE"]:
        return DFRV_epsilon_approx(QX_all, QY_all, q, e_l, e_u, alpha)
    else:
        raise "method has to be either 'EXACT' or 'APPROXIMATE'!"

def DFRV_naive(QX_all, QY_all,quantiles, rho, alpha):
    def estimate_std_from_quantiles(quantiles_arr):
        q1 = quantiles_arr[0.25]
        q3 = quantiles_arr[0.75]
        iqr = q3 - q1
        std_approx = iqr / 1.35
        return std_approx

    quantiles = np.array(quantiles)
    total_num_weeks = len(QX_all)
    rslt = []
    k = np.sqrt(1/(1-alpha))
    for t in range(total_num_weeks):
        QX = QX_all[t]
        QY = QY_all[t]

        QX_dict = {q:qx for q,qx in zip(quantiles,QX)}
        QY_dict = {q:qy for q,qy in zip(quantiles,QY)}

        ux = QX_dict[0.5]
        uy = QY_dict[0.5]
        sigx = estimate_std_from_quantiles(QX_dict)
        sigy = estimate_std_from_quantiles(QY_dict)

        sigmaz = np.sqrt(sigx**2 + sigy ** 2 - 2 * sigx * sigy * rho)

        b = k * sigmaz + ux - uy
        a = ux - uy - k * sigmaz

        rslt.append([a,b])

    return np.array(rslt)




