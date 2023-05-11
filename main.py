from lib import *
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta

def result2():
    date_str = "2021-12-21"
    date_obj = datetime.strptime(date_str,'%Y-%m-%d')

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
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum',q=config['quantiles'])
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum',q=config['quantiles'])
        # result = DFRV(scenarioX, scenarioY, t_app=4, q=quantiles, alpha=0.75, epsilon_method="APPROXIMATE",
        #                dfrv_method="APPROX")
        result = DFRV_epsilon(scenarioX,scenarioY,config['quantiles'],currel,curreu,alpha=curralpha,dfrv_method=dfrv_method)
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
        scenarioX = extract_data(df=df, date_str=date_str, scenario_name=scenarios[0], num_weeks=12, target_type='case', inc_cum='cum',q=config['quantiles'])
        scenarioY = extract_data(df=df, date_str=date_str, scenario_name=scenarios[1], num_weeks=12, target_type='case', inc_cum='cum',q=config['quantiles'])
        # result = DFRV(scenarioX, scenarioY, t_app=4, q=quantiles, alpha=0.75, epsilon_method="APPROXIMATE",
        #                dfrv_method="APPROX")
        result = DFRV_epsilon(scenarioX,scenarioY,config['quantiles'],currel,curreu,alpha=curralpha,dfrv_method=dfrv_method)
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


# if __name__ == "__main__":
    # result2()
    # states = ['US','California','Illinois','Texas','Florida','New York','Florida']
    # for round in [9,11,14,15]:
    #     meta = round_metadata(round)
    #     df_list = load_data(round)
    #     df_list_dict = extract_df_list(df_list,config['quantiles'])
    #     all_scenarios = meta['scenarios']
    #     for target_type in ['case','death','hosp']:
    #         for scenarios in all_scenarios:
    #             for model in tqdm(meta['models']):
    #                 for state in states:
    #
    #                     QX_all = extract_data_dict(df_dict=df_list_dict[model], date_str=meta['date_str'], scenario_name=scenarios[0],
    #                                           num_weeks=meta['num_weeks'], target_type=target_type,
    #                                           inc_cum='cum',q=config['quantiles'],state_id=state_to_id(state))
    #
    #                     QY_all = extract_data_dict(df_dict=df_list_dict[model], date_str=meta['date_str'], scenario_name=scenarios[1],
    #                                           num_weeks=meta['num_weeks'], target_type=target_type,
    #                                           inc_cum='cum',q=config['quantiles'],state_id=state_to_id(state))
    #
    #                     if QX_all.sum() == 0 or QY_all.sum() == 0:
    #                         continue
    #                     if QX_all.shape != QY_all.shape:
    #                         continue
    #
    #                     e_l,e_u = epsilon(QX_all, QY_all, meta['t_app'],config['quantiles'], epsilon_method="ESTIMATE")
    #                     # dfrv_rslt = DFRV_epsilon_exact(QX_all,QY_all,config['quantiles'],e_l,e_u,0.5)
    #                     dfrv_rslt = DFRV_naive(QX_all,QY_all,0.5)
    #                     plot_meta = {}
    #                     plot_meta['e_l'] = e_l
    #                     plot_meta['e_u'] = e_u
    #                     plot_meta['model'] = model
    #                     plot_meta['round'] = round
    #                     plot_meta['sceanrios'] = scenarios
    #                     plot_meta['target_type'] = target_type
    #                     plot_meta['date_str'] = meta['date_str']
    #                     plot_meta['state'] = state
    #                     plot_result(dfrv_rslt,config['quantiles'],plot_meta=plot_meta,save_fig=True,naive=True)

if __name__ == "__main__":
    # result2()
    states = ['US', 'California', 'Illinois', 'Texas', 'Florida', 'New York', 'Florida']
    # states = ['California']
    for round in [9,11,14,15]:
    # for round in [14]:
        meta = round_metadata(round)
        df_list = load_data(round)
        # df_list_dict = extract_df_list(df_list,config['quantiles'],include_list=['USC-SIkJalpha'])
        df_list_dict = extract_df_list(df_list,config['quantiles'])
        all_scenarios = meta['scenarios']
        for target_type in ['case']:
            for scenarios in all_scenarios:
                for model in meta['models']:
                    for state in states:
                        # if model != 'USC-SIkJalpha':
                        #     continue
                        QX_all = extract_data_dict(df_dict=df_list_dict[model], date_str=meta['date_str'], scenario_name=scenarios[0],
                                              num_weeks=meta['num_weeks'], target_type=target_type,
                                              inc_cum='cum',q=config['quantiles'],state_id=state_to_id(state))

                        QY_all = extract_data_dict(df_dict=df_list_dict[model], date_str=meta['date_str'], scenario_name=scenarios[1],
                                              num_weeks=meta['num_weeks'], target_type=target_type,
                                              inc_cum='cum',q=config['quantiles'],state_id=state_to_id(state))

                        if QX_all.sum() == 0 or QY_all.sum() == 0:
                            continue
                        if QX_all.shape != QY_all.shape:
                            continue

                        # e_l,e_u = epsilon(QX_all, QY_all, meta['t_app'],config['quantiles'], epsilon_method="ESTIMATE")
                        dev = np.round(estimate_dev(QX_all,QY_all,meta['t_app'],config['quantiles']),2)
                        dfrv_rslt = DFRV_epsilon_exact(QX_all,QY_all,config['quantiles'],dev,dev,0.95)
                        # dfrv_rslt = DFRV_naive(QX_all,QY_all,config['quantiles'],-1,0.5)
                        plot_meta = {}
                        # plot_meta['e_l'] = dev
                        # plot_meta['e_u'] = dev
                        plot_meta['dev'] = dev
                        plot_meta['model'] = model
                        plot_meta['round'] = round
                        plot_meta['sceanrios'] = scenarios
                        plot_meta['target_type'] = target_type
                        plot_meta['date_str'] = meta['date_str']
                        plot_meta['state'] = state
                        plot_result(dfrv_rslt,config['quantiles'],plot_meta=plot_meta,save_fig=True,naive=False)
                        # plot_result_naive(dfrv_rslt,config['quantiles'],plot_meta=plot_meta,save_fig=True)
