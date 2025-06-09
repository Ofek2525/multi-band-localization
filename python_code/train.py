
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
#from python_code import DEVICE
from estimation.net import SubSpaceNET
from estimation.multiband_net import Multi_Band_SubSpaceNET, Init_Single_Band_SubSpaceNET, Encoder_6k, Encoder_12k, Encoder_18k, Encoder_24k, Decoder
from estimation.loss import music_loss
from estimation.estimate import single_band_autocorrection
from channel.generate_channel import ues_rows_channel
from exp_params import seed, K, Nr, fc, BW,NS,input_power, main_band_idx, num_of_BSs
from utils.bands_manipulation import get_bands_from_conf
from utils.learning_rate_schedule import lr_schedule
from test import test_1sample
from channel.channel_loader import get_ue_info_by_row
from dir_definitions import RAYTRACING_DIR, ALLBSs_DIR
from utils.check_if_close import too_close
from plotting.learning_plot import plot_validation
import shutil
import datetime
import os
import random


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experment_name = "multi_bs"
load_path =r""
#learning_rate=0.0001/4
def train(learning_rate=15e-04, batch_size=20, data_samples=200000, ues_num=2, step=2000, alpha = 0.5, experment_name = "", load_path =""):
   
    #create experment dir
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    experment_name = date + "#" + experment_name + ("#" if experment_name !="" else "") + f"ues={ues_num},k={K},Nr={Nr},fc={fc},BW={BW},NS={NS},input_power={input_power}dBm"
    experment_dir =fr"z_exp/{experment_name}"
    os.makedirs(experment_dir, exist_ok=True)
    # Initialize model, loss function, and optimizer
    if len(fc) == 1:
        # enc = Encoder_6k().to(DEVICE)
        # dec = Decoder(32).to(DEVICE)
        # model = Init_Single_Band_SubSpaceNET(enc, dec).to(DEVICE)
        model = SubSpaceNET().to(DEVICE)
        shutil.copyfile(r"python_code/estimation/net.py", fr"{experment_dir}/net.py")
    elif len(fc) == 4:
        model = Multi_Band_SubSpaceNET().to(DEVICE)
        shutil.copyfile(r"python_code/estimation/multiband_net.py", fr"{experment_dir}/multiband_net.py")
    else:
        print("error with params")
    
    if load_path != "":
        model.load_state_dict(torch.load(load_path,weights_only=True))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
    bands = get_bands_from_conf(fc, Nr, K, BW)
    error_list = []

    main_band = bands[0] if len(fc) == 1 else bands[main_band_idx]

    # Training loop
    for batch_num in range(data_samples // batch_size):
        tmp_ues_num = np.random.choice([1,ues_num], p=[0,1])
        lr_schedule(optimizer,batch_num,step,alpha)
        # get data
        per_band_y, per_band_data = [], []
        BS_num = random.randint(1, num_of_BSs)
            ##### only for df len:
        band_freq_file_in_G = int(main_band.fc / 1000)
        csv_filename = rf"{ALLBSs_DIR}/bs_{BS_num}/train_{band_freq_file_in_G}Ghz.csv"
        df = pd.read_csv(csv_filename)
        num_rows = len(df)
            #####
        try:
            csv_rows_per_sample = [random.sample(range(1, num_rows), ues_num) for _ in range(batch_size)]
        except:
            print("##############", BS_num)
            continue
        # for each frequency sub-band
        for band in bands:
            # generate the channel
            ys, ues_data = ues_rows_channel(band, batch_size, tmp_ues_num, csv_rows_per_sample, BS_num=BS_num, state="train")
            per_band_y.append(ys)
            per_band_data.append(ues_data)
        
        # Forward
        multiband = "MULTI" if len(per_band_y) > 1 else "SINGLE"
        if multiband == 'SINGLE':
            RY = single_band_autocorrection(per_band_y[0])
            alternative_RY = model(RY)
        elif multiband == 'MULTI':
            per_band_RY = []
            for y in per_band_y:
                per_band_RY.append(single_band_autocorrection(y))
            alternative_RY = model(per_band_RY)
        #print(f"diff = {torch.sum(torch.abs(RY - RY.transpose(1,2).conj()))}")
        loss = music_loss(alternative_RY, [per_band_data[main_band_idx]], main_band, tmp_ues_num)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_num) % 1 == 0:
            add = "               " if tmp_ues_num != ues_num else "" 
            print(
                f'batch [{batch_num}/{data_samples // batch_size}], {add}Loss: {loss:.4f}')
        if batch_num % 100 == 0  and batch_num != 0 :
            torch.save(model.state_dict(), fr"{experment_dir}/model_params.pth")
            print("saved")
        if batch_num % 40 == 0:
            model.eval()
            mean_distance = test_1sample(model,np.array([[75, 75],[140,175]]), toPlot=True) /5
            mean_distance += test_1sample(model,np.array([[190,245],[75, 75]]), toPlot=True)/5
            mean_distance += test_1sample(model,np.array([[215,315], [320,430]]), toPlot=True)/5
            mean_distance += test_1sample(model,np.array([[215,315], [235,335]]), toPlot=True)/5
            mean_distance += test_1sample(model,np.array([[150,165], [170,205]]), toPlot=True)/5
            error_list.append(mean_distance)
            plot_validation(error_list,experment_dir)
            print(f"mean error ={mean_distance}") 
            test_1sample(model,np.array([[40, 5]]), toPlot=True)
            model.train()
    torch.save(model.state_dict(), fr"{experment_dir}/model_params.pth")
    return model


if __name__ == "__main__":
    model = train(experment_name=experment_name, load_path=load_path)
