# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#%%

import numpy as np
import torch, os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__)))
import fourcastnetv2 as nvs


# # setting
# weight_path_global = './weight'

# # load_weight
# # Input
# area = [90, 0, -90, 360 - 0.25]
# grid = [0.25, 0.25]

# # setting
# n_lat = 720
# n_lon = 1440
# device = "cpu"
# cpu_num = 10



class FCNV2_model:
    def __init__(self, weight, device='cpu', cpu_num=10):
        self.weight = weight
        self.device = device
        self.cpu_num = cpu_num
        self.variables = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",\
            "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
            "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925","v1000",
            "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925","z1000",
            "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925","t1000", 
            "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

    
    def load_statistics(self):
        path = os.path.join(self.weight, "global_means.npy")
        global_means = np.load(path)
        # global_means = global_means[:, :channels, ...]
        global_means = global_means.astype(np.float32)

        path = os.path.join(self.weight, "global_stds.npy")
        global_stds = np.load(path)
        # global_stds = global_stds[:, :channels, ...]
        global_stds = global_stds.astype(np.float32)

        return global_means, global_stds
    
    
    def load_model(self):
        model = nvs.FourierNeuralOperatorNet()

        model.zero_grad()
        # Load weights
        checkpoint_file = os.path.join(self.weight, "weights.tar")
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

        weights = checkpoint["model_state"]
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # Make sure the parameter names are the same as the checkpoint
        # need to use strict = False to avoid this error message when
        # using sfno_76ch::
        # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # Missing key(s) in state_dict: "module.trans_down.weights",
        # "module.itrans_up.pct",
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        return model
    
    def initialize(self):
        if not os.path.exists(self.weight):
            raise FileNotFoundError(f"Weight file not found: {self.weight}") 
        self.model = self.load_model() 
        self.global_means, self.global_stds = self.load_statistics()
        torch.set_num_threads(self.cpu_num) if self.device=='cpu' else None
        print("FCNV2 Model and weights are loaded.")
    
    
    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        dims = data.shape[1]
        if reverse:
            new_data = data[:,:dims,...] * self.global_stds[:,:dims,...] + self.global_means[:,:dims,...]
        else:
            new_data = (data[:,:dims,...] - self.global_means[:,:dims,...]) / self.global_stds[:,:dims,...]
        return new_data
    
    def predict_one_step(self, input_data):
        # print(f'start  FCNV2 predict')
        all_fields_numpy = input_data[np.newaxis, :, :, :]
        all_fields_numpy = self.normalise(all_fields_numpy)
        input_iter = torch.from_numpy(all_fields_numpy).to(self.device)
        torch.set_grad_enabled(False)  
        output = self.model(input_iter)   
        output = self.normalise(output.cpu().numpy(),reverse=True)
        return output.squeeze()
    
    def predict_multiple_step(self, input_data, output_dir, fore_hour = 240):
        fore_hour = np.int_(fore_hour)
        # save output FCN_weather
        if not os.path.isdir(f'{output_dir}'):
            os.mkdir(f'{output_dir}')
        
        np.save(os.path.join(output_dir, f'output_weather_0h'), input_data.squeeze())
        
        print(f'start  FCNV2 predict')
        all_fields_numpy = input_data[np.newaxis, :, :, :]
        all_fields_numpy = self.normalise(all_fields_numpy)
        input_iter = torch.from_numpy(all_fields_numpy).to(self.device)
        torch.set_grad_enabled(False) 
        for time_index in range(np.int_(fore_hour/6)):
            output = self.model(input_iter)
            input_iter = output
            # reverse normalise
            # output = nan_extend(normalise(output.cpu().numpy(),global_means, global_stds, reverse=True))
            output = self.normalise(output.cpu().numpy(), reverse=True)
            
            np.save(os.path.join(output_dir, f'output_weather_{(time_index+1)*6}h'), output.squeeze())
            if np.mod(time_index,4)==3:
                print(f'finish {int(time_index/4)+1} days')
        print(f'Done')
        
        
    
    
