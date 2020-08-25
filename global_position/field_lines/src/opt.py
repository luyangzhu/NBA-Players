import os
import os.path as osp
import random
import yaml
import re

class Params():
    """
    Class that loads hyperparameters from a yaml file.
    """

    def __init__(self, yaml_path=None):
        if yaml_path is not None:
            with open(yaml_path) as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
                self.update_from_dict(params)
            # recursive get all keys, in 'a.b':c format
            all_keys = self.get_all_keys(all_keys={}, dic=self.dict, parent_key='')
            # replace placeholder in self.dict using all_keys
            self.replace_ph(all_keys)

    def update_from_dict(self, params):
        for k,v in params.items():
            if isinstance(v, dict):
                if k not in self.dict.keys():
                    self.dict[k] = Params()
                    self.dict[k].update_from_dict(v)
                else:
                    if isinstance(self.dict[k], Params):
                        self.dict[k].update_from_dict(v)
                    else:
                        raise ValueError('Can not update value using dict')
            else:
                self.dict[k] = v
    
    def get_all_keys(self, all_keys, dic, parent_key=''):
        ''' 
            get key value pair from current dict.
            In format: m1.m2.m3: v
        '''
        for k, v in dic.items():
            if isinstance(v, Params):
                new_parent_key = k if parent_key == '' else parent_key + '.'+ k
                all_keys = self.get_all_keys(all_keys, v.dict, new_parent_key)
            else:
                new_key = k if parent_key == '' else parent_key + '.'+ k
                all_keys[new_key] = v
        return all_keys
    
    def recursive_replace(self, cur_str, all_keys):
        '''
            Recursively Replace all placeholders in str.
            it mainly deals with following situation: a = ${b}, b = ${c}, c=2.
            We should get a=2,b=2,c=2, instead of a=${c}, b=2, c=2
        '''
        # find ph str of placeholder: start with '${', end with '}', including any character except $.
        ph_str_list = re.findall(r'\$\{[^\$]+\}', cur_str)
        # if no placeholder, directly return cur_str
        if len(ph_str_list) == 0:
            return cur_str
        # iterate over all placeholders in cur_str
        for ph_str in ph_str_list:
            # remove '${}' from ph_str to get placeholder
            ph = ph_str[2:-1] 
            # valid placeholder should be inside all_keys 
            assert ph in all_keys.keys()
            # get target string
            tgt_str = str(all_keys[ph])
            # if tgt_str contains placeholder, recursive replace placeholder
            if len(re.findall(r'\$\{[^\$]+\}', tgt_str)) > 0:
                tgt_str = self.recursive_replace(tgt_str, all_keys)
            # replace ph_str with tgt_str
            cur_str = cur_str.replace(ph_str, tgt_str)
        return cur_str

    def replace_ph(self, all_keys):
        '''
            Replace all placeholder in self.dict using all_keys
        '''
        for k,v in self.dict.items():
            if isinstance(v, Params):
                v.replace_ph(all_keys)
            elif isinstance(v, str):
                # recursively replace all placeholders in string v.
                v = self.recursive_replace(v, all_keys)
            self.dict[k] = v


    def save(self, yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(self.dict, f)
            
    def update(self, yaml_path):
        """Update parameters from yaml file"""
        with open(yaml_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.update_from_dict(params)
        # recursive get all keys, in 'a.b':c format
        all_keys = self.get_all_keys(all_keys={}, dic=self.dict, parent_key='')
        # replace placeholder in self.dict using all_keys
        self.replace_ph(all_keys)
    
    def print_params(self):
        all_keys = self.get_all_keys(all_keys={}, dic=self.dict, parent_key='')
        for k,v in all_keys.items():
            print(k, v)


    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
