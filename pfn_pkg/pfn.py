# added soft link
# from typing import Callable
# from torch_geometric.nn import SAGEConv, GCNConv, APPNP
# print current working directory
import os
print(f"Woring directory: {os.getcwd()}")
if 'LMC' in os.getcwd():
    try:
        from lmc_autoscale.layers import MaskGCNConv
    except:
        print("from lmc_autoscale.layers import MaskGCNConv failed.")
import torch

def get_attr_recursively(obj, attr_name):
    '''
    example:
        get_attr_recursively(obj, ".lin.weight")
        get_attr_recursively(obj, ".bias")
    '''
    name_split = attr_name.split(".")[1:]
    for name in name_split:
        obj = getattr(obj, name)
    return obj

"""  
Relative distance:
    type1: (point_wise_distance_check=False)
    [default]
        return: [queue_size, scalar]
    type2: (point_wise_distance_check=True)
        return: [queue_size, same shape as input]
"""
'''
    relative distance to POINTWISE abs of current parameter value 
'''
def type1_distance_func_mean_pointwise_relative(x, x_queue):
    return torch.mean(
        torch.abs(x_queue - x)/torch.abs(x),
        dim=list(range(1, len(x.shape) ))
    )
def type1_distance_func_max_pointwise_relative(x, x_queue):
    def multi_max(a, num_dim):
        for i in range(num_dim-1):
            a = torch.max(a, dim=1)[0]
        return x
    return multi_max(
        torch.abs(x_queue - x)/torch.abs(x), 
        len(x.shape)
    )
def type2_distance_func_pointwise_relative(x, x_queue):
    return torch.abs(x_queue - x)/torch.abs(x)

'''
    relative distance to MEAN(abs) of current parameter value 
'''
def type1_distance_func_mean_overall_relative(x, x_queue):
    return torch.mean(
        torch.abs(x_queue - x)/torch.mean(torch.abs(x)),
        dim=list(range(1, len(x.shape) ))
    )
def type1_distance_func_max_overall_relative(x, x_queue):
    def multi_max(a, num_dim):
        for i in range(num_dim-1):
            a = torch.max(a, dim=1)[0]
        return x
    return multi_max(
        torch.abs(x_queue - x)/torch.mean(torch.abs(x)), 
        len(x.shape)
    )
def type2_distance_func_overall_relative(x, x_queue):
    return torch.abs(x_queue - x)/torch.mean(torch.abs(x))


class PFN(object):
    def __init__(self, num_nodes, num_layers, queue_size=8, queue_device=torch.device("cuda"),
                 grad_update_again=False, 
                 add_padding_queues=False, add_param_value=False,
                 use_padding_grad=False, use_grad_distances_check=False,
                 point_wise_distance_check=False,
                 distance_func=None,
                 check_grad_distances_lambda=0.1,
                 padding_beta=0.9, 
                 grad_inflation=False,
                 grad_add=False,
                 grad_normlized_add=False,
                 grad_normlized_add_alpha=0.9,
                 not_use_nid_access=False,
                 tmp_scale_up=1.0,
                 **kwargs):
        super().__init__()
        # list of handles for hooks
        self.hds = []
        # init self.meta
        self.meta = []

        # get the queue size
        self.queue_size = queue_size
        # get the queue device
        self.queue_device = queue_device

        # init current queue position
        self.growing_current_queue_pos = -1
        self.current_queue_pos = -1

        # name, to be changed, "queue_pos_for_nodes"
        # self.queue_pos_for_nodes = [torch.zeros((num_nodes), dtype=torch.int32, device=queue_device)-1 for _ in range(num_layers)]
        self.num_nodes  = num_nodes
        self.num_layers = num_layers

        # grad calculation options
        self.add_padding_queues = add_padding_queues
        self.add_param_value = add_param_value
        self.grad_update_again = grad_update_again
        assert (not grad_update_again) if add_padding_queues else True, "grad_update_again should be False when add_padding_queues is True. because the padding update should use non stall grad."
        self.directly_use_padding_grad = use_padding_grad
        assert add_padding_queues if use_padding_grad else True, "directly_use_padding_grad need to have mantained padding queues."
        
        self.use_grad_distances_check = use_grad_distances_check
        self.point_wise_distance_check = point_wise_distance_check
        if distance_func is not None:
            self.distance_func = distance_func
        elif use_grad_distances_check:
            assert ValueError("distance_func should be provided when use_grad_distances_check is True.")

        # hyperparameters
        self.padding_beta = padding_beta
        self.grad_normlized_add_alpha = grad_normlized_add_alpha

        self.check_grad_distances_lambda = check_grad_distances_lambda

        # experimental grad calculation options
        self.grad_inflation = grad_inflation
        self.grad_add = grad_add
        self.grad_normlized_add = grad_normlized_add
        self.not_use_nid_access = not_use_nid_access
        self.tmp_scale_up = tmp_scale_up

        # at least choose one of the grad calculation options
        assert (grad_inflation or grad_add or grad_normlized_add), "At least choose one of the grad calculation options."


    def generate_meta(self, layer_list, layer_name_list, layer_num_list, cfg=None):
        '''
        Generate meta info for layers in layer_list.
        Note:
            layer_num_list 
        '''
        assert len(layer_list) == len(layer_name_list), "layer_list and layer_name_list should have the same length."
        assert len(layer_list) == len(layer_num_list), "layer_list and layer_num_list should have the same length."
        assert all(0 <= layer_num < self.num_layers for layer_num in layer_num_list), "layer_num_list should be in [0, num_layers)."

        self.layer_name_list = layer_name_list ###
        self.layer_num_list = layer_num_list ###

        if len(self.meta) > 0:
            raise ValueError("Meta info already exists.")
        for layer in layer_list:
            self.meta.append(self.get_history_meta_one_layer(layer, cfg))

        self.queue_pos_for_nodes = {layer_num_list[i]: {para_meta[0]: torch.zeros((self.num_nodes), dtype=torch.int32, device=self.queue_device)-1 for para_meta in layer_meta } for i,layer_meta in enumerate(self.meta)}
        print(f"[INFO] {[ {para_meta[0]: 1 for para_meta in layer_meta } for layer_meta in self.meta]}")

    def init_all_history_queues(self):
        '''
        Init all history queues for layers in layer_list.
        '''
        assert len(self.meta) > 0, "Meta info not generated."

        for i, layer_meta in enumerate(self.meta):
            for parameter_meta in layer_meta:
                queue_name = self.layer_name_list[i]+parameter_meta[0].replace(".", "_")
                setattr(self, queue_name+"_grad", torch.zeros((self.queue_size, *parameter_meta[1]), device=self.queue_device))
                if self.add_padding_queues: # mantain a exponential moving average of the grad
                    setattr(self, queue_name+"_grad_padding", torch.zeros(parameter_meta[1], device=self.queue_device))
                if self.add_param_value: # mantain the value of the parameter, same shape with the grad saving queue
                    setattr(self, queue_name+"_value", torch.zeros((self.queue_size, *parameter_meta[1]), device=self.queue_device))

    @staticmethod
    def get_history_meta_one_layer(layer, cfg=None):
        '''
        Return the meta info of the layer: 
            (name, size, obj, ...)
        Observe reset_parameters() to know what parameters are there.
        '''
        one_by_one = False
        if one_by_one:
            if layer.__class__ is SAGEConv:
                '''
                SAGEConv has TWO 'torch.nn.parameter.Parameter':
                    .lin.weight&bias(optional)(default: project=False)(bias is always True)
                    .lin_l.weight
                    .lin_l.bias(optional)(default: bias=True)
                    .lin_r.weight(optional)(default: root_weight=True)
                '''
                meta = [
                    (".lin_l.weight", layer.lin_l.weight.size(), get_attr_recursively(layer, ".lin_l.weight")), 
                ]
                if hasattr(layer.lin_l, "bias") and layer.lin_l.bias is not None: # bias
                    meta.append((".lin_l.bias", layer.lin_l.bias.size(), get_attr_recursively(layer, ".lin_l.bias")))
                if hasattr(layer, "lin") and layer.lin is not None: # project
                    meta.append((".lin.weight", layer.lin.weight.size(), get_attr_recursively(layer, ".lin.weight")))
                    meta.append((".lin.bias", layer.lin.bias.size(), get_attr_recursively(layer, ".lin.bias")))
                if hasattr(layer, "lin_r") and layer.lin_r is not None: # root_weight
                    meta.append((".lin_r.weight", layer.lin_r.weight.size(), get_attr_recursively(layer, ".lin_r.weight")))

            elif layer.__class__ is GCNConv:
                '''
                GAS use this conv.
                GCNConv has TWO 'torch.nn.parameter.Parameter': 
                    .lin.weight
                    .bias(optional)(default: bias=True)
                '''
                meta = [
                    (".lin.weight", layer.lin.weight.size(), get_attr_recursively(layer, ".lin.weight")), 
                ]
                if hasattr(layer, "bias") and layer.bias is not None:
                    meta.append((".bias", layer.bias.size(), get_attr_recursively(layer, ".bias")))


        else:
            # raise NotImplementedError(f"layer {layer.__class__} not supported yet.")
            meta = []
            for name, param in layer.named_parameters():
                meta.append(('.'+name, param.size(), param))
        

        return meta

    def detached_grad_correction(self, n_id_list):
        self.growing_current_queue_pos = self.growing_current_queue_pos + 1  # the first trigger give 0
        self.current_queue_pos = self.growing_current_queue_pos % self.queue_size # 0,..,queue_size-1

        def conv_hook_generator(layer_name, layer_num, param_name, param_obj, grad_update_again=False, grad_add=False):
            def backward_hook(grad):
                # # print original grad shape
                # print(f">original grad shape: {grad.shape}")
                # import pdb;pdb.set_trace()
                # print(f"[DEBUG] before self.calculate_weighted_sum_of_grad_hist_part")
                # # print layer_name, layer_num, param_name, n_id_list[layer_num], grad in [DEBUG]
                # print(f"[DEBUG] layer_name: {layer_name}")
                # print(f"[DEBUG] layer_num: {layer_num}")
                # print(f"[DEBUG] param_name: {param_name}")
                # print(f"[DEBUG] n_id_list[layer_num]: {n_id_list[layer_num]}")
                # print(f"[DEBUG] grad: {grad}")
                """ calculate new_grad from current grad and history """
                new_grad = self.calculate_weighted_sum_of_grad_hist_part(
                    layer_name, layer_num, param_name, n_id_list[layer_num], grad)

                """ update history queue by saving current grad or new_grad(stalled but low bias) """
                if not grad_update_again:
                    self.update_queues(
                        grad, layer_name, layer_num, param_name, param_obj, n_id_list[layer_num])
                else:
                    self.update_queues(
                        new_grad, layer_name, layer_num, param_name, param_obj, n_id_list[layer_num])

                # print(f"[DEBUG] ending")
                # print(f"[DEBUG] new_grad: {new_grad}")
                # # print new grad shape
                # print(f"new grad shape: {grad.shape}")
                return new_grad
            return backward_hook

        for i, layer_meta in enumerate(self.meta):
            for parameter_meta in layer_meta:
                # print(f"[DEBUG] parameter_meta: {parameter_meta}")
                # print(f"[DEBUG] parameter_meta[2]: {parameter_meta[2]}")
                # print(f"[DEBUG] parameter_meta[2].grad.data: {parameter_meta[2].grad.data}")
                # # print self.layer_name_list[i], self.layer_num_list[i], parameter_meta[0], parameter_meta[2], self.grad_update_again, self.grad_add in [DEBUG] 
                # print(f"[DEBUG] self.layer_name_list[i]: {self.layer_name_list[i]}")
                # print(f"[DEBUG] self.layer_num_list[i]: {self.layer_num_list[i]}")
                # print(f"[DEBUG] parameter_meta[0]: {parameter_meta[0]}")
                # print(f"[DEBUG] parameter_meta[2]: {parameter_meta[2]}")
                # print(f"[DEBUG] self.grad_update_again: {self.grad_update_again}")
                # print(f"[DEBUG] self.grad_add: {self.grad_add}")
                
                parameter_meta[2].grad.data = conv_hook_generator(
                    self.layer_name_list[i], self.layer_num_list[i], parameter_meta[0], parameter_meta[2], self.grad_update_again, self.grad_add
                )(parameter_meta[2].grad.data)

    def update_queues(self, grad_to_save, layer_name, layer_num, param_name, param_obj, nid):
        """
        update the queues: _grad, _grad_padding, _value
        """
        # import pdb;pdb.set_trace()
        queue_name = layer_name+param_name.replace(".", "_")
        queue_name_grad = queue_name+"_grad"
        queue_name_grad_padding = queue_name+"_grad_padding"
        queue_name_value = queue_name+"_value"

        eval("self."+queue_name_grad)[self.current_queue_pos] = grad_to_save
        if self.add_padding_queues:
            setattr(self, queue_name_grad_padding, getattr(self, queue_name_grad_padding) * (1 - self.padding_beta) + \
                grad_to_save * self.padding_beta)
        if self.add_param_value:
            eval("self."+queue_name_value)[self.current_queue_pos] = param_obj.detach()

        self.queue_pos_for_nodes[layer_num][param_name][nid] = self.growing_current_queue_pos

    def calculate_weighted_sum_of_grad_hist_part(self, layer_name, layer_num, param_name, nid=None, grad=None):
        """
        calculate the weighted sum of gradient history part.
        """
        # import pdb;pdb.set_trace()
        queue_name = layer_name+param_name.replace(".", "_")
        queue_name_grad = queue_name+"_grad"
        queue_name_grad_padding = queue_name+"_grad_padding"
        queue_name_value = queue_name+"_value"

        if self.directly_use_padding_grad:
            return getattr(self, queue_name_grad_padding)
        
        # the possible values in queue_pos_for_nodes[layer] are growing_current_queue_pos
            # (-1,) 0, 1, ..., queue_size-1, ..., current 'growing_current_queue_pos'.
            # we only keep the latest queue_size caches. 
        # count the number of all possible values except -1 in queue_pos_for_nodes[layer] 
        if self.not_use_nid_access:
            nid = None
        if nid is None:
            # if not access historical grad by nid, then we just use those within the queue_size
            effective_queue_pos = self.queue_pos_for_nodes[layer_num][param_name][self.queue_pos_for_nodes[layer_num][param_name] > max(self.growing_current_queue_pos-1 - self.queue_size, -1)] \
                % self.queue_size
            effective_queue_pos = effective_queue_pos.unique(return_counts=True)
            # print(f"effective_queue_pos: {effective_queue_pos}")
        else:
            within_queue_mask = self.queue_pos_for_nodes[layer_num][param_name] > max(self.growing_current_queue_pos-1 - self.queue_size, -1)
            effective_queue_pos = self.queue_pos_for_nodes[layer_num][param_name][nid][ within_queue_mask[nid] ] \
                % self.queue_size
            effective_queue_pos = effective_queue_pos.unique(return_counts=True)
            # print(f"effective_queue_pos: {effective_queue_pos}")
            # print(f"layer_name: {layer_name}, layer_num: {layer_num}, param_name: {param_name}, nid: {nid}, effective_queue_pos: {effective_queue_pos}")
            # import pdb;pdb.set_trace()
        # tuple convert to list
        effective_queue_pos = list(effective_queue_pos)
        # effective_queue_pos[0] is the idx, convert to long
        queue_idx  = effective_queue_pos[0].long()
        # print(f"{layer_name}, {layer_num} , effective_queue_pos: {effective_queue_pos}")


        if not self.point_wise_distance_check:
            if self.use_grad_distances_check:
                # cal distance for items in queue
                value_queue = getattr(self, queue_name_value)
                distance_queue = self.distance_func(value_queue[self.current_queue_pos], value_queue)
                # 用 self.current_queue_pos 索引可能不对，要输入当前的 weigh 的值  

                # check distance 
                grad_distances_check_mask = distance_queue < self.check_grad_distances_lambda
                # filter out item with small distances
                queue_idx = queue_idx[grad_distances_check_mask] # unique 之后的 idx
                effective_queue_pos[1] = effective_queue_pos[1][grad_distances_check_mask] # unique 对应后的 count

            if self.grad_inflation:
                # 归一化地求和但是 scale up
                #     effective_queue_pos[1].sum()  /  effective_queue_pos[1][queue_idx == self.current_queue_pos]
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad = (getattr(self, queue_name_grad)[queue_idx] * \
                    (effective_queue_pos[1]/effective_queue_pos[1][queue_idx == self.current_queue_pos]).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)
                return old_grad+grad
            elif self.grad_add:
                # new_gard= (getattr(self, queue_name_grad)[queue_idx]).sum(dim=0)
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad= (getattr(self, queue_name_grad)[queue_idx] * \
                    (effective_queue_pos[1]/effective_queue_pos[1].sum()).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)
                return old_grad*self.tmp_scale_up+grad
            elif self.grad_normlized_add:
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad= (getattr(self, queue_name_grad)[queue_idx] * \
                    (effective_queue_pos[1]/effective_queue_pos[1].sum()).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)

                return old_grad+self.grad_normlized_add_alpha*(grad-old_grad)
            
        else: # use point_wise_distance_check
            raise NotImplementedError("seem redundant, not implemented yet.")
            assert self.point_wise_distance_check, "point_wise_distance_check should be True."
            value_queue = getattr(self, queue_name_value)
            distance_queue = self.distance_func(value_queue[self.current_queue_pos], value_queue)
            grad_distances_check_mask = distance_queue < self.check_grad_distances_lambda
            # use the grad when W with small distance from now. 

            if self.grad_inflation:
                raise ValueError("grad_inflation is not supported in point_wise_distance_check.")
            elif self.grad_add:
                return (getattr(self, queue_name_grad)*grad_distances_check_mask).sum(dim=0)
            else:
                raise NotImplementedError("Complexity is high, not implemented yet, need calculate number of nodes for each point differently.")

    def __memsize__(self, unit="MB"):
        # add the memory size of all torch tensors member variables
        mem_size = 0
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                mem_size += value.element_size() * value.nelement()
        # add the memory size of tensors in queue_pos_for_nodes
        for i in range(len(self.queue_pos_for_nodes)):
            for para_name in self.queue_pos_for_nodes[i].keys():
                mem_size += self.queue_pos_for_nodes[i][para_name].element_size() * self.queue_pos_for_nodes[i][para_name].nelement()
        # convert to MB
        if unit == "MB":
            mem_size = mem_size / (1024**2)
        elif unit == "KB":
            mem_size = mem_size / 1024
        elif unit == "GB":
            mem_size = mem_size / (1024**3)
        return mem_size