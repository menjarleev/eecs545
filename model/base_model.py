import torch as t
import os

class BaseModel(t.nn.Module):
    """
    base module for all network, encapsulate save & load method
    """
    def initialize(self, ):
        self.is_train = False

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def get_curr_visuals(self):
        return self.input

    def get_curr_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids, save_dir):
        file_name = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, file_name)
        t.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and t.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        file_name = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, file_name)
        if not os.path.isfile(save_path):
            print('%s does not exist!' % save_path)
        else:
            try:
                network.load_state_dict(t.load(save_path))
            except:
                pretrained_dict = t.load(save_path)
                model_dict = network.state_dict()

                initialized = set()
                for k, v in pretrained_dict.items():
                    initialized.add(k.split('.')[0])
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers. Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers. The following are not initialized: ' % network_label)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate(self, epoch, model):
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in getattr(self, 'optimizer_' + model).param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def concat(self,  tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            tensors_cat =[]
            for i in range(len(tensors[0])):
                tensors_cat.append(self.concat([tensors[0][i], tensors[1][i]], dim=dim))
            return tensors_cat
        elif tensors[0] is not None:
            return tensors[0]
        else:
            return tensors[1]

