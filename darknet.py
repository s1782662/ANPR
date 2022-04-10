import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import numpy as np 


class EmptyLayer(nn.Module):
    
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):

    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors



def parse_cfg(cfg_file):
    
    '''
        Takes the configuration file 

        Returns a list of blocks. Each block describes a block in the neural
        network to be built. Block is represented as a dictionary in the list

    '''

    f = open(cfg_file,'r')
    lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks 


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of the block 
        # create a new module for the block 
        # append to module_list 

        if (x['type'] == 'convolutional'):
            # get the info about the layer 
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True 
            
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer 
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module('conv_{0}'.format(index),conv)

            # Add the Batch Norm Layer 
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index),bn)

            # Check the activation 
            # it is either linear or a leaky relu for yolo 
            if activation == 'leaky':
                activation = nn.LeakyReLU(0.1,inplace=True)
                module.add_module('Leaky_{0}'.format(index),activation)

        elif (x['type'] == 'upsample'):

            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2,mode='bilinear')
            module.add_module('upsample_{0}'.format(index),upsample)
        
        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            # Start of a route
            start = int(x['layers'][0])
            # end if provided
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{0}'.format(index),route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif (x['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index),shortcut)
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('DetectionLayer_{0}'.format(index),detection)

        module_list.append(module)
        prev_filters =  filters      
        output_filters.append(filters)
    
    return net_info,module_list

class Darknet(nn.Module):

    def __init__(self,cfg):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg)
        self.net_info,self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} # we cache the outputs for the route layer 

        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0]  = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2),1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                
                # get the input dimensions 
                inp_dim = int(self.net_info['height'])

                # get the number of classes 
                num_classes = int(module['classes'])

                # Transform 
                x = x.data 
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detecions,x),1)
        outputs[i] = x
        return detections 

    
    def load_weights(self,weightfile):
        # open the weights file 
        fp = open(weightfile,'rb')
            
        # The first five values are header information 
        # 1. Major version number 
        # 2. Minor version number 
        # 3. Subversion number 
        # 4,5. Images seen by the network (during Training)


        header = np.fromfile(fp,dtype=np.int32,count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp,dtype=np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            # if the module type is convolutional then load weights 
            # Otherwise ignore 

            if module_type == 'convolutional':
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of batch_normalization layer 
                    no_weights = bn.bias.numel()

                    # Load the weights 
                    bn_biases = torch.from_numpy(weights[ptr:ptr+no_weights])
                    ptr += no_weights

                    bn_weights = torch.from_numpy(weights[ptr:ptr+no_weights])
                    ptr += no_weights

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+no_weights])
                    ptr += no_weights
                    
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+no_weights])
                    ptr += no_weights

                    # cast the loaded weights into the dimension of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weights.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model 
                    bn.bias.data.copy_(bn_biases)
                    bn.weights.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # number of biases 
                    num_bias = conv.bias.numel()
                    
                    # load the weights 
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_bias])
                    ptr = ptr + num_bias 

                    # cast the loaded weights into dimension of model weights 
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the data to model 
                    conv.bias.data.copy_(conv_biases)

                # Load the weights for the convolutional layers
                num_weights = conv.weight.numel()

                # load the weights 
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])

                # cast the weights into the dimension of model weights 
                conv_weights = conv_weights.view_as(conv.weight.data)

                # copy the weights into the model 
                conv.weight.data.copy_(conv_weights)







        
model = Darknet('./yolov3.cfg')
model.load_weights('./yolov3.weights') 
